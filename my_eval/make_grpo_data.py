#!/usr/bin/env python3
"""
把 collect_grpo_rollouts.py 产生的 rollout 文件转成 GRPO 训练数据。

流程:
  1. 读取 rank*.jsonl
  2. 按题目分组
  3. 过滤掉组内 reward 全相同的题（无梯度）
  4. 组内归一化 → advantages
  5. 输出 {prompt, completion, advantage} 格式

用法:
  python my_eval/make_grpo_data.py \
    --rollout_dir output/rl/grpo/rollouts \
    --output      dLLM_trainer/VRPO/data/grpo_train.jsonl
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

# ── Prompt builder（与 collect_grpo_rollouts.py 保持一致）─────────────────────
SYSTEM_PROMPT = (
    "You are a Web Information Seeking Master. Your task is to thoroughly seek "
    "the internet for information and provide accurate answers to questions. "
    "No matter how complex the query, you will not give up until you find the "
    "corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n"
    "1. **Persistent Actions for Answers**: You will engage in many interactions, "
    "delving deeply into the topic to explore all possible aspects until a satisfactory "
    "answer is found.\n\n"
    "2. **Repeated Verification**: Before presenting a Final Answer, you will "
    "**cross-check** and **validate the information** you've gathered to confirm its "
    "accuracy and reliability.\n\n"
    "3. **Attention to Detail**: You will carefully analyze each information source "
    "to ensure that all data is current, relevant, and from credible origins."
)

USER_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }
    },
    "required": ["query"]
  }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question.

Example response:
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query string 1", "query string 2"]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think>
thinking process here
</think>
<|box_start|>
answer here
<|box_end|>
The assistant must strictly abide by the above format.
User: """


def build_prompt(question):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{USER_PROMPT_TEMPLATE + question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def messages_to_completion(messages):
    """把 messages[2:] 拼成 completion 字符串（含 tool_response）。"""
    parts = []
    for m in messages[2:]:
        if m["role"] == "assistant":
            parts.append(m["content"].strip())
        elif m["role"] == "user":
            content = m["content"].strip()
            parts.append(
                f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            )
    return "".join(parts)


def group_normalize(rewards):
    """组内 z-score 归一化，返回 advantages 列表。"""
    import statistics
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--min_group",   type=int, default=2,
                        help="一题至少有多少条 rollout 才纳入训练")
    parser.add_argument("--max_correct_frac", type=float, default=1.0,
                        help="跳过正确率超过此分数的题（如 0.875 跳过 ≥7/8 正确的题，专注困难题）")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    all_rollouts = []
    for f in sorted(rollout_dir.glob("rank*.jsonl")):
        for line in open(f):
            try: all_rollouts.append(json.loads(line))
            except: pass
    print(f"读取 rollout: {len(all_rollouts)} 条")

    # 按题目分组
    by_question = defaultdict(list)
    for r in all_rollouts:
        if r.get("messages"):  # 跳过 error 记录
            by_question[r["question"]].append(r)

    print(f"共 {len(by_question)} 道题")

    skipped_no_var = 0
    skipped_small  = 0
    skipped_easy   = 0
    total_samples  = 0

    with open(args.output, "w") as out_f:
        for question, rollouts in by_question.items():
            if len(rollouts) < args.min_group:
                skipped_small += 1
                continue

            rewards = [r.get("reward", 0.0) for r in rollouts]

            # 过滤：组内 reward 全相同时 advantage=0，无梯度，跳过
            if len(set(rewards)) == 1:
                skipped_no_var += 1
                continue

            # 过滤：正确率过高的"简单题"（信号噪声大、advantage 方差小）
            if args.max_correct_frac < 1.0:
                correct_frac = sum(1 for r in rewards if r > 0) / len(rewards)
                if correct_frac > args.max_correct_frac:
                    skipped_easy += 1
                    continue

            advantages = group_normalize(rewards)
            prompt = build_prompt(question)

            for rollout, adv in zip(rollouts, advantages):
                completion = messages_to_completion(rollout["messages"])
                if not completion.strip():
                    continue

                record = {
                    "prompt":     prompt,
                    "completion": completion,
                    "advantage":  round(adv, 6),
                    "reward":     rollout.get("reward", 0.0),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"跳过（组太小）: {skipped_small}")
    print(f"跳过（reward 无方差）: {skipped_no_var}")
    print(f"跳过（题太简单 >{args.max_correct_frac:.0%}）: {skipped_easy}")
    print(f"输出训练样本: {total_samples} 条 → {args.output}")

    # 简单统计
    rewards_all = [r.get("reward", 0.0) for rollouts in by_question.values() for r in rollouts]
    correct_rate = sum(1 for r in rewards_all if r > 0) / max(len(rewards_all), 1)
    print(f"整体正确率: {correct_rate:.1%} ({sum(1 for r in rewards_all if r > 0)}/{len(rewards_all)})")


if __name__ == "__main__":
    main()
