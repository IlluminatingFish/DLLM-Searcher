#!/usr/bin/env python3
"""
把多次 rollout 结果转成 DPO (prompt, chosen, rejected) 对。

流程：
  1. 读取 collect_dpo_rollouts.py 产生的 rank*.jsonl
  2. 对每道题的多次 rollout 用 LLM judge 判断对错
  3. 配对：最好的正确 rollout (chosen) + 最差的错误 rollout (rejected)
  4. 输出 DPO 训练数据 dpo_pairs.jsonl

用法:
  python my_eval/make_dpo_pairs.py \
    --rollout_dir output/dpo_rollouts \
    --output dLLM_trainer/VRPO/data/x0pred_dpo_pairs.jsonl
"""

import argparse, json, os, re, time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ── LLM Judge ─────────────────────────────────────────────────────────────────
def _load_config():
    for p in ["/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json"]:
        if os.path.exists(p):
            return json.load(open(p))
    return {}

_cfg   = _load_config()
_key   = os.getenv("OPENAI_API_KEY") or _cfg.get("judge_api_key") or _cfg.get("openai_api_key", "")
_base  = os.getenv("OPENAI_API_BASE") or _cfg.get("judge_api_base") or "https://api.openai.com/v1"
_model = os.getenv("JUDGE_MODEL") or _cfg.get("judge_model", "gpt-4o")
_client = OpenAI(api_key=_key, base_url=_base)

JUDGE_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.
Respond with ONLY True if the prediction is correct and ONLY False otherwise.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}"""

def llm_judge(question, reference, prediction):
    if not prediction or not prediction.strip():
        return False
    prompt = JUDGE_PROMPT.format(question=question, reference=reference,
                                 prediction=prediction or "(no answer)")
    for _ in range(2):
        try:
            resp = _client.chat.completions.create(
                model=_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=5
            )
            return resp.choices[0].message.content.strip().lower() == "true"
        except Exception:
            time.sleep(1)
    return False

# ── Prompt builder (同 run_llada_eval.py) ─────────────────────────────────────
SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins."""

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
    from transformers import AutoTokenizer
    # 不依赖tokenizer，直接用字符串拼接
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT_TEMPLATE + question}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def messages_to_completion(messages):
    """把 messages 列表拼成 completion 字符串（DPO 训练格式）。"""
    parts = []
    for m in messages[2:]:   # 跳过 system + 第一个 user
        if m["role"] == "assistant":
            parts.append(m["content"].strip())
        elif m["role"] == "user":
            content = m["content"].strip()
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True, help="collect_dpo_rollouts 输出目录")
    parser.add_argument("--output",      required=True, help="DPO pairs 输出 jsonl")
    parser.add_argument("--max_workers", type=int, default=20, help="LLM judge 并发数")
    parser.add_argument("--min_pairs",   type=int, default=1, help="每题最少配对数")
    args = parser.parse_args()

    # 读取所有 rollout 文件
    rollout_dir = Path(args.rollout_dir)
    all_rollouts = []
    for f in sorted(rollout_dir.glob("rank*.jsonl")):
        for l in open(f):
            try: all_rollouts.append(json.loads(l))
            except: pass
    print(f"共读取 {len(all_rollouts)} 条 rollout")

    # 按题目分组
    by_question = defaultdict(list)
    for r in all_rollouts:
        by_question[r["question"]].append(r)

    print(f"共 {len(by_question)} 道题，并行 LLM judge...")

    # 并行 judge
    def judge_one(r):
        r["correct"] = llm_judge(r["question"], r["answer"], r["prediction"])
        return r

    judged = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(judge_one, r): r for r in all_rollouts}
        for f in tqdm(futures, total=len(all_rollouts), desc="LLM judge"):
            judged.append(f.result())

    # 重新分组
    by_question_judged = defaultdict(list)
    for r in judged:
        by_question_judged[r["question"]].append(r)

    # 配对
    pairs = []
    no_pair = 0
    for question, rollouts in by_question_judged.items():
        correct   = [r for r in rollouts if r["correct"] and r["prediction"]]
        incorrect = [r for r in rollouts if not r["correct"]]

        if not correct or not incorrect:
            no_pair += 1
            continue

        prompt = build_prompt(question)

        # 每题生成多对（最多 min(len(correct), len(incorrect)) 对）
        n_pairs = max(args.min_pairs, min(len(correct), len(incorrect)))
        for i in range(n_pairs):
            c = correct[i % len(correct)]
            r = incorrect[i % len(incorrect)]
            pairs.append({
                "prompt":   prompt,
                "chosen":   messages_to_completion(c["messages"]),
                "rejected": messages_to_completion(r["messages"]),
            })

    print(f"\n配对结果: {len(pairs)} 对  (无法配对: {no_pair} 题)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"DPO 数据已保存: {out_path}")

if __name__ == "__main__":
    main()
