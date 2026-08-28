#!/usr/bin/env python3
"""
Phase 3: Short-GT Reward 计算与训练文件生成

从 round0_merged.jsonl（或 rollouts_rank*.jsonl）读取 2048 条轨迹，
离线计算两种 Reward，生成两套 ELBO-PG 训练文件：
  - train_reward_a.jsonl  (Reward A: 纯 type-aware short-GT F1)
  - train_reward_b.jsonl  (Reward B: A - 0.2·I(no_answer) - 0.1·I(invalid))

每条训练记录格式（与 make_agentic_data.py Plan A 兼容）：
{
  "prompt":      "<|im_start|>system...<|im_end|>...<|im_start|>assistant\n",
  "completion":  "...",
  "advantage":   0.732,     # group z-score normalized
  "reward":      0.85,      # raw reward
  "question_id": "hotpot_train_00796",
  "rollout_id":  3,
  "reward_type": "A",
  "answer_type": "entity",
  "valid_format": true,
  "n_policy_tokens": 512
}

用法:
  python compute_rewards.py \
    --input  Agentic/output/round0_rollouts/round0_merged.jsonl \
    --output Agentic/output/round0_rollouts/ \
    [--min_group 4]  # 允许少于8条的group参与训练（默认8）
"""

import argparse
import json
import re
import statistics
import string
from collections import defaultdict
from pathlib import Path


# ── SYSTEM_PROMPT / USER_TEMPLATE（与 collect_round0_rollouts.py 完全一致）──────
# 注意：必须与 collect_round0_rollouts.py 保持完全相同，
# 否则 build_prompt() 生成的 prompt 将与 rollout 时的 prompt 不一致。
SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins."""

USER_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {{
    "type": "object",
    "properties": {{
      "query": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }}
    }},
    "required": [
      "query"
    ]
    }}
}}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question.

Example response:
<think>
thinking process here
</think>
<tool_call>
{{"name": "search", "arguments": {{"query": ["query string 1", "query string 2"]}}}}
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
User: {question}"""

# tool_response 在 completion 里的包裹格式（env tokens，与 collect 脚本一致）
TOOL_LEFT  = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<tool_response>"
TOOL_RIGHT = "</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# ── 答案归一化 ────────────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())

def bool_map(s: str) -> str:
    return {"True": "yes", "False": "no"}.get(str(s), str(s))

def extract_yes_no(s: str):
    """从预测文本提取 yes/no，返回 None 表示未检出。"""
    s = s.strip().lower()
    if re.match(r"^(yes[,\s]|yes$)", s): return "yes"
    if re.match(r"^(no[,\s]|no$)",  s): return "no"
    if re.search(r"\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b", s): return "no"
    if re.search(r"\bthe same\b|\bboth\b|\byes\b", s): return "yes"
    return None

def token_f1(prediction: str, reference: str) -> float:
    if not prediction or not str(prediction).strip():
        return 0.0
    pred_n = normalize_text(bool_map(str(prediction)))
    ref_n  = normalize_text(bool_map(str(reference)))
    pred_tok = set(pred_n.split())
    ref_tok  = set(ref_n.split())
    if not ref_tok:
        return 0.0
    common = pred_tok & ref_tok
    if not common:
        return 0.0
    precision = len(common) / max(len(pred_tok), 1)
    recall    = len(common) / len(ref_tok)
    return 2 * precision * recall / (precision + recall + 1e-9)

def parse_number(s: str):
    """尝试从字符串解析浮点数。"""
    try:
        return float(re.sub(r"[,]", "", s.strip()))
    except Exception:
        return None

def type_aware_f1(prediction: str, gold: dict) -> float:
    """
    Type-aware F1:
      - yes_no   → binary exact match (0/1)
      - number   → numeric fuzzy (within 1% relative tolerance, else token F1)
      - date     → year match + fallback token F1
      - entity / list / other_short_text → best-over-aliases token F1
    """
    if not prediction:
        return 0.0

    answer_type = gold.get("answer_type", "entity")
    short_answers = gold.get("short_answers") or []
    aliases = gold.get("aliases") or short_answers

    # 合并所有参考答案
    all_refs = list({a for a in (short_answers + aliases) if a})
    if not all_refs:
        return 0.0

    pred_str = str(prediction).strip()

    if answer_type == "yes_no":
        pred_yn = extract_yes_no(pred_str)
        for ref in all_refs:
            ref_yn = extract_yes_no(ref)
            if pred_yn is not None and ref_yn is not None and pred_yn == ref_yn:
                return 1.0
        # fallback: extract_yes_no failed → token F1
        return max(token_f1(pred_str, ref) for ref in all_refs)

    if answer_type == "number":
        pred_num = parse_number(pred_str)
        if pred_num is not None:
            for ref in all_refs:
                ref_num = parse_number(ref)
                if ref_num is not None:
                    if ref_num == 0:
                        if abs(pred_num) < 1e-6:
                            return 1.0
                    elif abs(pred_num - ref_num) / abs(ref_num) <= 0.01:
                        return 1.0
        # fallback: token F1
        return max(token_f1(pred_str, ref) for ref in all_refs)

    if answer_type == "date":
        # 先试年份严格匹配
        year_m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", pred_str)
        for ref in all_refs:
            ref_year_m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", ref)
            if year_m and ref_year_m and year_m.group() == ref_year_m.group():
                return 1.0
        # fallback: token F1
        return max(token_f1(pred_str, ref) for ref in all_refs)

    # entity / list / other_short_text → best over aliases
    return max(token_f1(pred_str, ref) for ref in all_refs)


# ── Reward 计算 ───────────────────────────────────────────────────────────────
_SPECIAL_TOK_RE = re.compile(r"<\|[^|>]+\|>|<[a-z_]+>")

def clean_answer(ans: str) -> str:
    """去除 LLaDA 生成时可能残留的特殊 token（如 <|eot_id|> <|endoftext|>）。"""
    if not ans:
        return ans
    return _SPECIAL_TOK_RE.sub("", ans).strip()

def compute_reward_a(record: dict) -> float:
    """Reward A = type-aware short-GT F1"""
    pred_raw = record.get("final_answer")
    pred = clean_answer(pred_raw) if pred_raw else None
    gold = record.get("gold", {})
    return round(type_aware_f1(pred, gold), 4)

def compute_reward_b(record: dict, reward_a: float) -> float:
    """
    Reward B = A - 0.2·I(no_answer) - 0.1·I(invalid_format)
    no_answer      = final_answer is None（模型没有给出任何答案）
    invalid_format = terminated_by != "final_answer"（没有通过 <answer> 格式给出答案）
    """
    r = reward_a
    if record.get("final_answer") is None:
        r -= 0.2   # no answer penalty
    if not record.get("valid_format"):
        r -= 0.1   # invalid format penalty（格式问题，哪怕有内容）
    return round(max(r, -1.0), 4)   # clamp 下界


# ── Trajectory → prompt + completion ─────────────────────────────────────────
def build_prompt(question: str) -> str:
    """构建 prompt，使用 Llama3 canonical 格式（等价于 apply_chat_template 输出）。
    USER_TEMPLATE 使用 str.format() 展开：{{/}} → {/}（JSON 里的大括号），{question} → 实际问题。
    """
    user_content = USER_TEMPLATE.format(question=question)
    return (
        f"<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def turns_to_completion(turns: list) -> str:
    """将 structured turns 重建为 completion 字符串。

    优先使用 turn["raw_text"]（collect_round0_rollouts.py v2+ 保存的原始生成文本），
    这样可以保证 completion 的 token sequence 与 rollout 时完全一致。

    若无 raw_text（旧版 rollout 数据），则从结构化字段精确重建原始格式（格式完全已知）：
      - tool_call JSON：{"name": "search", "arguments": {"query": [...]}}
        （USER_TEMPLATE example response 规定的格式，有 "arguments" 层，是一一映射）
      - answer：<|box_start|>\nanswer\n<|box_end|>
        （原始模型用 stop_string "<|box_end|>" 截断，box_start/box_end 是原始格式）
      - tool_response（env tokens）：<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<tool_response>...\n</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

    【重要】completion 必须与 rollout 时 policy 实际生成的 token sequence 尽量一致，
    才能保证 ELBO-PG 优化的是真正执行过的 trajectory：
        -A(τ_raw) · log π_θ(τ_raw)
    而不是对不同 token sequence 的 log-prob。
    """
    parts = []
    for turn in turns:
        # ── 优先路径：用保存的原始生成文本 ──────────────────────────────────────
        raw_text = turn.get("raw_text")
        if raw_text is not None:
            # raw_text 是 policy 生成的纯文本（截断到 stop_string 之后）
            # tool_response 需要作为 env tokens 追加
            parts.append(raw_text)
            tool_resp = turn.get("tool_response")
            if tool_resp is not None:
                # 这个 turn 是 tool_call turn，需要追加 env token（tool_response）
                # TOOL_LEFT 以 <|eot_id|> 开头，同时关闭 assistant turn
                parts.append(
                    f"{TOOL_LEFT}\n{tool_resp}\n{TOOL_RIGHT}"
                )
            continue

        # ── 无 raw_text：从结构化字段精确重建（格式已知，是一一映射）──────────────
        think = turn.get("think", "")
        content_parts = []
        if think:
            content_parts.append(f"<think>\n{think}\n</think>\n")

        if "final_answer" in turn:
            # 原始格式：<|box_start|>answer<|box_end|>
            # force_answer 时模型在已有 <|box_start|> 前缀下生成，结果拼成完整格式
            ans = turn["final_answer"] or ""
            content_parts.append(f"<|box_start|>\n{ans}\n<|box_end|>")
            parts.append("".join(content_parts))

        elif turn.get("tool_call") and turn.get("parse_success"):
            # 原始格式：<tool_call>{"name":"search","arguments":{"query":[...]}}</tool_call>
            query = turn["tool_call"].get("query", [])
            tc_str = json.dumps(
                {"name": "search", "arguments": {"query": query}},
                ensure_ascii=False
            )
            content_parts.append(f"<tool_call>\n{tc_str}\n</tool_call>")
            parts.append("".join(content_parts))
            # 追加 env tokens（tool_response）
            tool_resp = turn.get("tool_response") or ""
            # TOOL_LEFT 以 <|eot_id|> 开头，同时关闭 assistant turn
            parts.append(
                f"{TOOL_LEFT}\n{tool_resp}\n{TOOL_RIGHT}"
            )
            continue  # tool_call turn 已在上面加了 <|eot_id|>（via TOOL_LEFT）

        else:
            # no_action / parse_fail：把能还原的内容放进去
            if turn.get("tool_call") and not turn.get("parse_success"):
                query = turn["tool_call"].get("query", [])
                tc_str = json.dumps(
                    {"name": "search", "arguments": {"query": query}},
                    ensure_ascii=False
                )
                content_parts.append(f"<tool_call>\n{tc_str}\n</tool_call>")
            parts.append("".join(content_parts))

        parts.append("<|eot_id|>")

    return "".join(parts)


# ── Group z-score normalization ───────────────────────────────────────────────
def group_normalize(rewards: list) -> list:
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,
                        help="round0_merged.jsonl 或逗号分隔的多个 rank 文件")
    parser.add_argument("--output",    required=True,
                        help="输出目录")
    parser.add_argument("--min_group", type=int, default=8,
                        help="最小 group 大小（默认8，即要求完整8条）")
    parser.add_argument("--max_all_correct_frac", type=float, default=0.95,
                        help="超过此比例答对则认为 group 无梯度（过滤）")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取所有 records ──────────────────────────────────────────────────────
    records = []
    for fpath in args.input.split(","):
        fpath = fpath.strip()
        if not fpath:
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    print(f"读取 records: {len(records)}")

    # ── 按 question_id 分组 ───────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in records:
        groups[r["question_id"]].append(r)

    print(f"问题数: {len(groups)}")
    group_sizes = [len(v) for v in groups.values()]
    complete = sum(1 for s in group_sizes if s == 8)
    print(f"完整8条的问题: {complete}/{len(groups)}")
    print(f"group 大小分布: min={min(group_sizes)}, max={max(group_sizes)}, mean={sum(group_sizes)/len(group_sizes):.1f}")

    # ── 统计 ──────────────────────────────────────────────────────────────────
    all_stats = {
        "total_records": len(records),
        "total_questions": len(groups),
        "complete_groups": complete,
        "skipped_small_group": 0,
        "skipped_zero_variance_a": 0,
        "skipped_zero_variance_b": 0,
        "kept_a": 0,
        "kept_b": 0,
    }

    # ── 逐 group 处理 ─────────────────────────────────────────────────────────
    out_a = open(out_dir / "train_reward_a.jsonl", "w")
    out_b = open(out_dir / "train_reward_b.jsonl", "w")

    for qid, rollouts in groups.items():
        if len(rollouts) < args.min_group:
            all_stats["skipped_small_group"] += 1
            continue

        question = rollouts[0].get("question", "")
        prompt   = build_prompt(question)

        # 计算每条的 reward
        rewards_a = [compute_reward_a(r) for r in rollouts]
        rewards_b = [compute_reward_b(r, ra) for r, ra in zip(rollouts, rewards_a)]

        # z-score
        advantages_a = group_normalize(rewards_a)
        advantages_b = group_normalize(rewards_b)

        # 过滤 zero-variance groups
        has_var_a = any(abs(a) > 1e-6 for a in advantages_a)
        has_var_b = any(abs(a) > 1e-6 for a in advantages_b)
        if not has_var_a:
            all_stats["skipped_zero_variance_a"] += 1
        if not has_var_b:
            all_stats["skipped_zero_variance_b"] += 1

        for rollout, ra, rb, adv_a, adv_b in zip(
                rollouts, rewards_a, rewards_b, advantages_a, advantages_b):

            turns      = rollout.get("turns", [])
            completion = turns_to_completion(turns)
            if not completion.strip():
                continue   # 空 completion 跳过

            base_rec = {
                "prompt":          prompt,
                "completion":      completion,
                "question_id":     rollout.get("question_id"),
                "rollout_id":      rollout.get("rollout_id"),
                "answer_type":     (rollout.get("gold") or {}).get("answer_type", "entity"),
                "valid_format":    rollout.get("valid_format", False),
                "final_answer":    rollout.get("final_answer"),
                "n_policy_tokens": rollout.get("n_policy_tokens", 0),
                "n_env_tokens":    rollout.get("n_env_tokens", 0),
                "terminated_by":   rollout.get("terminated_by"),
                "num_searches":    rollout.get("num_searches", 0),
            }

            if has_var_a:
                rec_a = {**base_rec, "reward": ra, "advantage": round(adv_a, 4), "reward_type": "A"}
                out_a.write(json.dumps(rec_a, ensure_ascii=False) + "\n")
                all_stats["kept_a"] += 1

            if has_var_b:
                rec_b = {**base_rec, "reward": rb, "advantage": round(adv_b, 4), "reward_type": "B"}
                out_b.write(json.dumps(rec_b, ensure_ascii=False) + "\n")
                all_stats["kept_b"] += 1

    out_a.close()
    out_b.close()

    # ── 报告 ──────────────────────────────────────────────────────────────────
    print("\n=== Phase 3 Reward 计算完成 ===")
    print(f"  总 records         : {all_stats['total_records']}")
    print(f"  总 questions        : {all_stats['total_questions']}")
    print(f"  完整8条 groups      : {all_stats['complete_groups']}")
    print(f"  跳过（group<{args.min_group}）  : {all_stats['skipped_small_group']}")
    print(f"  Reward A zero-var  : {all_stats['skipped_zero_variance_a']}")
    print(f"  Reward B zero-var  : {all_stats['skipped_zero_variance_b']}")
    print(f"  train_reward_a     : {all_stats['kept_a']} records → {out_dir}/train_reward_a.jsonl")
    print(f"  train_reward_b     : {all_stats['kept_b']} records → {out_dir}/train_reward_b.jsonl")

    # ── 分析 reward 分布 ──────────────────────────────────────────────────────
    # 重读文件进行统计
    for tag, fname in [("A", "train_reward_a.jsonl"), ("B", "train_reward_b.jsonl")]:
        fpath = out_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = [json.loads(l) for l in f if l.strip()]
        if not data:
            continue
        rewards    = [d["reward"]    for d in data]
        advantages = [d["advantage"] for d in data]
        n_valid    = sum(1 for d in data if d.get("valid_format"))
        n_no_ans   = sum(1 for d in data if d.get("final_answer") is None)
        mean_r     = sum(rewards) / len(rewards)
        f1_gt0 = sum(1 for r in rewards if r > 0) / len(rewards)
        f1_gt5 = sum(1 for r in rewards if r > 0.5) / len(rewards)
        f1_gt8 = sum(1 for r in rewards if r > 0.8) / len(rewards)
        by_type = defaultdict(list)
        for d in data:
            by_type[d.get("answer_type", "?")].append(d["reward"])
        print(f"\n--- Reward {tag} 分布 ({len(data)} records) ---")
        print(f"  mean F1       : {mean_r:.3f}")
        print(f"  F1>0 / >0.5 / >0.8 : {f1_gt0:.1%} / {f1_gt5:.1%} / {f1_gt8:.1%}")
        print(f"  answer rate   : {n_valid}/{len(data)} = {n_valid/len(data):.1%}")
        print(f"  no-answer     : {n_no_ans}/{len(data)} = {n_no_ans/len(data):.1%}")
        print(f"  by type:")
        for atype, rs in sorted(by_type.items()):
            print(f"    {atype:20s}: n={len(rs):4d}  mean={sum(rs)/len(rs):.3f}")

    print("\n完成！")


if __name__ == "__main__":
    main()
