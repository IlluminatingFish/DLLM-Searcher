#!/usr/bin/env python3
"""
诊断脚本 ①：离线分析 Round 1 rollout 数据（纯 CPU）

解答的核心问题：
  98.8% "正确率" 到底是什么？是答案正确还是只是 reward > 0？
  GRPO 组内 advantage 是否真的有足够信号？
  reward 各分量（correctness / efficiency / format_bonus）贡献比例？

用法：
  python diagnose_rollouts.py
"""

import json, re, string, os
from pathlib import Path
from collections import defaultdict

ROOT = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher")
ROLLOUT_DIR = ROOT / "dLLM_trainer/Agentic/output/plan_a/rollouts/round1/plan_a"
OUT_FILE    = ROOT / "dLLM_trainer/Agentic/output/plan_a/logs/diagnose_rollouts.txt"

# ── 复制 collect_agentic_rollouts.py 里的 F1 函数（保持一致） ──────────────
def _normalize(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())

def _bool_map(s: str) -> str:
    return {"True": "yes", "False": "no"}.get(s, s)

def _first_sentence(s: str) -> str:
    s = s.strip()
    for sep in [". ", ".\n", "\n\n", "\n"]:
        idx = s.find(sep)
        if idx > 10:
            return s[: idx + 1]
    return s[:200]

def _yes_no(s: str):
    s = s.strip().lower()
    if re.match(r"^(yes[,\s]|yes$)", s): return "yes"
    if re.match(r"^(no[,\s]|no$)", s):   return "no"
    if re.search(r"\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b", s): return "no"
    if re.search(r"\bthe same\b|\bboth\b|\byes\b", s): return "yes"
    return None

def token_f1(prediction: str, ground_truth: str) -> float:
    if not prediction or not str(prediction).strip(): return 0.0
    gt_key    = _normalize(_bool_map(_first_sentence(str(ground_truth))))
    pred_norm = _normalize(_bool_map(str(prediction)))
    gt_tok    = set(gt_key.split())
    pred_tok  = set(pred_norm.split())
    if not gt_tok: return 0.0
    common = gt_tok & pred_tok
    if not common: return 0.0
    p = len(common) / max(len(pred_tok), 1)
    r = len(common) / len(gt_tok)
    return 2 * p * r / (p + r + 1e-9)

def norm_exact_match(prediction: str, ground_truth: str) -> bool:
    if not prediction: return False
    gt_key    = _normalize(_bool_map(_first_sentence(str(ground_truth))))
    pred_norm = _normalize(_bool_map(str(prediction)))
    return pred_norm == gt_key

def base_correctness(prediction, ground_truth):
    """Returns (score, is_yes_no)"""
    if not prediction: return 0.0, False
    gt_yn   = _yes_no(str(ground_truth))
    pred_yn = _yes_no(str(prediction))
    if gt_yn is not None and pred_yn is not None:
        return (1.0 if gt_yn == pred_yn else 0.0), True
    return token_f1(prediction, ground_truth), False

def has_valid_tool_call(messages: list) -> bool:
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            try:
                start = content.find("<tool_call>")
                end   = content.find("</tool_call>")
                if start != -1 and end != -1:
                    payload = json.loads(content[start+11:end].strip())
                    if "query" in payload.get("arguments", {}):
                        return True
            except Exception:
                pass
    return False

def reconstruct_reward(record: dict):
    """重新计算各分量（不依赖 saved reward）"""
    prediction   = record.get("prediction", "")
    ground_truth = record.get("answer", "")
    n_searches   = record.get("n_searches", 0)
    messages     = record.get("messages", [])

    r_corr, is_yn = base_correctness(prediction, ground_truth)
    efficiency    = max(0.5, 1.0 - 0.1 * max(0, n_searches - 2))
    base = r_corr * efficiency if r_corr > 0 else 0.0
    fmt  = 0.1 if has_valid_tool_call(messages) else 0.0
    total = min(1.0, base + fmt)

    f1_raw = token_f1(prediction, ground_truth) if not is_yn else r_corr
    em     = norm_exact_match(prediction, ground_truth)

    return {
        "r_corr":      round(r_corr,  4),
        "f1_raw":      round(f1_raw,  4),
        "em":          em,
        "efficiency":  round(efficiency, 4),
        "format_bonus":round(fmt, 4),
        "reward_recon":round(total, 4),
        "reward_saved":round(record.get("reward", 0.0), 4),
        "is_yn":       is_yn,
        "n_searches":  n_searches,
    }

# ── 加载数据 ──────────────────────────────────────────────────────────────────
def load_rollouts(rollout_dir: Path):
    records = []
    for f in sorted(rollout_dir.glob("rank*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def pct(n, total):
    return f"{n}/{total} ({100*n/max(total,1):.1f}%)"

def hist_str(values, bins=10, width=30):
    if not values: return "(empty)"
    lo, hi = min(values), max(values)
    if lo == hi: return f"all = {lo:.3f}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    mx = max(counts)
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / max(mx, 1))
        lo_b = lo + i * step
        lines.append(f"  [{lo_b:5.2f},{lo_b+step:5.2f}) {bar} {c}")
    return "\n".join(lines)

# ── 主分析 ────────────────────────────────────────────────────────────────────
def main():
    print(f"加载 rollout 数据: {ROLLOUT_DIR}")
    records = load_rollouts(ROLLOUT_DIR)
    print(f"总 rollout 数: {len(records)}")

    # 分析每条 rollout
    stats = [reconstruct_reward(r) for r in records]

    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr("=" * 70)
    pr("诊断报告：Plan A Round 1 Rollout 分析")
    pr("=" * 70)
    pr(f"总 rollout 数: {len(records)}")

    # ── 1. 原始 "正确率" 复现 ─────────────────────────────────────────────────
    pr()
    pr("── 1. 原始统计（run_agentic_loop.sh 里的计算方式）")
    reward_gt0 = sum(1 for s in stats if s["reward_saved"] > 0)
    pr(f"  reward_saved > 0  (即'正确')  : {pct(reward_gt0, len(stats))}")
    pr(f"  → 这是 run_agentic_loop.sh 报告的 '98.8%' 来源")

    # ── 2. 真实正确率 ─────────────────────────────────────────────────────────
    pr()
    pr("── 2. 真实答案质量（重新计算 F1/EM）")
    f1_gt0   = sum(1 for s in stats if s["r_corr"] > 0)
    f1_gt05  = sum(1 for s in stats if s["r_corr"] > 0.5)
    f1_gt08  = sum(1 for s in stats if s["r_corr"] > 0.8)
    em_count = sum(1 for s in stats if s["em"])
    yn_count = sum(1 for s in stats if s["is_yn"])

    pr(f"  r_corr > 0   (F1 > 0)        : {pct(f1_gt0,  len(stats))}")
    pr(f"  r_corr > 0.5 (F1 > 0.5)      : {pct(f1_gt05, len(stats))}")
    pr(f"  r_corr > 0.8 (F1 > 0.8)      : {pct(f1_gt08, len(stats))}")
    pr(f"  Exact Match (norm.)           : {pct(em_count, len(stats))}")
    pr(f"  Yes/No 类型问题               : {pct(yn_count, len(stats))}")

    f1_values = [s["r_corr"] for s in stats]
    pr(f"\n  F1 分布:")
    pr(hist_str(f1_values))
    pr(f"  F1 均值={sum(f1_values)/len(f1_values):.4f}  中位={sorted(f1_values)[len(f1_values)//2]:.4f}")

    # ── 3. Format bonus 的影响 ────────────────────────────────────────────────
    pr()
    pr("── 3. Format Bonus 影响分析")
    fmt_only = sum(1 for s in stats if s["reward_saved"] > 0 and s["r_corr"] <= 0)
    has_fmt  = sum(1 for s in stats if s["format_bonus"] > 0)
    pr(f"  有 format bonus 的 rollout    : {pct(has_fmt,  len(stats))}")
    pr(f"  reward>0 但 r_corr=0 的      : {pct(fmt_only, len(stats))}")
    pr(f"  → 这些是 '假正确'：reward>0 纯靠格式奖励")

    # ── 4. 搜索次数分布 ──────────────────────────────────────────────────────
    pr()
    pr("── 4. 搜索次数分布")
    search_counts = defaultdict(int)
    for s in stats:
        search_counts[s["n_searches"]] += 1
    for k in sorted(search_counts):
        pr(f"  n_searches={k}: {pct(search_counts[k], len(stats))}")

    # ── 5. Group 分析（GRPO advantage 信号）────────────────────────────────────
    pr()
    pr("── 5. Group 分析（同一问题的 8 条 rollout）")
    groups = defaultdict(list)
    for r, s in zip(records, stats):
        groups[r["question"]].append(s)

    group_sizes = [len(v) for v in groups.values()]
    pr(f"  问题数: {len(groups)}  平均每题 rollout 数: {sum(group_sizes)/len(group_sizes):.1f}")

    # 按 reward 的 group variance
    all_zero_reward = 0   # 组内全为 0
    all_pos_reward  = 0   # 组内全 > 0（都被算'正确'，advantage ≈ 0）
    mixed_reward    = 0   # 混合
    all_zero_f1     = 0   # 组内 F1 全 0
    all_pos_f1      = 0   # 组内 F1 全 > 0
    mixed_f1        = 0   # F1 有高有低

    reward_stds = []
    f1_stds     = []

    for q, grp in groups.items():
        rewards = [s["reward_saved"] for s in grp]
        f1s     = [s["r_corr"] for s in grp]

        # reward variance
        if all(r > 0  for r in rewards): all_pos_reward  += 1
        elif all(r == 0 for r in rewards): all_zero_reward += 1
        else: mixed_reward += 1

        # f1 variance
        if all(f > 0  for f in f1s): all_pos_f1  += 1
        elif all(f == 0 for f in f1s): all_zero_f1 += 1
        else: mixed_f1 += 1

        n = len(rewards)
        if n > 1:
            mu_r = sum(rewards)/n
            std_r = (sum((r-mu_r)**2 for r in rewards)/n)**0.5
            reward_stds.append(std_r)
            mu_f = sum(f1s)/n
            std_f = (sum((f-mu_f)**2 for f in f1s)/n)**0.5
            f1_stds.append(std_f)

    n_groups = len(groups)
    pr()
    pr(f"  [按 reward 分组]")
    pr(f"  全部 reward>0（advantage≈0） : {pct(all_pos_reward,  n_groups)}")
    pr(f"  全部 reward=0                 : {pct(all_zero_reward, n_groups)}")
    pr(f"  混合（有训练信号）             : {pct(mixed_reward,   n_groups)}")
    if reward_stds:
        pr(f"  reward 组内标准差均值         : {sum(reward_stds)/len(reward_stds):.4f}")

    pr()
    pr(f"  [按真实 F1 分组]")
    pr(f"  全部 F1>0（advantage≈0）      : {pct(all_pos_f1,  n_groups)}")
    pr(f"  全部 F1=0                      : {pct(all_zero_f1, n_groups)}")
    pr(f"  混合（有正确性信号）            : {pct(mixed_f1,   n_groups)}")
    if f1_stds:
        pr(f"  F1 组内标准差均值             : {sum(f1_stds)/len(f1_stds):.4f}")

    # ── 6. 重建 reward vs saved reward 一致性检查 ────────────────────────────
    pr()
    pr("── 6. 重建 reward 与 saved reward 一致性")
    diffs = [abs(s["reward_recon"] - s["reward_saved"]) for s in stats]
    big_diff = sum(1 for d in diffs if d > 0.05)
    pr(f"  重建与原始差异 > 0.05 的条数  : {pct(big_diff, len(stats))}")
    pr(f"  平均绝对误差                  : {sum(diffs)/len(diffs):.5f}")

    # ── 7. 几条典型样本 ──────────────────────────────────────────────────────
    pr()
    pr("── 7. 典型样本（reward>0 但 F1=0 的'假正确'）")
    shown = 0
    for r, s in zip(records, stats):
        if s["reward_saved"] > 0 and s["r_corr"] <= 0 and shown < 5:
            pr(f"\n  Q: {r['question'][:80]}")
            pr(f"  A(gold, first sent): {_first_sentence(r['answer'])[:80]}")
            pr(f"  Pred: {str(r.get('prediction',''))[:80]}")
            pr(f"  reward={s['reward_saved']}  F1={s['r_corr']}  format={s['format_bonus']}")
            shown += 1

    pr()
    pr("── 8. 典型样本（高 F1 的真正确）")
    shown = 0
    for r, s in zip(records, stats):
        if s["r_corr"] > 0.8 and shown < 3:
            pr(f"\n  Q: {r['question'][:80]}")
            pr(f"  A(gold, first sent): {_first_sentence(r['answer'])[:80]}")
            pr(f"  Pred: {str(r.get('prediction',''))[:80]}")
            pr(f"  reward={s['reward_saved']}  F1={s['r_corr']}")
            shown += 1

    pr()
    pr("=" * 70)

    # 保存结果
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\n结果已保存: {OUT_FILE}")

if __name__ == "__main__":
    main()
