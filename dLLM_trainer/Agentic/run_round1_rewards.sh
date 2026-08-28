#!/bin/bash
# =============================================================================
# Phase 3: Round 1 Reward 计算
#   输入：round1_merged.jsonl（π₁ rollout 2048 条）
#   输出：train_reward_a.jsonl / train_reward_b.jsonl
#   然后与 Round 0 做对比分析：mean reward、zero-var groups、effective groups
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH

INPUT=$AGENTIC/output/round1_rollouts/round1_merged.jsonl
OUTPUT=$AGENTIC/output/round1_rollouts

echo "================================================================"
echo "Phase 3: Round 1 Reward 计算  $(date)"
echo "Input : $INPUT"
echo "Output: $OUTPUT"
echo "================================================================"

$PYTHON $AGENTIC/compute_rewards.py \
    --input  "$INPUT" \
    --output "$OUTPUT"

echo ""
echo "Reward 计算完成，开始对比分析..."

$PYTHON - << 'EOF'
import json, statistics
from collections import defaultdict
from pathlib import Path

def load_reward_file(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def group_stats(records, label):
    """按 question_id 分组，统计 reward 分布和有效组数。"""
    groups = defaultdict(list)
    for r in records:
        qid = r.get("question_id", r.get("group_id", "?"))
        groups[qid].append(r["reward"])

    rewards_all   = [r["reward"] for r in records]
    mean_r        = statistics.mean(rewards_all)
    pos_rate      = sum(1 for r in rewards_all if r > 0) / len(rewards_all)

    n_groups      = len(groups)
    zero_var      = sum(1 for g in groups.values()
                        if len(set(round(x, 6) for x in g)) == 1)
    effective_g   = n_groups - zero_var

    print(f"\n=== {label} ===")
    print(f"  总 records      : {len(records)}")
    print(f"  总 groups       : {n_groups}")
    print(f"  mean reward     : {mean_r:.4f}")
    print(f"  positive rate   : {pos_rate*100:.1f}%")
    print(f"  zero-var groups : {zero_var}/{n_groups} ({zero_var/n_groups*100:.1f}%)")
    print(f"  effective groups: {effective_g}/{n_groups} ({effective_g/n_groups*100:.1f}%)")

    # reward 分布
    reward_hist = {f"≥{t:.1f}": sum(1 for r in rewards_all if r >= t)
                   for t in [0.0, 0.2, 0.5, 0.8, 1.0]}
    print(f"  reward hist     : {reward_hist}")

    return mean_r, pos_rate, zero_var, effective_g, n_groups

BASE = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output")

# Round 0 Reward A (π₀ baseline)
r0_a = load_reward_file(BASE / "round0_rollouts/train_reward_a.jsonl")
mr0, pr0, zv0, eg0, ng0 = group_stats(r0_a, "Round 0 Reward A (π₀ SFT ckpt_6)")

# Round 1 Reward A (π₁)
r1_path = BASE / "round1_rollouts/train_reward_a.jsonl"
if r1_path.exists():
    r1_a = load_reward_file(r1_path)
    mr1, pr1, zv1, eg1, ng1 = group_stats(r1_a, "Round 1 Reward A (π₁ ELBO-PG ckpt-25)")

    print(f"\n=== π₀ → π₁ Reward 变化 ===")
    print(f"  mean reward     : {mr0:.4f} → {mr1:.4f}  ({mr1-mr0:+.4f})")
    print(f"  positive rate   : {pr0*100:.1f}% → {pr1*100:.1f}%  ({(pr1-pr0)*100:+.1f}pp)")
    print(f"  zero-var groups : {zv0/ng0*100:.1f}% → {zv1/ng1*100:.1f}%  ({(zv1/ng1-zv0/ng0)*100:+.1f}pp)")
    print(f"  effective groups: {eg0} → {eg1}")

    if mr1 > mr0:
        print("\n  ✓ mean reward 提升 —— π₁ 在新问题上的搜索质量更好")
    elif mr1 < mr0 - 0.05:
        print("\n  [WARN] mean reward 明显下降，检查是否 search/behavior collapse")
    else:
        print("\n  → mean reward 基本持平（正常，25步更新幅度有限）")

    if zv1/ng1 < zv0/ng0:
        print("  ✓ zero-var groups 下降 —— reward 分布更多样")
    if eg1 >= eg0 * 0.8:
        print(f"  ✓ 有效训练组充足 ({eg1} ≥ {int(eg0*0.8)})")
    else:
        print(f"  [WARN] 有效训练组减少 ({eg1} < {int(eg0*0.8)})")
else:
    print(f"\n[INFO] Round 1 reward 文件尚不存在: {r1_path}")
    print("请先等待 compute_rewards.py 完成。")

EOF

echo ""
echo "================================================================"
echo "下一步："
echo "  如果 effective_groups ≥ 100，运行: bash run_phase4_round2.sh"
echo "  （π₁ + D₁ → ELBO-PG → π₂）"
echo "================================================================"
echo "Phase 3 Round 1 完成: $(date)"
