#!/bin/bash
# =============================================================================
# Phase 2: Round 1 Rollout (正式)
#   π₁ (ELBO-PG checkpoint-25) × 256题(新) × 8 rollouts = 2048 trajectories
#   新数据：rl_pool_round1.jsonl（pool 的第 256-511 题，Round 0 未用过）
#   8 GPU 并行，每卡 32题
#
#   前提：run_round1_sanity.sh 已通过（无 search/format collapse）
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

# π₁ = ELBO-PG Round 1 checkpoint
MODEL=$AGENTIC/output/phase4_elbopg/reward_a/model

INPUT=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_round1.jsonl  # 第 256-511 题（新数据）
OUTPUT=$AGENTIC/output/round1_rollouts
LOG_DIR=$OUTPUT/logs
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Phase 2: Round 1 Rollout (正式)  $(date)"
echo "Policy  : π₁ (ELBO-PG checkpoint-25)"
echo "Model   : $MODEL"
echo "Input   : $INPUT ($(wc -l < $INPUT)题，新数据)"
echo "Output  : $OUTPUT"
echo "================================================================"

# 交错启动 8 个 rank，每隔 90s 一批（避免 NFS 模型加载并发）
for RANK in $(seq 0 7); do
    SESSION="round1_rank${RANK}"
    LOG="$LOG_DIR/rank${RANK}.log"

    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$RANK
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_${USER}

echo "=== rank $RANK 开始 \$(date) ==="
$PYTHON -u $AGENTIC/collect_round0_rollouts.py \
    --rank     $RANK \
    --world    8 \
    --model    "$MODEL" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --n_rolls  8 \
    --round_id 1 \
    2>&1 | tee "$LOG"
echo "=== rank $RANK 完成 exit=\$? \$(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh

    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  rank $RANK launched (GPU=$RANK)  log=$LOG"

    if [ $RANK -eq 1 ] || [ $RANK -eq 3 ] || [ $RANK -eq 5 ]; then
        echo "  等待 90s..."
        sleep 90
    fi
done

echo ""
echo "================================================================"
echo "全部 8 个 rank 已启动（π₁ Round 1 正式 rollout）"
echo "监控: screen -list | grep round1"
echo "进度: tail -f $LOG_DIR/rank0.log"
echo "================================================================"

echo ""
echo "等待所有 rank 完成..."
while true; do
    DONE=0
    for RANK in $(seq 0 7); do
        screen -list | grep -q "round1_rank${RANK}" || DONE=$((DONE+1))
    done
    echo "  $(date)  完成 rank 数: $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 300
done

echo ""
echo "所有 rank 完成！合并并验证..."
$PYTHON - << 'EOF'
import json, statistics
from collections import Counter
from pathlib import Path

OUT = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round1_rollouts/round1")
all_records = []
for f in sorted(OUT.glob("rollouts_rank*.jsonl")):
    with open(f) as fp:
        records = [json.loads(l) for l in fp if l.strip()]
    print(f"  {f.name}: {len(records)} records")
    all_records.extend(records)

merged_file = OUT.parent / "round1_merged.jsonl"
with open(merged_file, "w") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

qid_counts = Counter(r["question_id"] for r in all_records)
complete   = sum(1 for c in qid_counts.values() if c == 8)
has_answer = sum(1 for r in all_records if r.get("valid_format"))
mean_srch  = sum(r.get("num_searches", 0) for r in all_records) / len(all_records) if all_records else 0
search_rate = sum(1 for r in all_records if r.get("num_searches", 0) > 0) / len(all_records)
term_dist  = Counter(r.get("terminated_by", "?") for r in all_records)

print(f"\n=== Round 1 Rollout (π₁) 汇总 ===")
print(f"总记录数      : {len(all_records)}")
print(f"独立问题数    : {len(qid_counts)}")
print(f"严格8条问题   : {complete}/{len(qid_counts)}")
print(f"search rate   : {search_rate*100:.1f}%")
print(f"avg search    : {mean_srch:.2f}")
print(f"有答案比例    : {has_answer}/{len(all_records)} = {has_answer/len(all_records)*100:.1f}%")
print(f"terminated_by : {dict(term_dist)}")
print(f"输出文件      : {merged_file}")

# 对比 π₀ Round 0 baseline
print(f"\n=== 对比 π₀ (Round 0 Baseline) ===")
r0_records = []
with open("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round0_rollouts/round0_merged.jsonl") as f:
    r0_records = [json.loads(l) for l in f if l.strip()]
r0_sr = sum(1 for r in r0_records if r.get("num_searches", 0) > 0) / len(r0_records)
r0_as = sum(r.get("num_searches", 0) for r in r0_records) / len(r0_records)
r0_ar = sum(1 for r in r0_records if r.get("valid_format")) / len(r0_records)
print(f"  π₀ search rate  : {r0_sr*100:.1f}%   π₁: {search_rate*100:.1f}%  ({(search_rate-r0_sr)*100:+.1f}pp)")
print(f"  π₀ avg search   : {r0_as:.2f}   π₁: {mean_srch:.2f}  ({mean_srch-r0_as:+.2f})")
print(f"  π₀ answer rate  : {r0_ar*100:.1f}%   π₁: {has_answer/len(all_records)*100:.1f}%  ({(has_answer/len(all_records)-r0_ar)*100:+.1f}pp)")

incomplete = [qid for qid, cnt in qid_counts.items() if cnt != 8]
if incomplete:
    print(f"\n[WARN] {len(incomplete)} 个问题不足8条: {incomplete[:5]}")
else:
    print(f"\n✓ 所有 {len(qid_counts)} 问题严格8条")
EOF

echo ""
echo "下一步：bash run_round1_rewards.sh 计算 Reward A/B 和 advantage"
echo "Phase 2 Round 1 完成: $(date)"
