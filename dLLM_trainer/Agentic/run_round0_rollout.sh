#!/bin/bash
# =============================================================================
# Phase 2: Round 0 Rollout
#   SFT ckpt_6 × 256题 × 8 rollouts = 2048 trajectories
#   8 GPU 并行，每卡 32题
#   用 screen 保证 SSH 断线后继续运行
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
INPUT=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_256.jsonl
OUTPUT=$AGENTIC/output/round0_rollouts
LOG_DIR=$OUTPUT/logs
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Phase 2: Round 0 Rollout  $(date)"
echo "Model : $MODEL"
echo "Input : $INPUT ($(wc -l < $INPUT)题)"
echo "Output: $OUTPUT"
echo "================================================================"

# 交错启动 8 个 rank，每隔 90s 一个（避免 NFS 模型加载并发）
for RANK in $(seq 0 7); do
    SESSION="round0_rank${RANK}"
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
    --rank  $RANK \
    --world 8 \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --n_rolls 8 \
    --round_id 0 \
    2>&1 | tee "$LOG"
echo "=== rank $RANK 完成 exit=\$? \$(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh

    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  rank $RANK launched (GPU=$RANK)  log=$LOG"

    # 每 2 个 rank 后等 90s（让模型加载完成再开下一批）
    if [ $RANK -eq 1 ] || [ $RANK -eq 3 ] || [ $RANK -eq 5 ]; then
        echo "  等待 90s..."
        sleep 90
    fi
done

echo ""
echo "================================================================"
echo "全部 8 个 rank 已启动"
echo "监控: screen -list | grep round0"
echo "查看某个: screen -r round0_rank0"
echo "进度: tail -f $LOG_DIR/rank0.log"
echo "================================================================"

# 等待所有 rank 完成
echo ""
echo "等待所有 rank 完成..."
while true; do
    DONE=0
    for RANK in $(seq 0 7); do
        screen -list | grep -q "round0_rank${RANK}" || DONE=$((DONE+1))
    done
    echo "  $(date)  完成 rank 数: $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 300
done

echo ""
echo "所有 rank 完成！合并并验证..."
$PYTHON - << 'EOF'
import json
from collections import Counter
from pathlib import Path

OUT = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round0_rollouts/round0")
all_records = []
for f in sorted(OUT.glob("rollouts_rank*.jsonl")):
    with open(f) as fp:
        records = [json.loads(l) for l in fp if l.strip()]
    print(f"  {f.name}: {len(records)} records")
    all_records.extend(records)

merged_file = OUT.parent / "round0_merged.jsonl"
with open(merged_file, "w") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

qid_counts = Counter(r["question_id"] for r in all_records)
complete   = sum(1 for c in qid_counts.values() if c == 8)
has_answer = sum(1 for r in all_records if r.get("valid_format"))
mean_srch  = sum(r.get("num_searches",0) for r in all_records) / len(all_records) if all_records else 0

print(f"\n=== Round 0 Rollout 汇总 ===")
print(f"总记录数      : {len(all_records)}")
print(f"独立问题数    : {len(qid_counts)}")
print(f"严格8条问题   : {complete}/{len(qid_counts)}")
print(f"有答案比例    : {has_answer}/{len(all_records)} = {has_answer/len(all_records)*100:.1f}%")
print(f"平均搜索次数  : {mean_srch:.2f}")
print(f"输出文件      : {merged_file}")

# 验证 group 完整性
print(f"\n=== Group 完整性验证 ===")
incomplete = [qid for qid, cnt in qid_counts.items() if cnt != 8]
if incomplete:
    print(f"  [WARNING] {len(incomplete)} 个问题不足8条: {incomplete[:5]}")
else:
    print(f"  ✓ 所有 {len(qid_counts)} 个问题严格8条")
EOF

echo ""
echo "Phase 2 完成: $(date)"
