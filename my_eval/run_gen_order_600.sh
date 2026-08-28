#!/usr/bin/env bash
# Run token reveal-order tracking for all 600 eval questions.
# Uses 8 GPUs on hermes: GPU 0-3 for sft_llada, GPU 4-7 for x0pred.
# Each GPU handles 150 questions. All 8 jobs run concurrently.
# Estimated wall time: ~2-4 hours.

set -e

SCRIPT="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/track_gen_order.py"
OUT_BASE="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order"
LOG_DIR="${OUT_BASE}/logs_600"
SHARD_DIR="${OUT_BASE}/shards_600"

mkdir -p "$LOG_DIR" "$SHARD_DIR"

PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllmsearch/bin/python

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
mkdir -p "$TRITON_CACHE_DIR"

echo "========================================"
echo "  gen_order 600 launch  $(date)"
echo "  shards  → $SHARD_DIR"
echo "  logs    → $LOG_DIR"
echo "========================================"

# ── sft_llada: GPU 0-3, offsets 0/150/300/450 ──
for i in 0 1 2 3; do
    OFFSET=$(( i * 150 ))
    OUT="${SHARD_DIR}/sft_llada_${OFFSET}.json"
    LOG="${LOG_DIR}/sft_llada_${OFFSET}.log"
    echo "[launch] sft_llada  offset=$OFFSET  gpu=$i  → $OUT"
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR}_sft${i}" \
    nohup "$PYTHON" -u "$SCRIPT" \
        --model_name sft_llada \
        --gpu $i \
        --n 150 \
        --offset $OFFSET \
        --out_path "$OUT" \
        > "$LOG" 2>&1 &
    echo $! > "${LOG_DIR}/sft_llada_${OFFSET}.pid"
done

# ── x0pred: GPU 4-7, offsets 0/150/300/450 ──
for i in 0 1 2 3; do
    GPU=$(( i + 4 ))
    OFFSET=$(( i * 150 ))
    OUT="${SHARD_DIR}/x0pred_${OFFSET}.json"
    LOG="${LOG_DIR}/x0pred_${OFFSET}.log"
    echo "[launch] x0pred     offset=$OFFSET  gpu=$GPU  → $OUT"
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR}_x0p${i}" \
    nohup "$PYTHON" -u "$SCRIPT" \
        --model_name x0pred \
        --gpu $GPU \
        --n 150 \
        --offset $OFFSET \
        --out_path "$OUT" \
        > "$LOG" 2>&1 &
    echo $! > "${LOG_DIR}/x0pred_${OFFSET}.pid"
done

echo ""
echo "8 jobs launched. Monitor with:"
echo "  tail -f ${LOG_DIR}/sft_llada_0.log"
echo "  watch -n30 'wc -l ${SHARD_DIR}/*.json 2>/dev/null || ls -lh ${SHARD_DIR}/'"
echo ""
echo "When all done, merge with:"
echo "  python ${OUT_BASE}/merge_shards_600.py"
