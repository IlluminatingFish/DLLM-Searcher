#!/bin/bash
# 8-GPU 并行 eval，600 题，每 GPU 75 题
# 用法: bash run_eval_parallel.sh <model_dir> <ckpt_idx> <block_size> <out_dir>
# 例:   bash run_eval_parallel.sh sft_llada_x0pred_bs128 6 128 results/bs128

set -e

MODEL_NAME=${1:-"sft_llada_x0pred_bs128"}
CKPT_IDX=${2:-6}
BLOCK_SIZE=${3:-64}
OUT_SUBDIR=${4:-"results/bs${BLOCK_SIZE}"}

SFT_BASE="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL"
MODEL="${SFT_BASE}/${MODEL_NAME}/ckpt_${CKPT_IDX}/optimized"
INPUT="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/eval/merged.jsonl"
OUT_DIR="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/${OUT_SUBDIR}"
PYTHON="/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python"
EVAL_SCRIPT="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/run_llada_eval.py"
NUM_STEPS=${BLOCK_SIZE}   # num_steps = block_size

mkdir -p "$OUT_DIR"

NGPU=8
TOTAL=600
PER_GPU=$(( TOTAL / NGPU ))  # 75

echo "====================================================="
echo "模型:       ${MODEL_NAME}/ckpt_${CKPT_IDX}"
echo "block_size: ${BLOCK_SIZE}  num_steps: ${NUM_STEPS}"
echo "每 GPU:     ${PER_GPU} 题  × ${NGPU} GPU = ${TOTAL} 题"
echo "输出目录:   ${OUT_DIR}"
echo "====================================================="

pids=()
for i in $(seq 0 $(( NGPU - 1 ))); do
    OFFSET=$(( i * PER_GPU ))
    OUT="${OUT_DIR}/ckpt${CKPT_IDX}_gpu${i}.jsonl"
    LOG="${OUT_DIR}/ckpt${CKPT_IDX}_gpu${i}.log"

    CUDA_VISIBLE_DEVICES=$i nohup $PYTHON $EVAL_SCRIPT \
        --model "$MODEL" \
        --input "$INPUT" \
        --output "$OUT" \
        --offset $OFFSET \
        --max_samples $PER_GPU \
        --block_size $BLOCK_SIZE \
        --num_steps $NUM_STEPS \
        > "$LOG" 2>&1 &
    pids+=($!)
    echo "GPU $i 启动: offset=$OFFSET  PID=$!"
done

echo "等待 ${#pids[@]} 个进程完成..."
for pid in "${pids[@]}"; do
    wait "$pid" || echo "PID $pid 异常退出"
done

# 合并结果
MERGED="${OUT_DIR}/ckpt${CKPT_IDX}_merged.jsonl"
cat "${OUT_DIR}/ckpt${CKPT_IDX}_gpu"*.jsonl > "$MERGED"
TOTAL_LINES=$(wc -l < "$MERGED")
CORRECT=$( python3 -c "
import json, sys
lines = [json.loads(l) for l in open('$MERGED') if l.strip()]
from pathlib import Path; sys.path.insert(0, '/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval')
try:
    from cal_acc import normalize_answer, exact_match
    ok = sum(1 for l in lines if l.get('prediction') and normalize_answer(str(l.get('prediction',''))) in normalize_answer(str(l.get('short_answer') or l.get('answer',''))))
except:
    ok = sum(1 for l in lines if str(l.get('short_answer') or l.get('answer','')).lower().strip() in str(l.get('prediction','')).lower())
print(ok)
" 2>/dev/null || echo "?")

echo ""
echo "===== ckpt_${CKPT_IDX} 完成 ======"
echo "合并行数: $TOTAL_LINES"
echo "正确数:   $CORRECT / $TOTAL_LINES"
