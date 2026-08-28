#!/bin/bash
# sft_llada_x0pred_bs128_llama3  ckpt_4 / ckpt_3 / ckpt_5
# hermes 8-GPU 并行，block_size=128, num_steps=128
# 用法: bash run_llama3_sft_eval.sh

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL_BASE="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_llama3"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
RESULTS="$ROOT/my_eval/results/bs128_llama3"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
EVAL_SCRIPT="$ROOT/my_eval/run_llada_eval.py"
CAL_ACC="$ROOT/my_eval/cal_acc.py"

OFFSETS=(0 75 150 225 300 375 450 525)
COUNTS=(75 75 75 75 75 75 75 75)

run_eval_ckpt() {
    local CKPT=$1
    local MODEL="$MODEL_BASE/ckpt_${CKPT}/optimized"
    local OUTDIR="$RESULTS/ckpt${CKPT}_parallel"
    local MERGED="$RESULTS/ckpt${CKPT}_merged.jsonl"

    echo "======================================================"
    echo "评估: sft_llada_x0pred_bs128_llama3/ckpt_${CKPT}"
    echo "模型: $MODEL"
    echo "block_size=128  num_steps=128  600题 × 8GPU"
    echo "======================================================"

    mkdir -p "$OUTDIR"
    PIDS=()
    for i in $(seq 0 7); do
        GPU=$i
        OUT="$OUTDIR/gpu${GPU}.jsonl"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$EVAL_SCRIPT" \
            --model  "$MODEL" \
            --input  "$INPUT" \
            --output "$OUT" \
            --offset "${OFFSETS[$i]}" \
            --max_samples "${COUNTS[$i]}" \
            --block_size 128 \
            --num_steps 128 \
            > "$OUTDIR/gpu${GPU}.log" 2>&1 &
        echo "  GPU $GPU → offset=${OFFSETS[$i]} PID=$!"
        PIDS+=($!)
    done

    echo "等待 8 个进程..."
    FAILED=0
    for i in $(seq 0 7); do
        wait ${PIDS[$i]} || { echo "[ERROR] GPU $i 失败"; FAILED=1; }
        CNT=$(wc -l < "$OUTDIR/gpu${i}.jsonl" 2>/dev/null || echo 0)
        echo "  GPU $i 完成: $CNT 题"
    done

    [ $FAILED -ne 0 ] && { echo "有进程失败，跳过合并"; return 1; }

    cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
    TOTAL=$(wc -l < "$MERGED")
    echo "合并: $TOTAL 题 → $MERGED"

    echo "--- 计算正确率 ---"
    $PYTHON "$CAL_ACC" --data "$MERGED" || true
    echo "======================================================"
}

mkdir -p "$RESULTS"

# 按优先级顺序：ckpt_4 → ckpt_3 → ckpt_5
run_eval_ckpt 4
run_eval_ckpt 3
run_eval_ckpt 5

echo "全部完成！"
