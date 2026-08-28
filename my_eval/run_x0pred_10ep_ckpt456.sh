#!/bin/bash
# x0pred 10ep: ckpt_4, ckpt_5, ckpt_6 顺序评测，各 600 题
# 8 GPU 并行（每卡 75 题），三个 ckpt 依次执行

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
BASE_MODEL_DIR="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_10ep"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
OUTROOT="$ROOT/dLLM_trainer/VRPO/output/llada_eval"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

GPUS=(0 1 2 3 4 5 6 7)
OFFSETS=(0 75 150 225 300 375 450 525)
COUNTS=(75 75 75 75 75 75 75 75)

for CKPT in ckpt_4 ckpt_5 ckpt_6; do
    MODEL="$BASE_MODEL_DIR/$CKPT/optimized"
    OUTDIR="$OUTROOT/x0pred_10ep_${CKPT}_600_parallel"
    MERGED="$OUTROOT/x0pred_10ep_${CKPT}_600.jsonl"

    mkdir -p "$OUTDIR"
    echo "=============================="
    echo "  评测 $CKPT"
    echo "  模型: $MODEL"
    echo "=============================="

    PIDS=()
    for i in 0 1 2 3 4 5 6 7; do
        GPU=${GPUS[$i]}
        OFFSET=${OFFSETS[$i]}
        COUNT=${COUNTS[$i]}
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$ROOT/my_eval/run_llada_eval.py" \
            --model "$MODEL" \
            --input "$INPUT" \
            --output "$OUTDIR/gpu${GPU}.jsonl" \
            --offset "$OFFSET" \
            --max_samples "$COUNT" \
            > "$OUTDIR/gpu${GPU}.log" 2>&1 &
        PIDS+=($!)
        echo "  GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  PID=$!"
    done

    echo "等待 $CKPT 完成..."
    FAILED=0
    for i in 0 1 2 3 4 5 6 7; do
        wait ${PIDS[$i]}; CODE=$?
        [ $CODE -ne 0 ] && echo "[ERROR] GPU ${GPUS[$i]} 退出码 $CODE" && FAILED=1
    done
    [ $FAILED -ne 0 ] && echo "$CKPT 有进程失败，退出" && exit 1

    cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
    echo "$CKPT 完成: $(wc -l < "$MERGED") 题 → $MERGED"
    echo ""
done

echo "=============================="
echo "全部评测完成"
echo "=============================="
