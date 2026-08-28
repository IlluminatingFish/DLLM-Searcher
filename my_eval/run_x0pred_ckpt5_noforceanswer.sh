#!/bin/bash
# ckpt_5 非强制答案版 eval，8 GPU，各 75 题

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_10ep/ckpt_5/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
OUTDIR="$ROOT/dLLM_trainer/VRPO/output/llada_eval/x0pred_10ep_ckpt_5_noforceanswer_parallel"
MERGED="$ROOT/dLLM_trainer/VRPO/output/llada_eval/x0pred_10ep_ckpt_5_noforceanswer.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

mkdir -p "$OUTDIR"
echo "====================================="
echo "  x0pred 10ep ckpt_5 非强制答案版"
echo "  8 GPU，各 75 题"
echo "====================================="

GPUS=(0 1 2 3 4 5 6 7)
OFFSETS=(0 75 150 225 300 375 450 525)

PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    GPU=${GPUS[$i]}
    OFFSET=${OFFSETS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model "$MODEL" \
        --input "$INPUT" \
        --output "$OUTDIR/gpu${GPU}.jsonl" \
        --offset "$OFFSET" \
        --max_samples 75 \
        --force_answer_turn 999 \
        > "$OUTDIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU: 题 $OFFSET ~ $((OFFSET+74))  PID=$!"
done

echo "等待完成..."
FAILED=0
for i in 0 1 2 3 4 5 6 7; do
    wait ${PIDS[$i]}; CODE=$?
    [ $CODE -ne 0 ] && echo "[ERROR] GPU ${GPUS[$i]} 退出码 $CODE" && FAILED=1
done
[ $FAILED -ne 0 ] && echo "有进程失败" && exit 1

cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
echo "完成: $(wc -l < "$MERGED") 题 → $MERGED"
