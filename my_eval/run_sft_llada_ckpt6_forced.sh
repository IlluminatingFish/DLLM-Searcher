#!/bin/bash
# sft_llada ckpt_6，600题，4 GPU，强制答案版推理

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL="/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
OUTDIR="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_llada_ckpt6_forced_600_parallel"
MERGED="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_llada_ckpt6_forced_600.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

mkdir -p "$OUTDIR"

echo "=========================================="
echo "sft_llada ckpt_6 [强制答案版] - 600题 - 4 GPU"
echo "模型: $MODEL"
echo "=========================================="

OFFSETS=(0   150  300  450)
COUNTS=(150  150  150  150)

PIDS=()
for i in 0 1 2 3; do
    GPU=$i
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    OUT="$OUTDIR/gpu${GPU}.jsonl"
    echo "GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model  "$MODEL" \
        --input  "$INPUT" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
        > "$OUTDIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

echo "4 个进程已启动: ${PIDS[*]}"
echo "等待完成..."

FAILED=0
for i in 0 1 2 3; do
    wait ${PIDS[$i]}; CODE=$?
    [ $CODE -ne 0 ] && echo "[ERROR] GPU $i 退出码 $CODE" && FAILED=1 \
        || echo "GPU $i 完成: $(wc -l < $OUTDIR/gpu${i}.jsonl) 题"
done

[ $FAILED -ne 0 ] && echo "有进程失败" && exit 1

echo "合并 → $MERGED"
cat "$OUTDIR"/gpu{0,1,2,3}.jsonl > "$MERGED"
echo "完成: $(wc -l < $MERGED) 题"
