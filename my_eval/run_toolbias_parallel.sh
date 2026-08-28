#!/bin/bash
# SFT baseline + tool-call logit bias +0.5，100 题，8GPU 并行
# 对比组：run_sft_baseline_parallel.sh（无 bias）

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
OUTDIR="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_toolbias_parallel"
MERGED="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_toolbias_ckpt6_100.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
LOGIT_BIAS=${1:-0.5}

mkdir -p "$OUTDIR"

OFFSETS=(0  13 26 39 52 65 78 91)
COUNTS=(13 13 13 13 13 13 13  9)

echo "=========================================="
echo "SFT baseline + tool-call logit bias=${LOGIT_BIAS}"
echo "模型: $MODEL"
echo "=========================================="

PIDS=()
for i in $(seq 0 7); do
    GPU=$i
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    OUT="$OUTDIR/gpu${GPU}.jsonl"

    echo "GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  → $OUT"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_toolbias_eval.py" \
        --model       "$MODEL" \
        --input       "$INPUT" \
        --output      "$OUT" \
        --offset      "$OFFSET" \
        --max_samples "$COUNT" \
        --logit_bias  "$LOGIT_BIAS" \
        > "$OUTDIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "8 个进程已启动: ${PIDS[*]}"
echo "等待全部完成..."

FAILED=0
for i in $(seq 0 7); do
    wait ${PIDS[$i]}
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[ERROR] GPU $i 退出码 $CODE，查看: $OUTDIR/gpu${i}.log"
        FAILED=1
    else
        COUNT=$(wc -l < "$OUTDIR/gpu${i}.jsonl" 2>/dev/null || echo 0)
        echo "GPU $i 完成: $COUNT 题"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "有进程失败，跳过合并"
    exit 1
fi

echo ""
echo "合并结果到 $MERGED ..."
cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
TOTAL=$(wc -l < "$MERGED")
echo "合并完成: $TOTAL 题"

echo ""
echo "打分（需先跑 origGT 后处理）:"
echo "  python my_eval/cal_acc.py --data $MERGED"
