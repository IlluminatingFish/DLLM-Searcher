#!/bin/bash
# 8-GPU 并行评测 baseline 模型（sft_llada/ckpt_6）
# 使用 toolcall_pre_rl 推理策略，block_size=128
# 用法: cd /research/cbim/vast/mz751/Projects/DLLM-Searcher && bash my_eval/run_baseline_eval_parallel.sh

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
OUTDIR="$ROOT/dLLM_trainer/VRPO/output/llada_eval/baseline_parallel"
MERGED="$ROOT/dLLM_trainer/VRPO/output/llada_eval/baseline_ckpt6_100.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

mkdir -p "$OUTDIR"

# 100 题分 8 份：每份 13 题（最后一份 9 题）
OFFSETS=(0  13 26 39 52 65 78 91)
COUNTS=(13 13 13 13 13 13 13  9)

echo "=========================================="
echo "LLaDA baseline eval (toolcall_pre_rl) - 8 GPU 并行"
echo "模型: $MODEL"
echo "策略: toolcall_pre_rl  block_size=128"
echo "=========================================="

PIDS=()
for i in $(seq 0 7); do
    GPU=$i
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    OUT="$OUTDIR/gpu${GPU}.jsonl"

    echo "GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  → $OUT"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_baseline_llada_eval.py" \
        --model  "$MODEL" \
        --input  "$INPUT" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
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
echo "打分:"
$PYTHON "$ROOT/my_eval/cal_acc.py" --data "$MERGED"
