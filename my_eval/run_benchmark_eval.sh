#!/bin/bash
# 三模型 × 三数据集 benchmark 评测
# 用法: CUDA_VISIBLE_DEVICES=X bash my_eval/run_benchmark_eval.sh <model_tag> <model_path> <eval_script>
# model_tag: sft_baseline / x0pred_standard / x0pred_latent
# eval_script: run_llada_eval.py / run_llada_latent_eval.py

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
cd /research/cbim/vast/mz751/Projects/DLLM-Searcher

MODEL_TAG=$1
MODEL_PATH=$2
EVAL_SCRIPT=$3

OUTBASE=dLLM_trainer/VRPO/output/rl/eval/benchmark
DATA_DIR=dLLM_trainer/VRPO/data

for DATASET in hotpot 2wiki musique; do
    INPUT="$DATA_DIR/${DATASET}_test_200.jsonl"
    OUTPUT="$OUTBASE/$MODEL_TAG/$DATASET/pred.jsonl"
    LOG="$OUTBASE/$MODEL_TAG/$DATASET/eval.log"

    # 如果已有完整结果则跳过
    if [ -f "$OUTPUT" ] && [ $(wc -l < "$OUTPUT") -ge 180 ]; then
        echo "[$MODEL_TAG/$DATASET] 已有结果 $(wc -l < $OUTPUT) 题，跳过"
        continue
    fi

    mkdir -p "$OUTBASE/$MODEL_TAG/$DATASET"
    echo "[$MODEL_TAG/$DATASET] 开始评测..."
    python "$EVAL_SCRIPT" \
        --model "$MODEL_PATH" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        2>&1 | tee "$LOG"

    COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    echo "[$MODEL_TAG/$DATASET] 完成，共 $COUNT 题"
done
echo "=== $MODEL_TAG 全部完成 ==="
