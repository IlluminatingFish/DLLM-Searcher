#!/bin/bash
# 简化版评估脚本：Dataroller rollout + cal_acc
# 使用前：1) 启动 SGLang/vLLM 推理服务  2) 配置 config.json

set -e
cd "$(dirname "$0")"
MODEL_NAME="${1:-Llama-3.1-8B}"  # 与 SGLang 中模型名一致

echo "=== Step 1: Dataroller Rollout (model=$MODEL_NAME) ==="
cd Dataroller
bash run.sh "$MODEL_NAME" example base
OUTPUT_JSONL="$(pwd)/base/${MODEL_NAME}_sglang/example/iter1.jsonl"
cd ..

if [ ! -f "$OUTPUT_JSONL" ]; then
    echo "Error: Rollout output not found at $OUTPUT_JSONL"
    exit 1
fi

echo ""
echo "=== Step 2: Evaluation (cal_acc) ==="
echo "Output: $OUTPUT_JSONL"
echo ""
echo "运行评估（需在 config.json 配置 LLM Judge 的 openai_api_base、openai_api_key、judge_model）："
echo "  python my_eval/cal_acc.py --data $OUTPUT_JSONL"
