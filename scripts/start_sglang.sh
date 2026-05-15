#!/bin/bash
# SGLang 推理服务启动脚本（支持 ReAct 长对话）
# 用法: bash scripts/start_sglang.sh [模型路径] [端口] [context_length]
#
# 方式 1：--allow-auto-truncate（超长自动截断，省显存）
# 方式 2：--context-length 32768（显式支持 32k，需更多显存）

MODEL_PATH="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-30000}"
CONTEXT_LEN="${3:-32768}"

echo "Starting SGLang: $MODEL_PATH port=$PORT context_length=$CONTEXT_LEN"
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --context-length "$CONTEXT_LEN" \
  --allow-auto-truncate
