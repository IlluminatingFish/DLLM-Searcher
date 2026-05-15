#!/bin/bash
# DLLM-Searcher 环境安装脚本
# 环境名: dllmsearch
# 用途: 数据采集 (Dataroller) + 基础依赖

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="dllmsearch"

echo "========================================"
echo "Creating conda environment: $ENV_NAME"
echo "Project root: $PROJECT_ROOT"
echo "========================================"

# 1. 创建 conda 环境
conda create -n "$ENV_NAME" python=3.10 -y

# 2. 激活并安装
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. 安装 PyTorch (CUDA 12.1，无 GPU 则装 CPU 版)
if command -v nvidia-smi &> /dev/null; then
    echo "Detected GPU, installing PyTorch with CUDA 12.1..."
    pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected, installing PyTorch CPU version..."
    pip install torch torchvision
fi

# 4. Qwen-Agent + Dataroller 依赖 (数据采集核心)
pip install "qwen-agent[gui,rag,code_interpreter,mcp]"
pip install requests tqdm rich openai tiktoken transformers
pip install "huggingface-hub>=0.34.0,<1.0"  # transformers 兼容性

echo ""
echo "========================================"
echo "Environment '$ENV_NAME' ready!"
echo "========================================"
echo "Activate: conda activate $ENV_NAME"
echo ""
echo "Note: SGLang 需单独安装/运行，用于模型推理服务。"
echo "      Dataroller 客户端通过 OpenAI API 调用 SGLang/vLLM 服务。"
echo ""
echo "Before running Dataroller, set:"
echo "  export GOOGLE_SEARCH_KEY=\"your_serper_api_key\""
echo "  # 在 react_agent.py 中配置 OPENAI_API_BASE 和 OPENAI_API_KEY"
echo ""
echo "Then: cd $PROJECT_ROOT/Dataroller && bash scripts/test.sh"
echo "========================================"
