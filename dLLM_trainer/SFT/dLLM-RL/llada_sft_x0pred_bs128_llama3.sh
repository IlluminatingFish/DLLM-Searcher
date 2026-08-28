#!/bin/bash
# Llama3-format SFT 重训练
# Protocol: end-to-end Llama3 canonical format
#   - prompt: apply_chat_template → <|start_header_id|>system...
#   - tool_response: <|eot_id|><|start_header_id|>user...<tool_response>...</tool_response><|eot_id|><|start_header_id|>assistant...
# Data: data/sft_train_final_llama3.json (converted from sft_train_final.json)
# Run from: /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/

export WANDB_DISABLED=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_sft_llama3   # 本地缓存避免 NFS 崩溃

# hermes 只用 GPU 0-3（4张），避免跨 NUMA 崩溃
NGPUS=${NUM_GPUS:-4}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=$GPUS

ACCEL_CFG="accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml"

echo "=============================="
echo "LLaDA SFT  [Llama3 canonical protocol]"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS GPUs)"
echo "Data: sft_train_final_llama3.json"
echo "Config: configs/sft_llada_x0pred_bs128_llama3.yaml"
echo "=============================="

# 确认数据文件存在
DATA_FILE="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/../data/sft_train_final_llama3.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: 数据文件不存在: $DATA_FILE"
    exit 1
fi
echo "Data file: $DATA_FILE  ($(wc -c < "$DATA_FILE") bytes)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="llada_sft_llama3_$(hostname)_${TIMESTAMP}.log"

/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8900 \
  --config_file "$ACCEL_CFG" \
  train/sft_llada_x0pred.py \
  config=configs/sft_llada_x0pred_bs128_llama3.yaml 2>&1 | tee "$LOG_FILE"
