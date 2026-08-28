#!/bin/bash
# Variant: x0-prediction SFT with block_size=128 (vs default 64)
# 对比实验: 观察 block_size 对搜索推理能力的影响
# Run from: /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/

export WANDB_DISABLED=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_sft_bs128  # 本地缓存避免 NFS 崩溃

NGPUS=${NUM_GPUS:-8}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$GPUS

if [ "$NGPUS" -eq 8 ]; then
    ACCEL_CFG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"
elif [ "$NGPUS" -eq 6 ]; then
    ACCEL_CFG="accelerate_configs/1_node_6_gpus_deepspeed_zero3.yaml"
else
    ACCEL_CFG="accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml"
fi

echo "=============================="
echo "LLaDA x0-prediction SFT  [block_size=128]"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS GPUs)"
echo "Accel config: $ACCEL_CFG"
echo "=============================="

SFT_DATA="/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/sft_output/sft_train_final.json"
if [ ! -f "../data/sft_train_final.json" ]; then
    echo "Linking SFT data..."
    ln -sf "$SFT_DATA" "../data/sft_train_final.json"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="llada_sft_x0pred_bs128_$(hostname)_${TIMESTAMP}.log"

/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8899 \
  --config_file "$ACCEL_CFG" \
  train/sft_llada_x0pred.py \
  config=configs/sft_llada_x0pred_bs128.yaml 2>&1 | tee "$LOG_FILE"
