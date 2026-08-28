#!/bin/bash
# 对照实验: x0-prediction SFT with block_size=32 (LLaDA2.0 style)
# 对比: bs32 vs bs64(π₃) vs bs128
# Run from: /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/

export WANDB_DISABLED=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_sft_bs32  # 本地缓存避免 NFS 崩溃

NGPUS=${NUM_GPUS:-7}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
export CUDA_VISIBLE_DEVICES=$GPUS

if [ "$NGPUS" -eq 8 ]; then
    ACCEL_CFG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"
elif [ "$NGPUS" -eq 7 ]; then
    ACCEL_CFG="accelerate_configs/1_node_7_gpus_deepspeed_zero3.yaml"
elif [ "$NGPUS" -eq 6 ]; then
    ACCEL_CFG="accelerate_configs/1_node_6_gpus_deepspeed_zero3.yaml"
else
    ACCEL_CFG="accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml"
fi

echo "=============================="
echo "LLaDA x0-prediction SFT  [block_size=32, LLaDA2.0 style]"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS GPUs)"
echo "Accel config: $ACCEL_CFG"
echo "=============================="

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="llada_sft_x0pred_bs32_$(hostname)_${TIMESTAMP}.log"

/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8901 \
  --config_file "$ACCEL_CFG" \
  train/sft_llada_x0pred.py \
  config=configs/sft_llada_x0pred_bs32.yaml 2>&1 | tee "$LOG_FILE"
