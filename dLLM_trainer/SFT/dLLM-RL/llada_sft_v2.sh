#!/bin/bash
# LLaDA SFT v2 — 修复短答案数据后重训
# Run from: dLLM_trainer/SFT/dLLM-RL/

export WANDB_DISABLED=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

NGPUS=${NUM_GPUS:-8}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$GPUS

ACCEL_CFG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="llada_sft_v2_$(hostname)_${TIMESTAMP}.log"

echo "=============================="
echo "LLaDA SFT v2 (fixed short answers)"
echo "GPUs: $CUDA_VISIBLE_DEVICES  Config: $ACCEL_CFG"
echo "Log: $LOG_FILE"
echo "=============================="

/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8899 \
  --config_file "$ACCEL_CFG" \
  train/sft_llada.py \
  config=configs/sft_llada_v2.yaml 2>&1 | tee "$LOG_FILE"
