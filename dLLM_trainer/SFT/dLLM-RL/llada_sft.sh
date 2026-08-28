#!/bin/bash
# LLaDA-1.5 Agentic SFT training
# Run from: /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/
#
# Usage:
#   bash llada_sft.sh                        # use all 8 GPUs on hermes
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash llada_sft.sh   # use 4 GPUs

export WANDB_DISABLED=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default: 8 GPUs on hermes
NGPUS=${NUM_GPUS:-8}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$GPUS

# Pick accelerate config based on GPU count
if [ "$NGPUS" -eq 8 ]; then
    ACCEL_CFG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"
elif [ "$NGPUS" -eq 6 ]; then
    ACCEL_CFG="accelerate_configs/1_node_6_gpus_deepspeed_zero3.yaml"
else
    ACCEL_CFG="accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml"
fi

echo "=============================="
echo "LLaDA Agentic SFT"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS GPUs)"
echo "Accel config: $ACCEL_CFG"
echo "=============================="

# Copy SFT data to expected location
SFT_DATA="/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/sft_output/sft_train_final.json"
if [ ! -f "../data/sft_train_final.json" ]; then
    echo "Linking SFT data..."
    ln -sf "$SFT_DATA" "../data/sft_train_final.json"
fi

/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8899 \
  --config_file "$ACCEL_CFG" \
  train/sft_llada.py \
  config=configs/sft_llada.yaml
