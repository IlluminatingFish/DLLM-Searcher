#!/bin/bash
# Plan A: DPO on dionysos (8 GPU)

set -e
cd /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
DATA_PATH=data/dpo_pairs_plan_ab_ref.jsonl
OUTPUT_DIR=/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_a/model
LOG_FILE=output/rl/plan_a/train_$(date +%Y%m%d_%H%M%S).log

mkdir -p output/rl/plan_a
mkdir -p "$OUTPUT_DIR"

echo "[plan_a] 启动训练: $(date)" | tee "$LOG_FILE"

accelerate launch \
    --config_file recipes/accelerate_configs/zero3_cpu_offload.yaml \
    --num_processes 8 \
    my_train/llada_dpo_train_a.py \
    --model_path "$MODEL_PATH" \
    --data_path  "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 512 \
    --batch_size 1 \
    --grad_accum 8 \
    --block_length 64 \
    --num_mc 1 \
    --lr 5e-7 \
    --beta 0.1 \
    2>&1 | tee -a "$LOG_FILE"

echo "[plan_a] 训练完成: $(date)" | tee -a "$LOG_FILE"
