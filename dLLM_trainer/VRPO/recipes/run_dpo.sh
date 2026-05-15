#!/bin/bash
# =============================================================================
# SDAR DPO Training Launch Script
# =============================================================================

# Configuration
DATASET="dpo"
export LOGDIR=output
RUN_NAME=${DATASET}_dpo_$(date +%Y%m%d_%H%M%S)

# Model path
model_name_or_path="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_2/optimized"

# Dataset path
dataset_path="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/train.jsonl"

# Wandb configuration
export WANDB_DIR="$LOGDIR/$RUN_NAME"
export WANDB_PROJECT="sdar_dpo"

# Create output directory
mkdir -p "$LOGDIR/$RUN_NAME"

# DEBUG_DPO_IPDB=1: 单卡 + ipdb 断点调试
# NUM_GPUS=1: 单卡运行，无 ipdb（用于验证多卡 NaN 是否与单卡不同）
# 默认 4 卡，可通过 NUM_GPUS 覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [ "${DEBUG_DPO_IPDB}" = "1" ]; then
    NUM_PROCESSES=1
elif [ "${NUM_GPUS}" = "1" ]; then
    NUM_PROCESSES=1
else
    NUM_PROCESSES=${NUM_GPUS:-4}
fi

# DEBUG_DPO_FP32=1: FP32 训练，模型用 FP32 加载 + 前向，避免 bf16 下 SDAR block diffusion 导致 logits NaN
# 默认启用 FP32 以避免 logits NaN。若需 bf16 省显存可设 DEBUG_DPO_FP32=0
# FP32 显存大，默认用 ZeRO-3（参数分片）省显存。若仍 OOM 可设 DEBUG_DPO_ZERO3_OFFLOAD=1 启用参数 CPU offload
if [ "${DEBUG_DPO_FP32}" != "0" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [ "${DEBUG_DPO_ZERO3_OFFLOAD}" = "1" ]; then
        ACCEL_CONFIG="recipes/accelerate_configs/zero3_offload_fp32.yaml"
    else
        ACCEL_CONFIG="recipes/accelerate_configs/zero3_fp32.yaml"
    fi
    BF16_ARG="--bf16 false"
    DTYPE_ARG="--dtype float32"
else
    ACCEL_CONFIG="recipes/accelerate_configs/zero2.yaml"
    BF16_ARG=""
    DTYPE_ARG=""
fi

# Launch training
accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    my_train/my_dpo_train.py \
    $BF16_ARG \
    $DTYPE_ARG \
    --config recipes/dpo.yaml \
    --dataset_name "$DATASET" \
    --dataset_path "$dataset_path" \
    --model_name_or_path "$model_name_or_path" \
    --run_name "$RUN_NAME" \
    --output_dir "$LOGDIR/$RUN_NAME/checkpoints" \
    --logging_steps 1 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --beta 0.1 \
    --block_length 128 \
    --num_mc 1 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.5 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --save_steps 50 \
    --save_total_limit 5 \
    --gradient_checkpointing true \
    --use_peft false \
    --wandb_project "sdar_dpo"