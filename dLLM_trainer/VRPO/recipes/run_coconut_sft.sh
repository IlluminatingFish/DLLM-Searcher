#!/bin/bash
# Run COCONUT+SIM-CoT+LLaDA SFT training
# Config: recipes/coconut_sft_stage1.yaml
# Usage: bash recipes/run_coconut_sft.sh

set -e
cd "$(dirname "$0")/.."

CONFIG="recipes/coconut_sft_stage1.yaml"
PYTHON="/research/cbim/vast/mz751/miniforge3/envs/llada/bin/python"

# Parse YAML fields with Python
read_yaml() {
    $PYTHON -c "import yaml,sys; d=yaml.safe_load(open('$CONFIG')); print($1)"
}

MODEL_PATH=$(read_yaml "d['model']['pretrained_model']")
DATA_PATH=$(read_yaml "d['dataset']['path']")
OUTPUT_DIR=$(read_yaml "d['output']['output_dir']")
ACCEL_CFG=$(read_yaml "d['accelerate']['config_file']")
WANDB_PROJECT=$(read_yaml "d['wandb']['project']")
WANDB_RUN=$(read_yaml "d['wandb']['run_name']")
STAGE=$(read_yaml "d['coconut']['stage']")
N_LATENT=$(read_yaml "d['coconut']['n_latent']")
BLOCK_SIZE=$(read_yaml "d['coconut']['block_size']")
AUX_WEIGHT=$(read_yaml "d['coconut']['aux_loss_weight']")
MASK_ID=$(read_yaml "d['model']['mask_token_id']")
MAX_CTX=$(read_yaml "d['sequence']['max_ctx_len']")
MAX_ACT=$(read_yaml "d['sequence']['max_act_len']")
EPOCHS=$(read_yaml "d['training']['num_train_epochs']")
BSZ=$(read_yaml "d['training']['per_device_train_batch_size']")
GRAD_ACC=$(read_yaml "d['training']['gradient_accumulation_steps']")
GC=$(read_yaml "d['training']['gradient_checkpointing']")
LR=$(read_yaml "d['optimizer']['learning_rate']")
SCHEDULER=$(read_yaml "d['optimizer']['lr_scheduler_type']")
WARMUP=$(read_yaml "d['optimizer']['warmup_ratio']")
LOG_STEPS=$(read_yaml "d['output']['logging_steps']")
SAVE_STEPS=$(read_yaml "d['output']['save_steps']")
SAVE_LIMIT=$(read_yaml "d['output']['save_total_limit']")
SAVE_ONLY_MODEL=$(read_yaml "d['output'].get('save_only_model', False)")
MAX_STEPS=$(read_yaml "d['training'].get('max_steps', -1)")
RESUME_CKPT=$(read_yaml "d['training'].get('resume_from_checkpoint', '')")

echo "=== COCONUT SFT Stage ${STAGE} ==="
echo "Config:  ${CONFIG}"
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "Seq len: ctx=${MAX_CTX} act=${MAX_ACT}"
echo "=============================="

TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton_cache \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_RUN_NAME="${WANDB_RUN}" \
PATH="/research/cbim/vast/mz751/miniforge3/envs/llada/bin:$PATH" \
accelerate launch \
  --config_file "${ACCEL_CFG}" \
  my_train/coconut_sft_trainer.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dtype bfloat16 \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --stage ${STAGE} \
  --n_latent ${N_LATENT} \
  --block_size ${BLOCK_SIZE} \
  --aux_loss_weight ${AUX_WEIGHT} \
  --mask_token_id ${MASK_ID} \
  --max_ctx_len ${MAX_CTX} \
  --max_act_len ${MAX_ACT} \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BSZ} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --gradient_checkpointing ${GC} \
  --learning_rate ${LR} \
  --lr_scheduler_type ${SCHEDULER} \
  --warmup_ratio ${WARMUP} \
  --bf16 true \
  --logging_steps ${LOG_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_LIMIT} \
  --save_only_model ${SAVE_ONLY_MODEL} \
  --max_steps ${MAX_STEPS} \
  ${RESUME_CKPT:+--resume_from_checkpoint "${RESUME_CKPT}"} \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  --overwrite_output_dir true \
  --report_to wandb
