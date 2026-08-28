#!/bin/bash
# Auto-restart supervisor for the latent-only (SIM-CoT faithful) trainer.
set -u

OUTPUT_DIR=/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_latent_only
BASE_MODEL=/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/llada_model
WORKDIR=/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO
TARGET_STEPS=1000
LOG=/tmp/coconut_latent_only_supervisor.log

mkdir -p "$OUTPUT_DIR"
cd "$WORKDIR" || exit 1
source /research/cbim/vast/mz751/miniforge3/etc/profile.d/conda.sh
conda activate llada

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

while true; do
    for d in "$OUTPUT_DIR"/checkpoint-*/; do
        [ -d "$d" ] || continue
        if [ ! -f "${d}model.safetensors.index.json" ]; then
            log "Removing incomplete checkpoint: $d"
            rm -rf "$d"
        fi
    done

    LATEST=""
    LATEST_STEP=0
    for d in $(ls -d "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | sed 's#/$##' | sort -t- -k2 -n -r); do
        if [ -f "$d/model.safetensors.index.json" ] && [ -f "$d/trainer_state.json" ]; then
            LATEST="$d"
            LATEST_STEP=$(python3 -c "import json; print(json.load(open('$d/trainer_state.json'))['global_step'])" 2>/dev/null || echo 0)
            break
        fi
    done

    if [ -n "$LATEST" ]; then
        MODEL_PATH="$LATEST"
        log "Resuming from $LATEST (global_step=$LATEST_STEP)"
    else
        MODEL_PATH="$BASE_MODEL"
        LATEST_STEP=0
        log "No valid checkpoint found, starting fresh from base model"
    fi

    if [ "$LATEST_STEP" -ge "$TARGET_STEPS" ]; then
        log "Reached target of $TARGET_STEPS steps. Done."
        break
    fi

    log "Launching accelerate (target=$TARGET_STEPS, resume_step=$LATEST_STEP)..."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    unset ACCELERATE_DEEPSPEED_CONFIG_FILE
    unset ACCELERATE_CONFIG_DS_FIELDS
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml my_train/coconut_latent_only_trainer.py \
        --model_name_or_path "$MODEL_PATH" \
        --dtype bfloat16 \
        --dataset_path /research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data/sft_with_plans.jsonl \
        --output_dir "$OUTPUT_DIR" \
        --max_ctx_len 4096 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --gradient_checkpointing True \
        --learning_rate 5e-05 --lr_scheduler_type cosine --warmup_ratio 0.05 --bf16 true \
        --logging_steps 10 --save_steps 90 --save_total_limit 1 --save_only_model True \
        --max_steps "$TARGET_STEPS" \
        --dataloader_num_workers 0 --remove_unused_columns false --overwrite_output_dir true --report_to wandb \
        >> "$LOG" 2>&1
    EXIT_CODE=$?
    log "accelerate launch exited with code $EXIT_CODE. Checking whether to retry..."
    sleep 15
done

log "Supervisor loop finished."
