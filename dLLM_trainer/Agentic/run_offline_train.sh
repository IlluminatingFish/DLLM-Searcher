#!/bin/bash
# =============================================================================
# Offline Baseline Training（一次性 150 步）
#
# 前置：run_offline_rollouts.sh 已完成（6轮 rollout 全部生成）
#
# Pipeline：
#   Step 1: 合并 6 轮 rollout → merged，计算 reward → all_train_reward_a.jsonl
#   Step 2: 一次性训练 150 步，每 25 步存 checkpoint
#   Step 3: 并行 eval 6 个 checkpoint（dionysos 8 GPU，每 checkpoint 100q×6 shard）
#   Step 4: 汇报学习曲线，删除中间 checkpoint 的 model 权重（只保留 eval 结果）
#
# 磁盘策略：每个 checkpoint eval 完后删除 model.safetensors，只留最后一个（step-150）
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
OFFLINE_ROOT=$AGENTIC/output/offline_loop/offline_rollouts
OUTBASE=$AGENTIC/output/offline_loop
ALL_TRAIN=$OUTBASE/all_train_reward_a.jsonl
TRAIN_DIR=$OUTBASE/offline_train
MODEL_DIR=$TRAIN_DIR/model
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
LOG=$OUTBASE/offline_train.log

mkdir -p $TRAIN_DIR

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_train
export TORCH_EXTENSIONS_HOME=/tmp/torch_ext_mz751_offline_train
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log "========================================================"
log "Offline Baseline Training 启动"
log "========================================================"

# ── Step 1: 合并 rollout + 计算 reward ───────────────────────────────────────
# 修改：只合并实际存在的 round（offline 实验只跑 round0）
log "Step 1: 合并 offline rollout 并计算 reward..."

if [ ! -f "$ALL_TRAIN" ] || [ $(wc -l < "$ALL_TRAIN") -lt 10 ]; then
    ALL_MERGED=$OUTBASE/all_merged.jsonl
    > $ALL_MERGED
    FOUND=0
    for N in 0 1 2 3 4 5; do
        MERGED=$OFFLINE_ROOT/round${N}/merged.jsonl
        if [ -f "$MERGED" ] && [ $(wc -l < "$MERGED") -ge 10 ]; then
            cat "$MERGED" >> $ALL_MERGED
            log "  round${N}: $(wc -l < $MERGED) 条"
            FOUND=$((FOUND + 1))
        fi
    done
    if [ $FOUND -eq 0 ]; then
        log "[ERROR] 没有找到任何 merged.jsonl，请先运行 run_offline_rollouts.sh"
        exit 1
    fi
    TOTAL=$(wc -l < $ALL_MERGED)
    log "  合计: $TOTAL 条 rollout（共 $FOUND 轮）"

    # 计算 reward（输出到 OUTBASE，生成 all_train_reward_a.jsonl）
    $PYTHON $AGENTIC/compute_rewards.py \
        --input "$ALL_MERGED" \
        --output "$OUTBASE" \
        2>&1 | tee -a $LOG

    # compute_rewards 输出的是 train_reward_a.jsonl，重命名一下
    mv $OUTBASE/train_reward_a.jsonl $ALL_TRAIN 2>/dev/null || true
    mv $OUTBASE/train_reward_b.jsonl $OUTBASE/all_train_reward_b.jsonl 2>/dev/null || true

    NTRAIN=$(wc -l < "$ALL_TRAIN" 2>/dev/null || echo 0)
    log "  训练数据: $NTRAIN 条 → $ALL_TRAIN"
else
    NTRAIN=$(wc -l < "$ALL_TRAIN")
    log "  all_train_reward_a.jsonl 已存在 ($NTRAIN 条)，跳过"
fi

# ── Step 2: 一次性训练 150 步 ──────────────────────────────────────────────
log ""
log "Step 2: 训练 150 步 (save every 25 steps)..."
TRAIN_LOG=$TRAIN_DIR/train.log
CONFIG=$AGENTIC/configs/grpo_offline_150steps.yaml

if [ ! -d "$MODEL_DIR/checkpoint-125" ]; then
    cd $VRPO
    accelerate launch \
        --config_file $AGENTIC/configs/accel_zero3_8gpu.yaml \
        --num_processes 8 \
        my_train/llada_grpo_train.py \
        --config $CONFIG \
        2>&1 | tee $TRAIN_LOG

    TRAIN_EXIT=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT -ne 0 ]; then
        log "[ERROR] 训练失败 (exit=$TRAIN_EXIT)"
        exit 1
    fi
    log "  训练完成！"
else
    log "  checkpoint-150 已存在，跳过训练"
fi

# ── Step 3: eval 各 checkpoint ─────────────────────────────────────────────
log ""
log "Step 3: eval 6 个 checkpoint..."

CHECKPOINTS="25 50 75 100 125"
WORLD=8

for STEP in $CHECKPOINTS; do
    CKPT_DIR=$MODEL_DIR/checkpoint-${STEP}
    EVAL_DIR=$TRAIN_DIR/eval_step${STEP}
    SCORE_LOG=$EVAL_DIR/score.log
    mkdir -p $EVAL_DIR

    # ZeRO-3 fix: 确保 checkpoint 下有 model.safetensors
    if [ ! -f "$CKPT_DIR/model.safetensors" ] && [ -f "$CKPT_DIR/pytorch_model.bin" ]; then
        log "  [step-${STEP}] ZeRO-3 fix: 转换权重..."
        # 使用 ZeRO-3 consolidate
        $PYTHON -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('$CKPT_DIR', trust_remote_code=True)
m.save_pretrained('$CKPT_DIR')
print('done')
" 2>/dev/null || log "  [WARNING] 转换失败，跳过 step-${STEP}"
    fi

    if [ ! -f "$SCORE_LOG" ] && [ -f "$CKPT_DIR/model.safetensors" ]; then
        log "  eval step-${STEP} 启动..."
        for GPU in $(seq 0 $((WORLD-1))); do
            OFFSET=$((GPU * 75))
            TRITON_CACHE_DIR=/tmp/triton_offline_eval_s${STEP}_g${GPU}
            mkdir -p $TRITON_CACHE_DIR
            nohup bash -c "
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u \
    ${ROOT}/my_eval/run_llada_eval.py \
    --model  '${CKPT_DIR}' \
    --input  '${EVAL_INPUT}' \
    --output '${EVAL_DIR}/shard${GPU}.jsonl' \
    --offset ${OFFSET} \
    --max_samples 75 \
    2>&1 | tee '${EVAL_DIR}/shard${GPU}.log'
" > /dev/null 2>&1 &
            sleep 5
        done

        log "  等待 step-${STEP} eval 完成..."
        until [ $(ls $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l) -ge $WORLD ] && \
              [ $(cat $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l) -ge 590 ]; do
            LINES=$(cat $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l)
            log "    step-${STEP}: ${LINES}/600"
            sleep 120
        done

        # 合并 + 打分
        cat $EVAL_DIR/shard*.jsonl > $EVAL_DIR/all.jsonl
        $PYTHON $ROOT/my_eval/cal_acc.py \
            --data $EVAL_DIR/all.jsonl \
            2>&1 | tee $SCORE_LOG
        log "  step-${STEP} eval 完成"

        # 磁盘策略：eval 完后删除中间 checkpoint 的 model.safetensors（保留 step-150）
        if [ "$STEP" -ne 125 ]; then
            log "  [存储] 删除 step-${STEP} model.safetensors..."
            rm -f $CKPT_DIR/model.safetensors
            log "  [存储] 已清理 checkpoint-${STEP} 权重（保留其余文件供参考）"
        fi
    elif [ -f "$SCORE_LOG" ]; then
        log "  step-${STEP} eval 已存在，跳过"
    else
        log "  [WARNING] step-${STEP}: checkpoint 不存在或无 model.safetensors，跳过"
    fi
done

# ── Step 4: 学习曲线汇总 ───────────────────────────────────────────────────
log ""
log "========================================================"
log "Offline Baseline 学习曲线"
log "========================================================"
printf "%-12s %-10s\n" "Checkpoint" "CEM-1" | tee -a $LOG
printf "%-12s %-10s\n" "π₀ SFT" "43.33%" | tee -a $LOG
for STEP in $CHECKPOINTS; do
    SC=$TRAIN_DIR/eval_step${STEP}/score.log
    if [ -f "$SC" ]; then
        CEM=$(grep -oP '\d+\.\d+(?=%)' "$SC" | head -1)
        printf "%-12s %-10s\n" "step-${STEP}" "${CEM}%" | tee -a $LOG
    fi
done
log "========================================================"
log "全部完成！最终模型: $MODEL_DIR/checkpoint-125"
