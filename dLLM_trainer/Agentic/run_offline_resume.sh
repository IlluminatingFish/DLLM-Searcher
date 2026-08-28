#!/bin/bash
# =============================================================================
# Offline Resume Training（从 step-50 继续到 step-125）
#
# 前置：run_offline_train.sh 在 step-50 后崩溃（磁盘超配额）
# checkpoint-25 和 checkpoint-50 的 model.safetensors 已保留
# optimizer states 已删除（释放 148G 磁盘）
#
# 本脚本：
#   Step 1: 从 checkpoint-50 继续训练 75 步 → model_cont/checkpoint-{25,50,75}
#           同时后台 cleaner 清理 optimizer states 防止磁盘满
#   Step 2: 重命名 model_cont → model/checkpoint-{75,100,125}
#   Step 3: eval 所有 5 个 checkpoint（step-25,50,75,100,125）
#   Step 4: 汇报学习曲线
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
OUTBASE=$AGENTIC/output/offline_loop
TRAIN_DIR=$OUTBASE/offline_train
MODEL_DIR=$TRAIN_DIR/model
MODEL_CONT=$TRAIN_DIR/model_cont
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
LOG=$OUTBASE/offline_resume.log
TRAIN_LOG=$TRAIN_DIR/train_resume.log

mkdir -p $TRAIN_DIR $MODEL_CONT

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_resume
export TORCH_EXTENSIONS_HOME=/tmp/torch_ext_mz751_offline_resume
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log "========================================================"
log "Offline Resume Training 启动（step-51 → step-125）"
log "  Base model: checkpoint-50"
log "  续训: 75 步（save every 25 steps）"
log "========================================================"

# ── Step 1: 后台 cleaner（每 5 分钟删除 optimizer states，防止磁盘满）────────
log "启动 cleaner（防磁盘满）..."
(while true; do
    sleep 300
    for CONT_CKPT_DIR in $MODEL_CONT/checkpoint-*/; do
        if [ -d "$CONT_CKPT_DIR" ]; then
            for GSTEP in $CONT_CKPT_DIR/global_step*/; do
                if [ -d "$GSTEP" ] && [ -f "${CONT_CKPT_DIR}model.safetensors" ]; then
                    echo "[CLEANER $(date '+%H:%M:%S')] 删除 $GSTEP" >> $LOG
                    rm -rf "$GSTEP"
                fi
            done
            # 清理其他非必要文件
            rm -f $CONT_CKPT_DIR/rng_state_*.pth 2>/dev/null
            rm -f $CONT_CKPT_DIR/scheduler.pt 2>/dev/null
            rm -f $CONT_CKPT_DIR/zero_to_fp32.py 2>/dev/null
            rm -f $CONT_CKPT_DIR/latest 2>/dev/null
        fi
    done
done) &
CLEANER_PID=$!
log "  cleaner PID=$CLEANER_PID"

# ── Step 2: 训练 75 步 ─────────────────────────────────────────────────────
log ""
log "Step 1: 训练 75 步 (base=checkpoint-50, save every 25 steps)..."
CONFIG=$AGENTIC/configs/grpo_offline_resume.yaml

if [ ! -d "$MODEL_CONT/checkpoint-75" ]; then
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
        kill $CLEANER_PID 2>/dev/null
        exit 1
    fi
    log "  训练完成！"
else
    log "  checkpoint-75 已存在，跳过训练"
fi

# 停止 cleaner
kill $CLEANER_PID 2>/dev/null || true

# 最终清理 model_cont 中的 optimizer states
log "最终清理 model_cont 中的 optimizer states..."
rm -rf $MODEL_CONT/checkpoint-*/global_step* 2>/dev/null || true
rm -f $MODEL_CONT/checkpoint-*/rng_state_*.pth 2>/dev/null || true
rm -f $MODEL_CONT/checkpoint-*/scheduler.pt 2>/dev/null || true
rm -f $MODEL_CONT/checkpoint-*/zero_to_fp32.py 2>/dev/null || true
rm -f $MODEL_CONT/checkpoint-*/latest 2>/dev/null || true
log "  完成，当前磁盘："
quota -s 2>/dev/null | grep -v Filesystem | head -2 || true

# ── Step 3: 整合 checkpoint 到 model 目录（重命名）────────────────────────
log ""
log "Step 2: 整合 model_cont → model（重命名 checkpoint 编号）..."
for PAIR in "25:75" "50:100" "75:125"; do
    SRC=$(echo $PAIR | cut -d: -f1)
    DST=$(echo $PAIR | cut -d: -f2)
    SRCDIR=$MODEL_CONT/checkpoint-${SRC}
    DSTDIR=$MODEL_DIR/checkpoint-${DST}
    if [ -d "$SRCDIR" ] && [ ! -d "$DSTDIR" ]; then
        mv "$SRCDIR" "$DSTDIR"
        log "  checkpoint-${SRC} → checkpoint-${DST} ✓"
    elif [ -d "$DSTDIR" ]; then
        log "  checkpoint-${DST} 已存在，跳过"
    else
        log "  [WARNING] checkpoint-${SRC} 不存在"
    fi
done

# ── Step 4: eval 所有 5 个 checkpoint ─────────────────────────────────────
log ""
log "Step 3: eval 5 个 checkpoint（25, 50, 75, 100, 125）..."
CHECKPOINTS="25 50 75 100 125"
WORLD=8

for STEP in $CHECKPOINTS; do
    CKPT_DIR=$MODEL_DIR/checkpoint-${STEP}
    EVAL_DIR=$TRAIN_DIR/eval_step${STEP}
    SCORE_LOG=$EVAL_DIR/score.log
    mkdir -p $EVAL_DIR

    if [ -f "$SCORE_LOG" ]; then
        log "  step-${STEP} eval 已存在，跳过"
        continue
    fi

    if [ ! -f "$CKPT_DIR/model.safetensors" ]; then
        log "  [WARNING] step-${STEP}: model.safetensors 不存在，跳过"
        continue
    fi

    log "  eval step-${STEP} 启动（${WORLD} GPU × 75 题）..."
    for GPU in $(seq 0 $((WORLD-1))); do
        OFFSET=$((GPU * 75))
        TRITON_TMP=/tmp/triton_offline_eval_s${STEP}_g${GPU}
        mkdir -p $TRITON_TMP
        nohup bash -c "
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=${TRITON_TMP}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PYTHON} -u ${ROOT}/my_eval/run_llada_eval.py \
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

    # 合并 + 计算分数
    cat $EVAL_DIR/shard*.jsonl > $EVAL_DIR/all.jsonl
    $PYTHON $ROOT/my_eval/cal_acc.py \
        --data $EVAL_DIR/all.jsonl \
        2>&1 | tee $SCORE_LOG
    log "  step-${STEP} eval 完成"

    # 磁盘策略：eval 完后删除 model.safetensors（保留 step-125）
    if [ "$STEP" -ne 125 ]; then
        log "  [存储] 删除 step-${STEP} model.safetensors..."
        rm -f $CKPT_DIR/model.safetensors
    fi
done

# ── Step 5: 学习曲线汇总 ───────────────────────────────────────────────────
log ""
log "========================================================"
log "Offline Baseline 学习曲线（5 checkpoints）"
log "========================================================"
printf "%-14s %-10s\n" "Checkpoint" "CEM-1" | tee -a $LOG
printf "%-14s %-10s\n" "π₀ SFT" "43.33%" | tee -a $LOG
for STEP in 25 50 75 100 125; do
    SC=$TRAIN_DIR/eval_step${STEP}/score.log
    if [ -f "$SC" ]; then
        CEM=$(grep -oP '\d+\.\d+(?=%)' "$SC" | head -1)
        printf "%-14s %-10s\n" "step-${STEP}" "${CEM}%" | tee -a $LOG
    else
        printf "%-14s %-10s\n" "step-${STEP}" "N/A" | tee -a $LOG
    fi
done
log "========================================================"
log "全部完成！最终模型: $MODEL_DIR/checkpoint-125"
