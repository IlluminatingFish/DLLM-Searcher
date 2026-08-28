#!/bin/bash
# =============================================================================
# Offline Baseline Loop (hermes 训练 × dionysos eval 流水线)
#
# 对照实验：Frozen-Behavior-Policy / Offline ELBO-PG
# 唯一与 on-policy 的区别：rollout 始终来自固定 SFT π₀，不随训练 policy 更新
#
# Pipeline:
#   hermes  : 顺序训练 π'₁ → π'₂ → ... → π'₆  (8 GPU ZeRO-3)
#   dionysos: 每轮训练完后立即并行 eval (GPU 0-5, 6 卡)
#   存储管理 : 训练完立即删 optimizer states (101GB)
#             eval 完立即删 model weights (除 π'₃ π'₆ 外)
#
# 前置条件：已运行 run_offline_rollouts.sh 生成全部离线数据
#
# 用法：
#   screen -S offline_loop bash run_offline_loop.sh
# =============================================================================
set -uo pipefail

START_ROUND=0   # π₀ 开始，产出 π'₁
MAX_ROUND=5     # 最后训练轮，产出 π'₆
POOL_BATCH=256
WORLD=8         # hermes GPU 数
DIONYSOS_WORLD=6  # dionysos GPU 0-5

# 永久保留的中间模型（不删 model.safetensors）
KEEP_ROUNDS="6"   # 只保留 π'₆ (final)，其余 eval 完立刻删除以节省配额

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
SFT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
OFFLINE_ROOT=$AGENTIC/output/offline_loop/offline_rollouts
OUTBASE=$AGENTIC/output/offline_loop
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
LOOP_LOG=$OUTBASE/loop_offline.log

mkdir -p $OUTBASE

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_train
export TORCH_EXTENSIONS_HOME=/tmp/torch_ext_mz751_offline_train
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOOP_LOG; }

should_keep() {
    local N=$1
    for K in $KEEP_ROUNDS; do [ "$N" -eq "$K" ] && return 0; done
    return 1
}

get_model_path() {
    local N=$1
    [ "$N" -eq 0 ] && echo "$SFT_MODEL" || echo "$OUTBASE/offline_pi${N}/model"
}

# ── dionysos eval 触发函数 ────────────────────────────────────────────────────
# 在 dionysos 上启动 6 GPU eval，返回后台 SSH 进程 PID（写入文件）
launch_dionysos_eval() {
    local ROUND_N=$1            # 被评估的 π'_N (即 NEXT_ROUND)
    local MODEL_PATH=$2
    local EVAL_DIR=$3
    local FLAG_FILE=$4          # 完成后 dionysos 写入此文件

    mkdir -p $EVAL_DIR
    > $FLAG_FILE.running        # 标记 eval 进行中

    ssh dionysos "
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
EVAL_INPUT=${EVAL_INPUT}
MODEL=${MODEL_PATH}
EVAL_DIR=${EVAL_DIR}
FLAG=${FLAG_FILE}
ROOT=${ROOT}

# 启动 6 GPU eval（每卡 100 题）
for GPU in \$(seq 0 5); do
    OFFSET=\$((GPU * 100))
    export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_eval_\${GPU}
    mkdir -p \$TRITON_CACHE_DIR
    nohup bash -c \"
        export CUDA_VISIBLE_DEVICES=\${GPU}
        export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_eval_\${GPU}
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        \${PYTHON} -u \${ROOT}/my_eval/run_llada_eval.py \\\\
            --model  '\${MODEL}' \\\\
            --input  '\${EVAL_INPUT}' \\\\
            --output '\${EVAL_DIR}/shard\${GPU}.jsonl' \\\\
            --offset \${OFFSET} \\\\
            --max_samples 100 \\\\
            2>&1 | tee '\${EVAL_DIR}/shard\${GPU}.log'
    \" > /dev/null 2>&1 &
    sleep 5
done

# 等待所有 shard 完成（每 shard 100 题）
until [ \$(ls \${EVAL_DIR}/shard*.jsonl 2>/dev/null | wc -l) -ge 6 ] && \
      [ \$(cat \${EVAL_DIR}/shard*.jsonl 2>/dev/null | wc -l) -ge 600 ]; do
    LINES=\$(cat \${EVAL_DIR}/shard*.jsonl 2>/dev/null | wc -l)
    echo \"dionysos eval π'_${ROUND_N}: \${LINES}/600\"
    sleep 60
done

# 合并 + 计算分数
cat \${EVAL_DIR}/shard*.jsonl > \${EVAL_DIR}/all.jsonl
\${PYTHON} \${ROOT}/my_eval/cal_acc.py --data \${EVAL_DIR}/all.jsonl 2>/dev/null \
    | tee \${EVAL_DIR}/score.log

# 写完成标志
echo 'DONE' > \${FLAG}
rm -f \${FLAG}.running
" &

    echo $!   # 返回 SSH 进程 PID
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
log "========================================================"
log "Offline Loop 启动：Round ${START_ROUND} → ${MAX_ROUND}  (目标: π'₆)"
log "  offline data: $OFFLINE_ROOT"
log "  永久保留: π'$(echo $KEEP_ROUNDS | tr ' ' '/')"
log "========================================================"

PREV_EVAL_FLAG=""
PREV_EVAL_PID=""
PREV_ROUND_MODEL=""
PREV_ROUND_N=0

for ROUND in $(seq $START_ROUND $MAX_ROUND); do
    NEXT_ROUND=$((ROUND + 1))
    BASE_MODEL=$(get_model_path $ROUND)
    ROUND_DIR=$OUTBASE/offline_pi${NEXT_ROUND}
    OFFLINE_DATA=$OFFLINE_ROOT/round${ROUND}/train_reward_a.jsonl
    OUT_MODEL=$ROUND_DIR/model
    EVAL_DIR=$ROUND_DIR/eval
    EVAL_FLAG=$ROUND_DIR/eval_done.flag
    TRAIN_LOG=$ROUND_DIR/train.log
    mkdir -p $ROUND_DIR

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    log ""
    log "========================================================"
    log "Round $ROUND → π'_${NEXT_ROUND}"
    log "  Base model  : $BASE_MODEL"
    log "  Offline data: $OFFLINE_DATA"
    log "========================================================"

    # ── 检查 offline 数据 ────────────────────────────────────────────────────
    if [ ! -f "$OFFLINE_DATA" ]; then
        log "[ERROR] offline 训练数据不存在: $OFFLINE_DATA"
        log "  请先运行: bash run_offline_rollouts.sh"
        exit 1
    fi
    NTRAIN=$(wc -l < "$OFFLINE_DATA")
    log "  [OK] offline 训练数据 $NTRAIN 条"

    # ── 训练 ─────────────────────────────────────────────────────────────────
    MODEL_READY=false
    MODEL_SIZE=$(stat -c%s "$OUT_MODEL/model.safetensors" 2>/dev/null || echo 0)
    [ -f "$OUT_MODEL/model.safetensors" ] && [ "$MODEL_SIZE" -gt 1000000000 ] && MODEL_READY=true

    if ! $MODEL_READY; then
        ROUND_CONFIG=$AGENTIC/configs/offline_pi${NEXT_ROUND}_${TIMESTAMP}.yaml
        sed "s|model_name_or_path:.*|model_name_or_path: ${BASE_MODEL}|;
             s|output_dir:.*|output_dir: ${OUT_MODEL}|;
             s|run_name:.*|run_name: offline_pi${NEXT_ROUND}_${TIMESTAMP}|;
             s|dataset_path:.*|dataset_path: ${OFFLINE_DATA}|" \
            $AGENTIC/configs/grpo_phase4_reward_a.yaml > $ROUND_CONFIG

        log "  开始训练 π'_${NEXT_ROUND}..."
        cd $VRPO
        accelerate launch \
            --config_file $AGENTIC/configs/accel_zero3_8gpu.yaml \
            --num_processes 8 \
            my_train/llada_grpo_train.py \
            --config $ROUND_CONFIG \
            2>&1 | tee $TRAIN_LOG

        TRAIN_EXIT=${PIPESTATUS[0]}
        if [ $TRAIN_EXIT -ne 0 ]; then
            log "[ERROR] Round $ROUND 训练失败 (exit=$TRAIN_EXIT)"
            exit 1
        fi

        # ZeRO-3 fix：从 checkpoint 复制完整权重到根目录
        LATEST_CKPT=$(ls -dt $OUT_MODEL/checkpoint-*/model.safetensors 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_SIZE=$(stat -c%s "$LATEST_CKPT" 2>/dev/null || echo 0)
            if [ "$CKPT_SIZE" -gt 1000000000 ]; then
                log "  [ZeRO-3 fix] 从 checkpoint 复制权重..."
                cp "$LATEST_CKPT" "$OUT_MODEL/model.safetensors"

                # 同步其他必要文件
                CKPT_DIR=$(dirname "$LATEST_CKPT")
                for f in config.json generation_config.json tokenizer_config.json \
                         tokenizer.json chat_template.jinja \
                         configuration_llada.py modeling_llada.py; do
                    cp "$CKPT_DIR/$f" "$OUT_MODEL/$f" 2>/dev/null || \
                    cp "$(get_model_path $ROUND)/$f" "$OUT_MODEL/$f" 2>/dev/null || true
                done
                log "  [OK] 权重复制完成"
            fi
        fi

        # !! 立即删除 optimizer states (释放 ~101GB) !!
        if [ -n "$(ls -dt $OUT_MODEL/checkpoint-* 2>/dev/null)" ]; then
            log "  [存储] 删除 optimizer states (checkpoint-*)..."
            rm -rf $OUT_MODEL/checkpoint-*
            log "  [存储] 已清理，model.safetensors 保留"
        fi

        log "  训练完成！π'_${NEXT_ROUND} 已保存"
    else
        log "  π'_${NEXT_ROUND} 已存在，跳过训练"
    fi

    # ── 等待上一轮 dionysos eval 完成（如有）────────────────────────────────
    if [ -n "$PREV_EVAL_FLAG" ] && [ ! -f "$PREV_EVAL_FLAG" ]; then
        log "  等待 dionysos eval π'_${PREV_ROUND_N} 完成..."
        while [ ! -f "$PREV_EVAL_FLAG" ]; do
            LINES=$(ssh dionysos "cat ${OUTBASE}/offline_pi${PREV_ROUND_N}/eval/shard*.jsonl 2>/dev/null | wc -l" 2>/dev/null || echo 0)
            log "  dionysos eval π'_${PREV_ROUND_N}: ${LINES}/600 题"
            sleep 60
        done
        log "  dionysos eval π'_${PREV_ROUND_N} 完成！"

        # 上一轮 eval 已完成，打印分数
        log "$(cat $OUTBASE/offline_pi${PREV_ROUND_N}/eval/score.log 2>/dev/null | grep 'CEM-1\|LLM Judge')"

        # 删除上一轮 model（若不是需要永久保留的）
        if ! should_keep $PREV_ROUND_N && [ -f "$PREV_ROUND_MODEL/model.safetensors" ]; then
            log "  [存储] 删除 π'_${PREV_ROUND_N} model.safetensors (不在保留列表)..."
            rm -f $PREV_ROUND_MODEL/model.safetensors
            log "  [存储] 已删除"
        else
            log "  [存储] 保留 π'_${PREV_ROUND_N}（在永久保留列表中或不存在）"
        fi
    fi

    # ── 启动本轮 dionysos eval（后台）───────────────────────────────────────
    if [ ! -f "$EVAL_FLAG" ]; then
        log "  启动 dionysos eval π'_${NEXT_ROUND}（后台）..."
        EVAL_PID=$(launch_dionysos_eval $NEXT_ROUND "$OUT_MODEL" "$EVAL_DIR" "$EVAL_FLAG")
        log "  dionysos eval SSH PID=$EVAL_PID"
    else
        log "  eval 已完成，跳过"
    fi

    # 记录状态供下一轮使用
    PREV_EVAL_FLAG=$EVAL_FLAG
    PREV_EVAL_PID=${EVAL_PID:-""}
    PREV_ROUND_MODEL=$OUT_MODEL
    PREV_ROUND_N=$NEXT_ROUND

done

# ── 等待最后一轮 eval 完成 ────────────────────────────────────────────────────
if [ -n "$PREV_EVAL_FLAG" ] && [ ! -f "$PREV_EVAL_FLAG" ]; then
    log "等待最后一轮 dionysos eval 完成 (π'_${PREV_ROUND_N})..."
    while [ ! -f "$PREV_EVAL_FLAG" ]; do
        sleep 60
    done
    log "最后 eval 完成！"
fi

# ── 最终结果汇总 ──────────────────────────────────────────────────────────────
log ""
log "========================================================"
log "Offline Loop 全部完成！Learning Curve:"
log "========================================================"
printf "%-10s %-10s %-10s\n" "Policy" "CEM-1" "LLM-Judge" | tee -a $LOOP_LOG
printf "%-10s %-10s %-10s\n" "π'₀ SFT" "43.17%" "45.33%"  | tee -a $LOOP_LOG

for N in $(seq 1 $((MAX_ROUND+1))); do
    SC=$OUTBASE/offline_pi${N}/eval/score.log
    if [ -f "$SC" ]; then
        CEM=$(grep -oP '(?<=CEM-1[^:]*:[ ]+)[0-9]+\.[0-9]+' $SC | head -1)
        JDG=$(grep -oP '(?<=LLM Judge[^:]*:[ ]+)[0-9]+\.[0-9]+' $SC | head -1)
        printf "%-10s %-10s %-10s\n" "π'_${N}" "${CEM:-?}%" "${JDG:-?}%" | tee -a $LOOP_LOG
    fi
done
log "========================================================"
