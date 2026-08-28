#!/bin/bash
# =============================================================================
# Agentic RL 主循环（Online GRPO for dLLM）
#
# 用法:
#   PLAN=A SERVER=hermes  bash run_agentic_loop.sh   # Plan A on hermes
#   PLAN=B SERVER=achilleas bash run_agentic_loop.sh  # Plan B on achilleas
#   PLAN=C SERVER=homer   bash run_agentic_loop.sh    # Plan C on homer
#
# 每 round 分三步：
#   Step 1: 8 GPU 并行 on-policy rollout（当前策略 + 真实搜索）
#   Step 2: 合并 rollout，计算 advantages（SALT-lite for Plan C）
#   Step 3: 8 GPU DeepSpeed ZeRO-2 GRPO 训练
#
# 持久性：脚本应在 tmux/screen/nohup 下运行，不会因 IDE 关闭而终止。
# =============================================================================

set -eo pipefail

PLAN=${PLAN:-A}
SERVER=${SERVER:-hermes}
START_ROUND=${START_ROUND:-1}
MAX_ROUNDS=${MAX_ROUNDS:-12}

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC_DIR=$ROOT/dLLM_trainer/Agentic
VRPO_DIR=$ROOT/dLLM_trainer/VRPO

# ── Python/PATH ──────────────────────────────────────────────────────────────
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
# Triton 编译缓存放本地 /tmp（避免 NFS 多进程文件锁竞争导致 CUDA 崩溃）
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

# ── Plan-specific params ─────────────────────────────────────────────────────
case $PLAN in
  A)
    N_QUESTIONS=480
    N_ROLLS=8
    LR=3e-7
    PLAN_LOWER=a
    ;;
  B)
    N_QUESTIONS=400
    N_ROLLS=6
    LR=3e-7
    PLAN_LOWER=b
    ;;
  C)
    N_QUESTIONS=320
    N_ROLLS=4
    LR=2e-7
    PLAN_LOWER=c
    ;;
  *)
    echo "[ERROR] PLAN must be A, B, or C"; exit 1
    ;;
esac

# hermes 和 homer 均有双 NUMA 拓扑（GPU 0-3 = NUMA0，GPU 4-7 = NUMA1）。
# 8-GPU 跨 NUMA 的 NCCL 在 DeepSpeed initialize_optimizer_states() 中无声崩溃
# （NCCL_P2P_DISABLE/IB_DISABLE/SHM_DISABLE/ALGO=Ring 均无效），故固定用 GPU 0-3
# （4-GPU，NUMA0 单节点）。
if [[ "$SERVER" =~ ^(homer|hermes)$ ]] && [ -z "$CUDA_GPUS" ]; then
    CUDA_GPUS="0 1 2 3"
fi

# 支持 CUDA_GPUS="3 4 5 6 7" 指定物理 GPU 列表（空格分隔）
if [ -n "$CUDA_GPUS" ]; then
    GPU_LIST=($CUDA_GPUS)
    N_GPUS=${#GPU_LIST[@]}
    CUDA_VISIBLE_DEVICES_TRAIN=$(IFS=,; echo "${GPU_LIST[*]}")
else
    GPU_LIST=($(seq 0 $((${N_GPUS:-8} - 1))))
    N_GPUS=${#GPU_LIST[@]}
    CUDA_VISIBLE_DEVICES_TRAIN=$(IFS=,; echo "${GPU_LIST[*]}")
fi
TRAIN_DATA_PATH=$AGENTIC_DIR/output/plan_${PLAN_LOWER}/grpo_train.jsonl
ROLLOUT_BASE=$AGENTIC_DIR/output/plan_${PLAN_LOWER}/rollouts
MODELS_BASE=$AGENTIC_DIR/output/plan_${PLAN_LOWER}/models
LOG_BASE=$AGENTIC_DIR/output/plan_${PLAN_LOWER}/logs
TRAIN_INPUT=$VRPO_DIR/data/sft_train_as_eval.jsonl
# 支持通过 N_GPUS 自动选择 accelerate 配置
if [ -n "$ACCEL_CONFIG_OVERRIDE" ]; then
    ACCEL_CONFIG=$ACCEL_CONFIG_OVERRIDE
elif [ "$N_GPUS" -eq 8 ]; then
    ACCEL_CONFIG=$AGENTIC_DIR/configs/accel_zero2_8gpu.yaml
else
    ACCEL_CONFIG=$AGENTIC_DIR/configs/accel_zero2_${N_GPUS}gpu.yaml
fi
TRAIN_SCRIPT=$VRPO_DIR/my_train/llada_grpo_train.py
ROLLOUT_SCRIPT=$AGENTIC_DIR/collect_agentic_rollouts.py
DATA_SCRIPT=$AGENTIC_DIR/make_agentic_data.py

INIT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized

mkdir -p "$ROLLOUT_BASE" "$MODELS_BASE" "$LOG_BASE"
MAIN_LOG=$LOG_BASE/agentic_rl_${SERVER}_$(date +%Y%m%d_%H%M%S).log

echo "================================================================" | tee "$MAIN_LOG"
echo "Agentic RL Plan $PLAN on $SERVER  $(date)"                        | tee -a "$MAIN_LOG"
echo "N_QUESTIONS=$N_QUESTIONS  N_ROLLS=$N_ROLLS  LR=$LR"             | tee -a "$MAIN_LOG"
echo "Rounds: $START_ROUND → $MAX_ROUNDS"                               | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"

# =============================================================================
# Main loop
# =============================================================================
for ROUND in $(seq $START_ROUND $MAX_ROUNDS); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "" | tee -a "$MAIN_LOG"
    echo "################################################################" | tee -a "$MAIN_LOG"
    echo "# Round $ROUND / $MAX_ROUNDS  $(date)"                           | tee -a "$MAIN_LOG"
    echo "################################################################" | tee -a "$MAIN_LOG"

    # ── 确定本 round 使用的模型 ──────────────────────────────────────────────
    if [ "$ROUND" -eq 1 ]; then
        MODEL=$INIT_MODEL
    else
        PREV=$((ROUND - 1))
        MODEL=$MODELS_BASE/round${PREV}/model
        if [ ! -d "$MODEL" ]; then
            echo "[WARN] round$PREV 模型不存在，尝试找最近的 checkpoint..."
            # 找最近可用的 round
            for R in $(seq $((ROUND-1)) -1 1); do
                CAND=$MODELS_BASE/round${R}/model
                if [ -d "$CAND" ]; then MODEL=$CAND; break; fi
            done
            if [ ! -d "$MODEL" ]; then MODEL=$INIT_MODEL; fi
            echo "[INFO] 使用模型: $MODEL"
        fi
    fi
    echo "模型: $MODEL" | tee -a "$MAIN_LOG"

    # collect_agentic_rollouts.py appends plan_x/ subdir automatically
    ROLLOUT_PARENT=$ROLLOUT_BASE/round${ROUND}
    ROLLOUT_DIR=$ROLLOUT_PARENT/plan_${PLAN_LOWER##plan_}
    OUT_MODEL=$MODELS_BASE/round${ROUND}/model
    mkdir -p "$ROLLOUT_PARENT" "$OUT_MODEL"

    # ==========================================================================
    # Step 1: On-policy rollout generation（8 GPU 并行）
    # ==========================================================================
    echo "" | tee -a "$MAIN_LOG"
    echo "[Step 1] Rollout 生成: $(date)" | tee -a "$MAIN_LOG"

    ROLLOUT_PIDS=()
    for I in "${!GPU_LIST[@]}"; do
        PHYS_GPU=${GPU_LIST[$I]}
        RANK_LOG=$LOG_BASE/rollout_r${ROUND}_gpu${PHYS_GPU}_${TIMESTAMP}.log
        CUDA_VISIBLE_DEVICES=$PHYS_GPU $PYTHON -u $ROLLOUT_SCRIPT \
            --model       "$MODEL" \
            --input       "$TRAIN_INPUT" \
            --output      "$ROLLOUT_PARENT" \
            --n_questions "$N_QUESTIONS" \
            --n_rolls     "$N_ROLLS" \
            --rank        "$I" \
            --world       "$N_GPUS" \
            --plan        "$PLAN" \
            > "$RANK_LOG" 2>&1 &
        ROLLOUT_PIDS+=($!)
        echo "  GPU$PHYS_GPU (rank $I) PID=${ROLLOUT_PIDS[-1]}" | tee -a "$MAIN_LOG"
    done

    # 等待所有 rollout 进程
    ROLLOUT_OK=true
    for I in "${!ROLLOUT_PIDS[@]}"; do
        PID=${ROLLOUT_PIDS[$I]}
        if ! wait "$PID"; then
            echo "[WARN] rollout GPU$I (PID=$PID) 失败，继续（容错）" | tee -a "$MAIN_LOG"
            ROLLOUT_OK=false
        fi
    done

    # 统计 rollout 结果
    TOTAL_ROLLS=$(cat "$ROLLOUT_DIR"/rank*.jsonl 2>/dev/null | wc -l)
    if [ "$TOTAL_ROLLS" -lt 50 ]; then
        echo "[ERROR] rollout 数量过少 ($TOTAL_ROLLS)，跳过本 round" | tee -a "$MAIN_LOG"
        continue
    fi

    CORRECT=$(cat "$ROLLOUT_DIR"/rank*.jsonl 2>/dev/null | $PYTHON -c "
import sys,json
lines=[json.loads(l) for l in sys.stdin if l.strip()]
c=sum(1 for l in lines if l.get('reward',0)>0.0)
print(f'{c}/{len(lines)} ({c/max(len(lines),1):.1%})')
" 2>/dev/null || echo "N/A")
    echo "Rollout 总数: $TOTAL_ROLLS  正确: $CORRECT" | tee -a "$MAIN_LOG"

    # ==========================================================================
    # Step 2: 生成 GRPO 训练数据（advantages）
    # ==========================================================================
    echo "" | tee -a "$MAIN_LOG"
    echo "[Step 2] 生成训练数据: $(date)" | tee -a "$MAIN_LOG"

    $PYTHON -u $DATA_SCRIPT \
        --rollout_dir "$ROLLOUT_DIR" \
        --output      "$TRAIN_DATA_PATH" \
        --plan        "$PLAN" \
        2>&1 | tee -a "$MAIN_LOG"

    TRAIN_SAMPLES=$(wc -l < "$TRAIN_DATA_PATH" 2>/dev/null || echo 0)
    echo "训练样本数: $TRAIN_SAMPLES" | tee -a "$MAIN_LOG"

    if [ "$TRAIN_SAMPLES" -lt 20 ]; then
        echo "[ERROR] 训练样本太少 ($TRAIN_SAMPLES)，跳过本 round" | tee -a "$MAIN_LOG"
        continue
    fi

    # ==========================================================================
    # Step 3: GRPO 训练（8 GPU DeepSpeed ZeRO-2）
    # ==========================================================================
    echo "" | tee -a "$MAIN_LOG"
    echo "[Step 3] GRPO 训练: $(date)" | tee -a "$MAIN_LOG"

    BASE_YAML=$AGENTIC_DIR/configs/grpo_plan_${PLAN_LOWER}.yaml
    TEMP_YAML=$LOG_BASE/grpo_r${ROUND}_${TIMESTAMP}.yaml
    sed "s|model_name_or_path:.*|model_name_or_path: $MODEL|;
         s|dataset_path:.*|dataset_path: $TRAIN_DATA_PATH|;
         s|output_dir:.*|output_dir: $OUT_MODEL|;
         s|run_name:.*|run_name: plan_${PLAN_LOWER}_r${ROUND}_${TIMESTAMP}|;
         s|learning_rate:.*|learning_rate: $LR|" \
        "$BASE_YAML" > "$TEMP_YAML"

    TRAIN_LOG=$LOG_BASE/train_r${ROUND}_${TIMESTAMP}.log

    cd "$VRPO_DIR"
    # 临时禁用 pipefail，防止 tee 磁盘写入失败（EDQUOT/ENOSPC）杀死整个脚本。
    # 训练本身（accelerate）的退出码通过 PIPESTATUS 单独检查。
    set +eo pipefail
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_TRAIN accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$N_GPUS" \
        "$TRAIN_SCRIPT" \
        --config "$TEMP_YAML" \
        2>&1 | tee "$TRAIN_LOG"
    _PIPE_ST=("${PIPESTATUS[@]}")
    set -eo pipefail
    _TRAIN_EC=${_PIPE_ST[0]}
    _TEE_EC=${_PIPE_ST[1]}
    if [ "$_TRAIN_EC" -ne 0 ]; then
        echo "[ERROR] 训练失败 (accelerate exit=$_TRAIN_EC)，跳过本 round" | tee -a "$MAIN_LOG"
        continue
    fi
    [ "$_TEE_EC" -ne 0 ] && echo "[WARN] tee 日志写入失败 (exit=$_TEE_EC)，但训练已完成" | tee -a "$MAIN_LOG" || true

    echo "" | tee -a "$MAIN_LOG"
    echo "[Step 3] 训练完成: $(date)" | tee -a "$MAIN_LOG"
    echo "模型已保存到: $OUT_MODEL" | tee -a "$MAIN_LOG"

    # 快速 eval（50 题采样，不跑全 600）
    echo "" | tee -a "$MAIN_LOG"
    echo "[Eval] 快速评测 50 题: $(date)" | tee -a "$MAIN_LOG"
    EVAL_OUT=$LOG_BASE/eval_r${ROUND}_${TIMESTAMP}.jsonl
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u $ROOT/my_eval/run_llada_eval.py \
        --model       "$OUT_MODEL" \
        --input       $VRPO_DIR/data/eval_600.jsonl \
        --output      "$EVAL_OUT" \
        --max_samples 50 \
        2>&1 | tail -5 | tee -a "$MAIN_LOG" &
    EVAL_PID=$!

    echo "" | tee -a "$MAIN_LOG"
    echo "Round $ROUND 完成! 下一 round 开始..." | tee -a "$MAIN_LOG"

    # 等待 eval（最多等 10 分钟，否则在后台继续）
    if wait -n "$EVAL_PID" 2>/dev/null; then
        ACC=$($PYTHON $ROOT/my_eval/cal_acc.py --data "$EVAL_OUT" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "N/A")
        echo "Quick eval ACC (50题): $ACC" | tee -a "$MAIN_LOG"
    fi

done  # end round loop

echo "" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"
echo "Agentic RL Plan $PLAN 全部 $MAX_ROUNDS rounds 完成: $(date)"      | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"
