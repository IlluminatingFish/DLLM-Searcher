#!/bin/bash
# =============================================================================
# GRPO 完整 Pipeline（hercules, 8×A40）
#
# 每个 round 分三步：
#   Step 1: 8 卡并行 rollout 生成（每卡一份，共 N_QUESTIONS 题 × N_ROLLS 次）
#   Step 2: 合并 rollout，计算 advantages，生成训练数据
#   Step 3: 8 卡 DeepSpeed ZeRO-3 GRPO 训练
#
# 用法:
#   bash run_grpo.sh             # 从 round 1 开始
#   ROUND=2 bash run_grpo.sh     # 从指定 round 开始（沿用上一轮的模型）
# =============================================================================

set -e
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
cd "$ROOT"

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 参数 ──────────────────────────────────────────────────────────────────────
ROUND=${ROUND:-1}
N_QUESTIONS=500       # 每 round 用多少道题做 rollout（控制 API 成本）
N_ROLLS=8             # 每题 rollout 次数（控制组内方差）
N_GPUS=8

# 初始模型：Plan A 最终 checkpoint（eval 56%，最佳起点）
INIT_MODEL=/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_a/model/checkpoint-3
COMMON_BASE=/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/grpo

# 训练数据（每 round 覆盖写入，训练完后即可删除节省空间）
GRPO_DATA=$ROOT/dLLM_trainer/VRPO/data/grpo_train.jsonl

# Accelerate 配置（ZeRO-3 + CPU offload，省显存）
ACCEL_CONFIG=$ROOT/dLLM_trainer/VRPO/recipes/accelerate_configs/zero3_cpu_offload.yaml

# ── 确定当前 round 使用的模型 ─────────────────────────────────────────────────
if [ "$ROUND" -eq 1 ]; then
    MODEL=$INIT_MODEL
else
    PREV_ROUND=$((ROUND - 1))
    MODEL=$COMMON_BASE/round${PREV_ROUND}/model
    if [ ! -d "$MODEL" ]; then
        echo "[ERROR] 上一 round 的模型不存在: $MODEL"
        exit 1
    fi
fi

OUT_MODEL=$COMMON_BASE/round${ROUND}/model
ROLLOUT_DIR=$ROOT/dLLM_trainer/VRPO/output/rl/grpo/rollouts/round${ROUND}
LOG_DIR=$ROOT/dLLM_trainer/VRPO/output/rl/grpo/logs/round${ROUND}
mkdir -p "$ROLLOUT_DIR" "$LOG_DIR" "$OUT_MODEL"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/grpo_round${ROUND}_${TIMESTAMP}.log

echo "================================================================" | tee "$LOG_FILE"
echo "GRPO Round ${ROUND}  $(date)"                                     | tee -a "$LOG_FILE"
echo "模型:       $MODEL"                                                | tee -a "$LOG_FILE"
echo "题目数:     $N_QUESTIONS × $N_ROLLS rolls = $((N_QUESTIONS * N_ROLLS)) rollouts" | tee -a "$LOG_FILE"
echo "输出模型:   $OUT_MODEL"                                            | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# =============================================================================
# Step 1: Rollout 生成（8 卡并行）
# =============================================================================
echo ""
echo "[Step 1] Rollout 生成: $(date)" | tee -a "$LOG_FILE"

PIDS=()
for RANK in $(seq 0 $((N_GPUS - 1))); do
    RANK_LOG=$LOG_DIR/rollout_rank${RANK}_${TIMESTAMP}.log
    CUDA_VISIBLE_DEVICES=$RANK python my_eval/collect_grpo_rollouts.py \
        --model       "$MODEL" \
        --input       "$ROOT/dLLM_trainer/VRPO/data/sft_train_as_eval.jsonl" \
        --output      "$ROLLOUT_DIR" \
        --n_questions "$N_QUESTIONS" \
        --n_rolls     "$N_ROLLS" \
        --rank        "$RANK" \
        --world       "$N_GPUS" \
        > "$RANK_LOG" 2>&1 &
    PIDS+=($!)
done

echo "等待 $N_GPUS 个 rollout 进程完成..."
for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "[ERROR] rollout 进程失败 PID=$PID"; exit 1; }
done
echo "[Step 1] Rollout 完成: $(date)" | tee -a "$LOG_FILE"

# 统计 rollout 结果
TOTAL_ROLLS=$(cat "$ROLLOUT_DIR"/rank*.jsonl | wc -l)
CORRECT=$(cat "$ROLLOUT_DIR"/rank*.jsonl | python3 -c "
import sys, json
lines = [json.loads(l) for l in sys.stdin if l.strip()]
correct = sum(1 for l in lines if l.get('reward', 0) > 0)
print(f'{correct}/{len(lines)} ({correct/max(len(lines),1):.1%})')
")
echo "Rollout 总数: $TOTAL_ROLLS，正确: $CORRECT" | tee -a "$LOG_FILE"

# =============================================================================
# Step 2: 生成 GRPO 训练数据
# =============================================================================
echo ""
echo "[Step 2] 生成训练数据: $(date)" | tee -a "$LOG_FILE"

python my_eval/make_grpo_data.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --output      "$GRPO_DATA" \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_SAMPLES=$(wc -l < "$GRPO_DATA")
echo "训练样本数: $TRAIN_SAMPLES" | tee -a "$LOG_FILE"

if [ "$TRAIN_SAMPLES" -lt 10 ]; then
    echo "[ERROR] 训练样本太少（$TRAIN_SAMPLES），退出"
    exit 1
fi

# =============================================================================
# Step 3: GRPO 训练
# =============================================================================
echo ""
echo "[Step 3] GRPO 训练: $(date)" | tee -a "$LOG_FILE"

# 更新 yaml 的 model_name_or_path 和 output_dir
GRPO_YAML=$ROOT/dLLM_trainer/VRPO/recipes/grpo_llada.yaml
TEMP_YAML=$LOG_DIR/grpo_round${ROUND}.yaml
sed "s|model_name_or_path:.*|model_name_or_path: $MODEL|;
     s|dataset_path:.*|dataset_path: $GRPO_DATA|;
     s|output_dir:.*|output_dir: $OUT_MODEL|;
     s|run_name:.*|run_name: grpo_round${ROUND}_${TIMESTAMP}|" \
    "$GRPO_YAML" > "$TEMP_YAML"

TRAIN_LOG=$LOG_DIR/train_${TIMESTAMP}.log
cd "$ROOT/dLLM_trainer/VRPO"

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$N_GPUS" \
    my_train/llada_grpo_train.py \
    --config "$TEMP_YAML" \
    2>&1 | tee "$TRAIN_LOG"

echo ""
echo "[Step 3] 训练完成: $(date)" | tee -a "$LOG_FILE"
echo "模型已保存到: $OUT_MODEL" | tee -a "$LOG_FILE"

echo ""
echo "================================================================" | tee -a "$LOG_FILE"
echo "Round ${ROUND} 全部完成: $(date)"                                 | tee -a "$LOG_FILE"
echo "下一步: ROUND=$((ROUND + 1)) bash run_grpo.sh"                   | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
