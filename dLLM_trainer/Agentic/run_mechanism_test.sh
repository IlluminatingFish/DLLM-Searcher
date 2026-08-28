#!/bin/bash
# =============================================================================
# Mechanism Test: 比较 Group A（raw ELBO）和 Group B（normalized ELBO）
#
# 数据来源：Plan A Round 1 的 rollout（480q × 8~16 rolls），
#           取前 64 个问题，reward 重新计算为 pure F1。
#
# Group A: block_length=64, pure_f1 reward, normalize_elbo=false (Round 1 baseline)
# Group B: block_length=64, pure_f1 reward, normalize_elbo=true  (+ Fix 2)
#
# 用法（在 hermes 或有空闲 GPU 的机器上）：
#   bash run_mechanism_test.sh
#
# 同时运行两组（用不同 GPU）：
#   GROUP=A CUDA_GPUS="0 1 2 3" bash run_mechanism_test.sh
#   GROUP=B CUDA_GPUS="4 5 6 7" bash run_mechanism_test.sh
# =============================================================================

set -eo pipefail

GROUP=${GROUP:-A}          # A or B
N_QUESTIONS=${N_QUESTIONS:-64}

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC_DIR=$ROOT/dLLM_trainer/Agentic
VRPO_DIR=$ROOT/dLLM_trainer/VRPO

# ── Python/PATH ──────────────────────────────────────────────────────────────
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

# ── GPU setup ─────────────────────────────────────────────────────────────────
if [ -n "$CUDA_GPUS" ]; then
    GPU_LIST=($CUDA_GPUS)
    N_GPUS=${#GPU_LIST[@]}
    CUDA_VIS=$(IFS=,; echo "${GPU_LIST[*]}")
else
    # 默认：homer/hermes 用 NUMA0 的 GPU 0-3
    CUDA_VIS="0,1,2,3"
    N_GPUS=4
fi
echo "Using GPUs: $CUDA_VIS  (N_GPUS=$N_GPUS)"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROLLOUT_DIR=$AGENTIC_DIR/output/plan_a/rollouts/round1/plan_a
MECH_DIR=$AGENTIC_DIR/output/mechanism_test
MECH_DATA=$MECH_DIR/grpo_train_${N_QUESTIONS}q_pure_f1.jsonl
OUT_MODEL=$MECH_DIR/group_${GROUP}/model
LOG_DIR=$MECH_DIR/group_${GROUP}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$MECH_DIR" "$OUT_MODEL" "$LOG_DIR"

ACCEL_CONFIG=$AGENTIC_DIR/configs/accel_zero2_${N_GPUS}gpu.yaml
TRAIN_SCRIPT=$VRPO_DIR/my_train/llada_grpo_train.py
INIT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized

echo "================================================================"
echo "Mechanism Test: Group $GROUP  $(date)"
echo "N_QUESTIONS=$N_QUESTIONS  N_GPUS=$N_GPUS"
echo "================================================================"

# =============================================================================
# Step 1: 准备数据（64 题 pure_f1 reward）
# =============================================================================
echo ""
echo "[Step 1] 准备机制测试数据"

if [ ! -f "$MECH_DATA" ]; then
    if [ ! -d "$ROLLOUT_DIR" ] || [ -z "$(ls $ROLLOUT_DIR/rank*.jsonl 2>/dev/null)" ]; then
        echo "[ERROR] Rollout 文件不存在: $ROLLOUT_DIR"
        echo "请先运行 Plan A Round 1 rollout，或修改 ROLLOUT_DIR 路径"
        exit 1
    fi

    echo "从 $ROLLOUT_DIR 生成 ${N_QUESTIONS}q pure_f1 训练数据..."
    $PYTHON -u $AGENTIC_DIR/make_agentic_data.py \
        --plan A \
        --rollout_dir "$ROLLOUT_DIR" \
        --output       "$MECH_DATA" \
        --reward_fn    pure_f1 \
        --n_questions  $N_QUESTIONS \
        --min_group    2

    TRAIN_SAMPLES=$(wc -l < "$MECH_DATA" 2>/dev/null || echo 0)
    echo "训练样本数: $TRAIN_SAMPLES"
else
    TRAIN_SAMPLES=$(wc -l < "$MECH_DATA" 2>/dev/null || echo 0)
    echo "数据已存在 ($TRAIN_SAMPLES 条)，跳过生成"
fi

if [ "$(wc -l < $MECH_DATA)" -lt 10 ]; then
    echo "[ERROR] 训练样本太少，退出"
    exit 1
fi

# =============================================================================
# Step 2: 生成训练 YAML（把 PLACEHOLDER 替换成真实路径）
# =============================================================================
echo ""
echo "[Step 2] 配置 Group $GROUP 训练参数"

BASE_YAML=$AGENTIC_DIR/configs/grpo_mech_$(echo $GROUP | tr A-Z a-z).yaml
TEMP_YAML=$LOG_DIR/grpo_mech_${GROUP}_${TIMESTAMP}.yaml
sed "s|model_name_or_path: PLACEHOLDER_MODEL|model_name_or_path: $INIT_MODEL|;
     s|dataset_path: PLACEHOLDER_DATA|dataset_path: $MECH_DATA|;
     s|output_dir: PLACEHOLDER_OUT|output_dir: $OUT_MODEL|;
     s|run_name: mech_group_.*|run_name: mech_group_${GROUP}_${TIMESTAMP}|" \
    "$BASE_YAML" > "$TEMP_YAML"

echo "YAML: $TEMP_YAML"
grep -E "block_length|normalize_elbo|learning_rate|num_train_epochs" "$TEMP_YAML" | sed 's/^/  /'

# =============================================================================
# Step 3: GRPO 训练
# =============================================================================
echo ""
echo "[Step 3] GRPO 训练 Group $GROUP: $(date)"
TRAIN_LOG=$LOG_DIR/train_mech_${GROUP}_${TIMESTAMP}.log

cd "$VRPO_DIR"
set +eo pipefail
CUDA_VISIBLE_DEVICES=$CUDA_VIS accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$N_GPUS" \
    "$TRAIN_SCRIPT" \
    --config "$TEMP_YAML" \
    2>&1 | tee "$TRAIN_LOG"
_EC=${PIPESTATUS[0]}
set -eo pipefail

if [ "$_EC" -ne 0 ]; then
    echo "[ERROR] 训练失败 (exit=$_EC)"
    exit 1
fi

# =============================================================================
# Step 4: 评测（32 题 dev set 快速评测）
# =============================================================================
echo ""
echo "[Step 4] 快速评测 32 题: $(date)"
EVAL_INPUT=$VRPO_DIR/data/eval_600.jsonl
EVAL_OUT=$LOG_DIR/eval_mech_${GROUP}_${TIMESTAMP}.jsonl

CUDA_VISIBLE_DEVICES=${CUDA_VIS%%,*} $PYTHON -u $ROOT/my_eval/run_llada_eval.py \
    --model       "$OUT_MODEL" \
    --input       "$EVAL_INPUT" \
    --output      "$EVAL_OUT" \
    --max_samples 32 \
    2>&1 | tee -a "$TRAIN_LOG" || echo "[WARN] eval 失败"

ACC=$($PYTHON $ROOT/my_eval/cal_acc.py --data "$EVAL_OUT" 2>/dev/null \
      | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "N/A")
echo ""
echo "================================================================"
echo "Group $GROUP 完成: $(date)"
echo "Eval ACC (32题): $ACC"
echo "模型: $OUT_MODEL"
echo "日志: $TRAIN_LOG"
echo "================================================================"
