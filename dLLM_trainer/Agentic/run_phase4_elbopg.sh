#!/bin/bash
# =============================================================================
# Phase 4: Normalized ELBO-PG 训练（Reward A 和 Reward B 各一次）
#
# 前提：Phase 3 compute_rewards.py 已生成
#       train_reward_a.jsonl 和 train_reward_b.jsonl
#
# 流程：
#   1. 串行训练 Reward A（8 GPU ZeRO-2）
#   2. 串行训练 Reward B（8 GPU ZeRO-2）
#   3. 每次训练后在 eval_600 上快速评测
#
# 用法：
#   bash run_phase4_elbopg.sh            # 两路都跑
#   REWARD=A bash run_phase4_elbopg.sh   # 只跑 Reward A
#   REWARD=B bash run_phase4_elbopg.sh   # 只跑 Reward B
# =============================================================================

set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export NCCL_DEBUG=WARN
# 把 torch/DeepSpeed JIT extension build cache 放到本地 /tmp（避免 NFS 文件锁卡住 8 个 rank）
export TORCH_EXTENSIONS_HOME=/tmp/torch_extensions_${USER}
export DS_BUILD_OPS=0          # 禁止 DeepSpeed 在启动时 pre-build ops（用 JIT on-demand）
export DS_SKIP_CUDA_CHECK=1    # 跳过 CUDA 版本检查，避免多 rank 重复打印 warning
# ZeRO-3：8B 模型 ZeRO-2 需要 44GB/卡（16参数+16梯度+12优化器）> A100 40GB
# ZeRO-3 参数分片后每卡仅 ~16GB，完全可行
export USE_ZERO_STAGE=3
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

ROLLOUT_DIR=$AGENTIC/output/round0_rollouts
PHASE4_DIR=$AGENTIC/output/phase4_elbopg
mkdir -p "$PHASE4_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PHASE4_DIR/phase4_${TIMESTAMP}.log"
exec 1> >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "Phase 4: Normalized ELBO-PG 训练  $(date)"
echo "================================================================"

# 确定要跑哪些 Reward（环境变量 REWARD=A 只跑A，未设置则跑 A 和 B）
REWARDS_TO_RUN=${REWARD:-"A B"}

# ── 检查训练数据 ──────────────────────────────────────────────────────────────
for RWD in $REWARDS_TO_RUN; do
    TRAIN_FILE="$ROLLOUT_DIR/train_reward_${RWD,,}.jsonl"
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "[ERROR] 训练文件不存在: $TRAIN_FILE"
        echo "  请先运行: python $AGENTIC/compute_rewards.py --input ... --output $ROLLOUT_DIR"
        exit 1
    fi
    N=$(wc -l < "$TRAIN_FILE")
    echo "  Reward $RWD: $TRAIN_FILE  ($N records)"
done

echo ""
echo "accel config: $AGENTIC/configs/accel_zero3_8gpu.yaml  (ZeRO-3)"

# ── 训练函数 ──────────────────────────────────────────────────────────────────
train_one_reward() {
    local REWARD=$1    # A 或 B（局部变量，不污染外层循环变量）
    local CONFIG="$AGENTIC/configs/grpo_phase4_reward_${REWARD,,}.yaml"
    local OUT_MODEL="$PHASE4_DIR/reward_${REWARD,,}/model"
    local TRAIN_LOG="$PHASE4_DIR/reward_${REWARD,,}/train.log"
    mkdir -p "$(dirname $TRAIN_LOG)"

    echo ""
    echo "================================================================"
    echo "训练 Reward ${REWARD}: $(date)"
    echo "  config: $CONFIG"
    echo "  output: $OUT_MODEL"
    echo "================================================================"

    cd "$VRPO"

    # 用 sed 覆盖 output_dir（确保路径正确）
    TEMP_YAML="/tmp/phase4_reward_${REWARD,,}_${TIMESTAMP}.yaml"
    sed "s|output_dir:.*|output_dir: $OUT_MODEL|;
         s|run_name:.*|run_name: phase4_elbopg_reward_${REWARD,,}_${TIMESTAMP}|" \
        "$CONFIG" > "$TEMP_YAML"

    set +e  # 防止训练失败整体退出
    accelerate launch \
        --config_file "$AGENTIC/configs/accel_zero3_8gpu.yaml" \
        --num_processes 8 \
        my_train/llada_grpo_train.py \
        --config "$TEMP_YAML" \
        2>&1 | tee "$TRAIN_LOG"
    TRAIN_EXIT=$?
    set -e

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[ERROR] Reward $REWARD 训练失败 (exit=$TRAIN_EXIT)"
        return $TRAIN_EXIT
    fi

    echo "[OK] Reward $REWARD 训练完成: $(date)"
    echo "  模型保存到: $OUT_MODEL"
}

# ── 快速 eval 函数（4 GPU 并行，eval_600 分片）────────────────────────────────
quick_eval() {
    local REWARD=$1
    local MODEL_PATH="$PHASE4_DIR/reward_${REWARD,,}/model"
    local EVAL_DIR="$PHASE4_DIR/reward_${REWARD,,}/eval_$(date +%H%M%S)"
    mkdir -p "$EVAL_DIR"

    echo ""
    echo "--- Phase 4 Reward ${REWARD} eval_600 快速评测 ---"

    # 找最新 checkpoint
    CKPT=$(ls -dt "$MODEL_PATH"/checkpoint-* 2>/dev/null | head -1 || echo "$MODEL_PATH")
    echo "  checkpoint: $CKPT"

    EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl

    # 4 GPU 并行（每 GPU 150 题）
    for GPU in 0 1 2 3; do
        OFFSET=$((GPU * 150))
        SESSION="phase4_eval_r${REWARD}_gpu${GPU}"
        cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -u $ROOT/my_eval/run_llada_eval.py \
    --model    "$CKPT" \
    --input    "$EVAL_INPUT" \
    --output   "$EVAL_DIR/shard${GPU}.jsonl" \
    --offset   $OFFSET \
    --max_samples 150 \
    2>&1 | tee "$EVAL_DIR/shard${GPU}.log"
INNER
        chmod +x /tmp/${SESSION}.sh
        screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
        echo "  launched GPU$GPU shard (offset=$OFFSET)"
        sleep 90  # 错开模型加载
    done

    # 等待完成
    echo "  等待 eval 完成..."
    while true; do
        DONE=0
        for GPU in 0 1 2 3; do
            screen -list | grep -q "phase4_eval_r${REWARD}_gpu${GPU}" || DONE=$((DONE+1))
        done
        [ $DONE -eq 4 ] && break
        sleep 60
    done

    # 合并打分
    cat "$EVAL_DIR/"shard*.jsonl > "$EVAL_DIR/merged.jsonl" 2>/dev/null
    N=$(wc -l < "$EVAL_DIR/merged.jsonl")
    echo "  合并后: $N 题"
    $PYTHON $ROOT/my_eval/cal_acc.py --data "$EVAL_DIR/merged.jsonl" \
        2>&1 | tee "$EVAL_DIR/score.log"
    echo "  eval 完成: $EVAL_DIR"
}

# ── 主流程 ────────────────────────────────────────────────────────────────────
for RWD in $REWARDS_TO_RUN; do
    train_one_reward "$RWD"
    echo ""
    echo "[Phase 4] Reward $RWD 训练完成，开始快速评测..."
    quick_eval "$RWD"
    echo ""
done

echo "================================================================"
echo "Phase 4 全部完成: $(date)"
echo "结果目录: $PHASE4_DIR"
echo ""
echo "下一步 (Phase 4 对比 & Phase 5):"
echo "  1. 比较 Reward A vs B 的 eval_600 得分"
echo "  2. 基于得分更好的 reward 进入 Phase 5 (Online vs Offline)"
echo "================================================================"
