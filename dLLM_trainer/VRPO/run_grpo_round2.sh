#!/bin/bash
# GRPO Round 2: 等 rollout 完成 → 处理数据 → 训练
# 从 Round 1 model 开始，on-policy 数据，1 epoch，lr=2e-7

set -e
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
cd "$ROOT"

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROLLOUT_DIR=dLLM_trainer/VRPO/output/rl/grpo/rollouts/round2
DATA_OUT=dLLM_trainer/VRPO/data/grpo_train_round2.jsonl
CONFIG=dLLM_trainer/VRPO/output/rl/grpo/logs/round2/grpo_round2_20260625_050354.yaml
TRAIN_LOG=dLLM_trainer/VRPO/output/rl/grpo/logs/round2/train_$(date +%Y%m%d_%H%M%S).log

echo "=== 等待 Round 2 rollout 完成 ==="

# 等所有 8 个 rank 完成（判断条件：每个 rank 日志出现 "完成！" 或 rank*.jsonl 稳定）
NQUESTIONS=477  # 每 rank 约 60 题
while true; do
    total_done=0
    for i in $(seq 0 7); do
        n=$(wc -l < "$ROLLOUT_DIR/rank${i}.jsonl" 2>/dev/null || echo 0)
        total_done=$((total_done + n))
    done
    # 477 题 × 8 rolls = 3816 条 rollout
    echo "$(date +%H:%M) rollout 总条数: $total_done / ~3816"
    if [ "$total_done" -ge 3750 ]; then
        echo "rollout 已足够，继续处理"
        break
    fi
    sleep 300
done

echo ""
echo "=== 处理 Round 2 训练数据 ==="
python my_eval/make_grpo_data.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --output "$DATA_OUT"

echo ""
echo "=== 启动 Round 2 GRPO 训练 ==="
mkdir -p /common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/grpo/round2/model

accelerate launch \
    --config_file dLLM_trainer/VRPO/recipes/accelerate_configs/zero3_cpu_offload.yaml \
    --num_processes 8 \
    dLLM_trainer/VRPO/my_train/llada_grpo_train.py \
    --config "$CONFIG" \
    2>&1 | tee "$TRAIN_LOG"

echo "=== Round 2 训练完成: $(date) ==="

# ── Round 2 Eval ────────────────────────────────────────────────────────────
echo ""
echo "=== 启动 Round 2 Eval ==="
bash dLLM_trainer/VRPO/run_eval_grpo_round2.sh
