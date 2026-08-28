#!/bin/bash
# =============================================================================
# Phase 2 Pilot：验证 prompt 格式修复是否生效
#   8 GPU × 每卡 1 道题 × 3 rollouts = 24 条 trajectories
#   确认模型确实调用搜索工具后，再全量运行 run_round0_rollout.sh
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
INPUT=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_256.jsonl
# Pilot 用独立目录，不污染正式输出
OUTPUT=$AGENTIC/output/round0_pilot
LOG_DIR=$OUTPUT/logs
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Phase 2 PILOT  $(date)"
echo "Model  : $MODEL"
echo "Config : 8 GPU × 1 题/rank × 3 rollouts = 24 条"
echo "Output : $OUTPUT"
echo "================================================================"

# 清理上次 pilot 残留
rm -f "$OUTPUT/round0/rollouts_rank"*.jsonl
mkdir -p "$OUTPUT/round0"

# 启动 8 个 rank，每隔 90s 一批（错开模型加载）
for RANK in $(seq 0 7); do
    SESSION="pilot_rank${RANK}"
    LOG="$LOG_DIR/rank${RANK}.log"

    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$RANK
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_${USER}

echo "=== pilot rank $RANK 开始 \$(date) ==="
$PYTHON -u $AGENTIC/collect_round0_rollouts.py \
    --rank          $RANK \
    --world         8 \
    --model         "$MODEL" \
    --input         "$INPUT" \
    --output        "$OUTPUT" \
    --n_rolls       3 \
    --max_questions 1 \
    --round_id      0 \
    2>&1 | tee "$LOG"
echo "=== pilot rank $RANK 完成 exit=\$? \$(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh

    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  rank $RANK launched (GPU=$RANK)"

    # 每 2 个 rank 后等 90s
    if [ $RANK -eq 1 ] || [ $RANK -eq 3 ] || [ $RANK -eq 5 ]; then
        echo "  等待 90s（错开模型加载）..."
        sleep 90
    fi
done

echo ""
echo "================================================================"
echo "8 个 pilot rank 已启动"
echo "监控: screen -list | grep pilot_rank"
echo "实时 log: tail -f $LOG_DIR/rank0.log"
echo "================================================================"

# 等待所有 rank 完成
echo ""
echo "等待 pilot 完成..."
while true; do
    DONE=0
    for RANK in $(seq 0 7); do
        screen -list | grep -q "pilot_rank${RANK}" || DONE=$((DONE+1))
    done
    echo "  $(date +%H:%M:%S)  完成 rank 数: $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 60
done

echo ""
echo "================================================================"
echo "Pilot 完成，开始验证..."
echo "================================================================"

# 自动调用验证脚本
bash $AGENTIC/verify_pilot.sh "$OUTPUT"
