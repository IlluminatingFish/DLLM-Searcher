#!/bin/bash
# =============================================================================
# GPU Snatcher: 监控 GPU 0-4，一旦 qt60 释放就立刻占上 Round 1 Rollout
#
# 策略：
#   - 每 20s 扫一次 GPU 0-4 的空闲内存
#   - 空闲 > 30GB 且没有我们的进程 → 立刻启动 round1 rollout rank
#   - world=8（和 GPU5/6/7 的 rank 合并成完整的 256题×8 rollout）
#   - rollout 支持断点续跑，随时可 kill 后重启正式任务
# =============================================================================

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

MODEL=$AGENTIC/output/phase4_elbopg/reward_a/model
INPUT=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_round1.jsonl
OUTPUT=$AGENTIC/output/round1_rollouts
LOG_DIR=$OUTPUT/logs
mkdir -p "$LOG_DIR"

FREE_THRESHOLD=18000   # MiB，超过此值认为卡空闲可用
WORLD=8

declare -A LAUNCHED   # 记录已启动的 rank

launch_rank() {
    local GPU=$1
    local RANK=$2
    local SESSION="round1_rank${RANK}"

    # 已启动且 screen 还在，跳过
    if screen -list | grep -q "${SESSION}"; then
        return
    fi

    echo "[$(date +%H:%M:%S)] GPU$GPU 空闲，启动 rank$RANK (world=$WORLD) → round1 rollout"

    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_${USER}

echo "=== rank$RANK (GPU$GPU) 开始 \$(date) ==="
$PYTHON -u $AGENTIC/collect_round0_rollouts.py \
    --rank     $RANK \
    --world    $WORLD \
    --model    "$MODEL" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --n_rolls  8 \
    --round_id 1 \
    2>&1 | tee "$LOG_DIR/rank${RANK}.log"
echo "=== rank$RANK 完成 exit=\$? \$(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh
    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    LAUNCHED[$RANK]=1
    sleep 15   # 错开模型加载，避免 NFS 同时读
}

echo "================================================================"
echo "GPU Snatcher 启动  $(date)"
echo "监控: GPU 0-4（threshold=${FREE_THRESHOLD}MiB）"
echo "目标: round1 rollout, world=$WORLD, π₁ policy"
echo "================================================================"

# GPU → Rank 映射（0-4 对应 rank 0-4，5/6/7 另外处理）
declare -A GPU_TO_RANK
GPU_TO_RANK[0]=0
GPU_TO_RANK[1]=1
GPU_TO_RANK[2]=2
GPU_TO_RANK[3]=3
GPU_TO_RANK[4]=4

while true; do
    ALL_DONE=true

    for GPU in 0 1 2 3 4; do
        RANK=${GPU_TO_RANK[$GPU]}
        SESSION="round1_rank${RANK}"

        # 已在跑，跳过
        if screen -list | grep -q "${SESSION}"; then
            ALL_DONE=false
            continue
        fi

        # 已完成（log 里有 "完成 exit=0"），跳过
        if grep -q "完成 exit=0" "$LOG_DIR/rank${RANK}.log" 2>/dev/null; then
            continue
        fi

        ALL_DONE=false

        # 检查 GPU 空闲内存
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null)
        if [ -z "$FREE" ]; then continue; fi

        if [ "$FREE" -gt "$FREE_THRESHOLD" ]; then
            launch_rank $GPU $RANK
        else
            echo "[$(date +%H:%M:%S)] GPU$GPU 空闲=${FREE}MiB，等待中..."
        fi
    done

    # 打印当前所有 rank 的进度摘要
    echo ""
    echo "[$(date +%H:%M:%S)] 进度快照:"
    for RANK in 0 1 2 3 4; do
        LOG="$LOG_DIR/rank${RANK}.log"
        if screen -list | grep -q "round1_rank${RANK}"; then
            PROG=$(tail -1 "$LOG" 2>/dev/null | grep -o "[0-9]*/[0-9]*" | tail -1)
            echo "  rank$RANK: 运行中 $PROG"
        elif grep -q "完成 exit=0" "$LOG" 2>/dev/null; then
            echo "  rank$RANK: ✓ 完成"
        else
            GPU=${RANK}
            FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null)
            echo "  rank$RANK: 等待 GPU$GPU 释放 (当前空闲 ${FREE}MiB)"
        fi
    done
    echo ""

    $ALL_DONE && break
    sleep 20
done

echo "================================================================"
echo "GPU Snatcher 完成: $(date)"
echo "GPU 0-4 的 rank 全部跑完，可运行 run_round1_rewards.sh"
echo "================================================================"
