#!/bin/bash
# =============================================================================
# Phase 0: Baseline 对比 eval
#   - SFT ckpt_6 (GPUs 0-3)  vs  Plan_A/round1 (GPUs 4-7)
#   - 评测集: eval_600.jsonl (600题, 含 short_answer)
#   - 每个模型 4 个并行进程, 各负责 150 题
#   - 用 screen -dmS 保证 SSH 断线后继续运行
#
# 用法（在 dionysos 上）:
#   bash phase0_eval.sh
#   bash phase0_eval.sh --merge   # 只做合并+打分（训练完成后）
# =============================================================================

set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC_DIR=$ROOT/dLLM_trainer/Agentic
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
EVAL_SCRIPT=$ROOT/my_eval/run_llada_eval.py
SCORE_SCRIPT=$ROOT/my_eval/cal_acc.py

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$AGENTIC_DIR/output/phase0_eval/$TIMESTAMP
mkdir -p "$OUT_DIR"

SFT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
PLAN_A_MODEL=$ROOT/dLLM_trainer/Agentic/output/plan_a/models/round1/model

echo "================================================================"
echo "Phase 0 Eval: $(date)"
echo "SFT model    : $SFT_MODEL"
echo "Plan_A model : $PLAN_A_MODEL"
echo "Eval input   : $EVAL_INPUT  ($(wc -l < $EVAL_INPUT) 题)"
echo "Output dir   : $OUT_DIR"
echo "================================================================"

# 合并+打分模式
if [ "${1:-}" == "--merge" ]; then
    for MODEL_TAG in sft plan_a; do
        echo ""
        echo "=== 合并 $MODEL_TAG 结果 ==="
        MERGED=$OUT_DIR/${MODEL_TAG}_merged.jsonl
        cat $OUT_DIR/${MODEL_TAG}_shard*.jsonl > "$MERGED" 2>/dev/null || \
            ls $AGENTIC_DIR/output/phase0_eval/*/  # 列出所有目录让用户选
        TOTAL=$(wc -l < "$MERGED")
        echo "$MODEL_TAG 合并后: $TOTAL 题"
        echo "打分中..."
        $PYTHON $SCORE_SCRIPT --data "$MERGED"
    done
    exit 0
fi

# ── 函数: 启动一个 eval shard ────────────────────────────────────────────────
launch_shard() {
    local MODEL_TAG=$1   # sft / plan_a
    local MODEL_PATH=$2
    local GPU=$3
    local OFFSET=$4
    local N=$5           # 每片题数
    local SHARD_OUT=$OUT_DIR/${MODEL_TAG}_shard${GPU}.jsonl
    local SHARD_LOG=$OUT_DIR/${MODEL_TAG}_shard${GPU}.log
    local SESSION="phase0_${MODEL_TAG}_gpu${GPU}"

    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_${USER}

echo "=== [GPU $GPU] $MODEL_TAG  offset=$OFFSET  n=$N  $(date) ==="
$PYTHON -u $EVAL_SCRIPT \
    --model    "$MODEL_PATH" \
    --input    "$EVAL_INPUT" \
    --output   "$SHARD_OUT" \
    --offset   $OFFSET \
    --max_samples $N \
    2>&1 | tee "$SHARD_LOG"
echo "=== [GPU $GPU] DONE  exit=\$?  $(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh

    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  launched screen[$SESSION]  GPU=$GPU  offset=$OFFSET  n=$N  log=$SHARD_LOG"
}

# ── 交错启动，避免 8 进程同时从 NFS 读取同一模型文件造成死锁 ──────────────────
# 策略：先启动 SFT GPU0 和 Plan_A GPU4（两个不同模型），等候 NFS 缓存热身
# 再每隔 90s 依次启动剩余 shard

echo ""
echo "[交错启动] 每 90s 启动一个 shard，避免 NFS 并发死锁"

# t=0: SFT GPU0 和 Plan_A GPU4 同时起（不同模型文件，无冲突）
launch_shard sft    "$SFT_MODEL"     0   0  150
launch_shard plan_a "$PLAN_A_MODEL"  4   0  150

echo "  等待 90s (模型预热)..."
sleep 90

# t=90s: SFT GPU1 和 Plan_A GPU5
launch_shard sft    "$SFT_MODEL"     1 150  150
launch_shard plan_a "$PLAN_A_MODEL"  5 150  150

echo "  等待 90s ..."
sleep 90

# t=3min: SFT GPU2 和 Plan_A GPU6
launch_shard sft    "$SFT_MODEL"     2 300  150
launch_shard plan_a "$PLAN_A_MODEL"  6 300  150

echo "  等待 90s ..."
sleep 90

# t=4.5min: SFT GPU3 和 Plan_A GPU7
launch_shard sft    "$SFT_MODEL"     3 450  150
launch_shard plan_a "$PLAN_A_MODEL"  7 450  150

echo ""
echo "================================================================"
echo "全部 8 个 shard 已在 screen 中启动"
echo "监控: screen -list"
echo "查看某个 shard: screen -r phase0_sft_gpu0"
echo "查看日志: tail -f $OUT_DIR/sft_shard0.log"
echo ""
echo "完成后合并打分:"
echo "  bash $AGENTIC_DIR/phase0_eval.sh --merge"
echo "================================================================"

# ── 等待所有 shard 完成并自动合并打分 ────────────────────────────────────────
echo ""
echo "等待所有 shard 完成..."
sleep 30

while true; do
    DONE=0
    for TAG in sft plan_a; do
        for GPU in 0 1 2 3; do
            [ "$TAG" == "plan_a" ] && GPU=$((GPU+4))
            SESSION="phase0_${TAG}_gpu${GPU}"
            screen -list | grep -q "$SESSION" || DONE=$((DONE+1))
        done
    done
    echo "  $(date)  完成 shard 数: $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 120
done

echo ""
echo "所有 shard 完成，开始合并打分..."

for MODEL_TAG in sft plan_a; do
    MERGED=$OUT_DIR/${MODEL_TAG}_merged.jsonl
    cat $OUT_DIR/${MODEL_TAG}_shard*.jsonl > "$MERGED"
    TOTAL=$(wc -l < "$MERGED")
    echo ""
    echo "================================================================"
    echo "=== $MODEL_TAG  合并后: $TOTAL 题 ==="
    echo "================================================================"
    $PYTHON $SCORE_SCRIPT --data "$MERGED" 2>&1 | tee "$OUT_DIR/${MODEL_TAG}_score.log"
done

echo ""
echo "Phase 0 评测完成: $(date)"
echo "结果目录: $OUT_DIR"
