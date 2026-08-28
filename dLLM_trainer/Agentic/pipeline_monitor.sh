#!/bin/bash
# =============================================================================
# Pipeline Monitor
# 监控各 Phase 完成情况，自动串联启动下一阶段
# 在自己的 screen 里运行，不受 SSH 断线影响
# =============================================================================

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH

LOG=$AGENTIC/output/pipeline_monitor.log
exec 1> >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "Pipeline Monitor 启动: $(date)"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 等待 Phase 0 所有 shard 完成
# ─────────────────────────────────────────────────────────────────────────────
wait_phase0() {
    local PHASE0_DIR=$(ls -dt $AGENTIC/output/phase0_eval/2026* 2>/dev/null | head -1)
    echo "[Monitor] Phase 0 dir: $PHASE0_DIR"

    while true; do
        total=0
        done_shards=0
        for tag in sft plan_a; do
            for i in 0 1 2 3; do
                [ "$tag" = "plan_a" ] && gpu=$((i+4)) || gpu=$i
                f="$PHASE0_DIR/${tag}_shard${gpu}.jsonl"
                n=$(wc -l < "$f" 2>/dev/null || echo 0)
                total=$((total+n))
                [ "$n" -ge 150 ] && done_shards=$((done_shards+1))
            done
        done
        echo "  $(date +%H:%M:%S)  Phase0: ${total}/1200题  ${done_shards}/8 shard 完成"

        if [ $done_shards -ge 8 ]; then
            echo "[Monitor] Phase 0 全部完成!"
            # 合并
            for MODEL_TAG in sft plan_a; do
                MERGED="$PHASE0_DIR/${MODEL_TAG}_merged.jsonl"
                cat $PHASE0_DIR/${MODEL_TAG}_shard*.jsonl > "$MERGED"
                TOTAL=$(wc -l < "$MERGED")
                echo "[Monitor] 合并 $MODEL_TAG: $TOTAL 题 → $MERGED"
            done
            break
        fi
        sleep 120  # 每2分钟检查一次
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 打分 Phase 0
# ─────────────────────────────────────────────────────────────────────────────
score_phase0() {
    local PHASE0_DIR=$(ls -dt $AGENTIC/output/phase0_eval/2026* 2>/dev/null | head -1)
    echo ""
    echo "[Monitor] ====== Phase 0 打分 ======"
    for MODEL_TAG in sft plan_a; do
        MERGED="$PHASE0_DIR/${MODEL_TAG}_merged.jsonl"
        echo ""
        echo "--- $MODEL_TAG ---"
        $PYTHON $ROOT/my_eval/cal_acc.py --data "$MERGED" 2>&1
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 启动 Phase 2 (Round 0 rollout)
# ─────────────────────────────────────────────────────────────────────────────
launch_phase2() {
    echo ""
    echo "[Monitor] ====== 启动 Phase 2: Round 0 Rollout ======"

    # 先等 GPU 释放（Phase 0 完成后 GPU 应该已空）
    sleep 30

    screen -dmS round0_master bash $AGENTIC/run_round0_rollout.sh
    echo "[Monitor] Phase 2 round0_master screen 已启动"
    echo "[Monitor] 开始时间: $(date)"
}

# ─────────────────────────────────────────────────────────────────────────────
# 等待 Phase 2 完成，然后触发 Phase 3
# ─────────────────────────────────────────────────────────────────────────────
wait_phase2() {
    local OUT_DIR=$AGENTIC/output/round0_rollouts/round0
    echo ""
    echo "[Monitor] 等待 Phase 2 完成..."

    while true; do
        total=$(cat $OUT_DIR/rollouts_rank*.jsonl 2>/dev/null | wc -l || echo 0)
        running=$(screen -list 2>/dev/null | grep round0_rank | wc -l)
        echo "  $(date +%H:%M:%S)  Phase2: ${total}/2048 records  ${running} ranks running"

        # total>=1024 即可触发（128题×8），无需等满2048；ranks退出说明跑完了
        if [ $total -ge 1024 ] || ([ $running -eq 0 ] && [ $total -ge 500 ]); then
            echo "[Monitor] Phase 2 达到触发阈值 (${total} records)，开始 Phase 3！"
            break
        fi
        sleep 300  # 每5分钟检查一次
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

echo "[Monitor] 等待 Phase 0 完成..."
wait_phase0

echo "[Monitor] Phase 0 打分中..."
score_phase0

echo "[Monitor] 启动 Phase 2..."
launch_phase2

echo "[Monitor] 等待 Phase 2 完成..."
wait_phase2

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: 合并 + Reward 计算
# ─────────────────────────────────────────────────────────────────────────────
run_phase3() {
    local OUT_DIR=$AGENTIC/output/round0_rollouts
    echo ""
    echo "[Monitor] ====== Phase 3: 合并 + Reward 计算 ======"

    # 合并所有 rank 文件
    MERGED="$OUT_DIR/round0_merged.jsonl"
    cat $OUT_DIR/round0/rollouts_rank*.jsonl > "$MERGED" 2>/dev/null
    TOTAL=$(wc -l < "$MERGED")
    echo "[Monitor] 合并完成: $TOTAL records → $MERGED"

    # 计算 Reward A 和 B，生成训练文件
    $PYTHON $AGENTIC/compute_rewards.py \
        --input  "$MERGED" \
        --output "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/compute_rewards.log"

    echo "[Monitor] Phase 3 完成: $(date)"
    echo "  train_reward_a.jsonl: $(wc -l < $OUT_DIR/train_reward_a.jsonl 2>/dev/null || echo 0) records"
    echo "  train_reward_b.jsonl: $(wc -l < $OUT_DIR/train_reward_b.jsonl 2>/dev/null || echo 0) records"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Normalized ELBO-PG 训练（Reward A + B）
# ─────────────────────────────────────────────────────────────────────────────
launch_phase4() {
    echo ""
    echo "[Monitor] ====== 启动 Phase 4: Normalized ELBO-PG 训练 ======"

    # 等 Phase 2 rollout GPU 释放
    sleep 30

    screen -dmS phase4_elbopg bash $AGENTIC/run_phase4_elbopg.sh
    echo "[Monitor] Phase 4 screen 已启动: phase4_elbopg"
    echo "[Monitor] 查看日志: screen -r phase4_elbopg"
    echo "[Monitor] 开始时间: $(date)"
}

echo "[Monitor] 运行 Phase 3..."
run_phase3

echo "[Monitor] 启动 Phase 4..."
launch_phase4

echo ""
echo "================================================================"
echo "[Monitor] Phase 0 / 2 / 3 完成，Phase 4 已启动: $(date)"
echo "监控 Phase 4: screen -r phase4_elbopg"
echo "下一步: Phase 4 训练完成后对比 Reward A vs B 的 eval_600 得分"
echo "================================================================"
