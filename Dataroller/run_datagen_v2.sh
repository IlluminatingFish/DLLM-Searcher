#!/bin/bash
# SFT 数据生成全流程（三步）
# 用法: bash run_datagen_v2.sh
# 在 hercules 上跑，8 个并行进程分片

set -e
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
SCRIPT=/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/generate_sft_data.py
OUT=/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/sft_output_v2
DATASETS="hotpotqa,2wikimultihopqa"
SAMPLES=2048        # 每个数据集采样题数
EVAL_SAMPLES=500    # 每个数据集测试集题数
WORKERS=10          # 每个分片的并行 API workers（8进程×10=80并发，避免限速）
WORLD=8

mkdir -p "$OUT"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "Step 1: 生成测试集 (val split, 无需 API)"
echo "======================================================"
$PYTHON "$SCRIPT" \
    --mode eval \
    --datasets "$DATASETS" \
    --eval_samples_per_dataset "$EVAL_SAMPLES" \
    --output_dir "$OUT"
echo "测试集生成完毕 → $OUT/eval_test_set.jsonl"

echo ""
echo "======================================================"
echo "Step 2: 8 进程并行 rollout (train split + GPT-4o)"
echo "======================================================"
PIDS=()
for rank in 0 1 2 3 4 5 6 7; do
    LOG="$LOG_DIR/sft_rank${rank}.log"
    $PYTHON "$SCRIPT" \
        --mode sft \
        --datasets "$DATASETS" \
        --samples_per_dataset "$SAMPLES" \
        --rank "$rank" \
        --world_size "$WORLD" \
        --max_workers "$WORKERS" \
        --output_dir "$OUT" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  Rank $rank 已启动 (PID=${PIDS[-1]}) → $LOG"
done

echo "等待所有 rank 完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Rank $i 完成"
    else
        echo "  Rank $i 失败！"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "有 rank 失败，请检查 $LOG_DIR 下的日志"
    exit 1
fi

echo ""
echo "======================================================"
echo "Step 3: 合并分片 + 过滤 + 生成最终 SFT 数据"
echo "======================================================"
$PYTHON "$SCRIPT" \
    --mode merge \
    --output_dir "$OUT" \
    --max_workers 50

echo ""
echo "======================================================"
echo "全部完成！"
echo "  训练数据 → $OUT/sft_train_*.json"
echo "  测试集   → $OUT/eval_test_set.jsonl"
echo "======================================================"
