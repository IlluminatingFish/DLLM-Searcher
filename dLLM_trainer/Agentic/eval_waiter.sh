#!/bin/bash
# 等训练完成，立刻 8 卡抢占 eval_600

LOG="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/phase4_elbopg_round2/reward_a/train_20260810_120018.log"
OUT_MODEL="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/phase4_elbopg_round2/reward_a/model"
EVAL_INPUT="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/eval_600.jsonl"
EVAL_DIR="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/phase4_elbopg_round2/reward_a/eval_123044"
PYTHON="/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python"
ROOT="/research/cbim/vast/mz751/Projects/DLLM-Searcher"

echo "[$(date +%H:%M:%S)] 监控训练日志: $LOG"

# 等待训练完成（日志出现 "Training completed" 或 "train_loss"）
until grep -q "train_loss\|Training completed\|=== Round 2 完成" "$LOG" 2>/dev/null; do
    STEP=$(grep -o "[0-9]*/26" "$LOG" 2>/dev/null | tail -1)
    echo "[$(date +%H:%M:%S)] 训练中... $STEP"
    sleep 20
done

echo "[$(date +%H:%M:%S)] 训练完成！立刻启动 8 卡 eval_600"
mkdir -p "$EVAL_DIR"

# 8 GPU 并行，每卡 75 题，错开 30s 加载模型
for GPU in $(seq 0 7); do
    OFFSET=$((GPU * 75))
    SESSION="r2eval_gpu${GPU}"
    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751
echo "=== eval GPU$GPU offset=$OFFSET $(date) ==="
$PYTHON -u $ROOT/my_eval/run_llada_eval.py \
    --model    "$OUT_MODEL" \
    --input    "$EVAL_INPUT" \
    --output   "$EVAL_DIR/shard${GPU}.jsonl" \
    --offset   $OFFSET \
    --max_samples 75 \
    2>&1 | tee "$EVAL_DIR/shard${GPU}.log"
echo "=== eval GPU$GPU 完成 exit=$? $(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh
    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  GPU$GPU launched (offset=$OFFSET)"
    sleep 30
done

echo "[$(date +%H:%M:%S)] 全部 8 个 eval 进程已启动"

# 等待所有完成
while true; do
    DONE=0
    for GPU in $(seq 0 7); do
        screen -list | grep -q "r2eval_gpu${GPU}" || DONE=$((DONE+1))
    done
    echo "[$(date +%H:%M:%S)] eval 完成 ${DONE}/8"
    [ $DONE -eq 8 ] && break
    sleep 60
done

# 合并打分
echo "[$(date +%H:%M:%S)] 合并并打分..."
cat "$EVAL_DIR"/shard*.jsonl > "$EVAL_DIR/merged.jsonl"
N=$(wc -l < "$EVAL_DIR/merged.jsonl")
echo "合并 $N 题"
$PYTHON $ROOT/my_eval/cal_acc.py --data "$EVAL_DIR/merged.jsonl" | tee "$EVAL_DIR/score.log"
echo "[$(date +%H:%M:%S)] eval_600 完成！结果: $EVAL_DIR/score.log"
