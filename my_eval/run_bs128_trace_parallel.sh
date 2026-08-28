#!/bin/bash
# bs128 trace 8-GPU 并行推理启动脚本
# 在 hermes 上执行（8× A100 40GB）
set -euo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
SCRIPT=$ROOT/my_eval/run_llada_eval_trace_bs128.py
TRACE_DIR=$ROOT/my_eval/results/bs128/trace
WORLD=8

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH

echo "启动 bs128 trace，$WORLD GPU 并行，共 $(wc -l < $TRACE_DIR/bs128_wrong_qs.jsonl) 道题"

for RANK in $(seq 0 $((WORLD-1))); do
    SESSION="bs128_trace_r${RANK}"
    INNER=/tmp/bs128_trace_r${RANK}.sh
    TC=/tmp/triton_bs128_trace_${RANK}
    OUT=$TRACE_DIR/rank${RANK}.jsonl

    # 跳过已完成的 rank
    if [ -f "$OUT" ]; then
        DONE=$(wc -l < "$OUT")
        EXPECTED=$(( (158 + WORLD - 1) / WORLD ))
        if [ "$DONE" -ge "$EXPECTED" ]; then
            echo "rank${RANK} 已完成 ($DONE/$EXPECTED)，跳过"
            continue
        fi
    fi

    cat > $INNER << INNER
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${RANK}
export TRITON_CACHE_DIR=${TC}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ${TC}
cd $ROOT
${PYTHON} -u ${SCRIPT} \
    --rank ${RANK} --world ${WORLD} \
    --out ${OUT} \
    2>&1 | tee ${TRACE_DIR}/trace_r${RANK}.log
INNER
    chmod +x $INNER
    screen -dmS "$SESSION" bash $INNER
    echo "rank${RANK} 已启动 (GPU=${RANK}, screen=$SESSION)"
    sleep 5
done

echo ""
echo "全部启动！监控: screen -ls | grep bs128_trace"
echo "等待完成后运行: python $ROOT/my_eval/build_bs128_trace_viz.py"
