#!/bin/bash
set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
SCRIPT=/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/track_gen_order.py
LOGDIR=/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order

mkdir -p $LOGDIR

echo "===== 启动 sft_llada (GPU 0) ====="
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    --model_name sft_llada --gpu 0 --n 30 --offset 0 \
    > $LOGDIR/sft_llada.log 2>&1 &
PID1=$!
echo "  PID=$PID1"

echo "===== 启动 x0pred (GPU 1) ====="
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
    --model_name x0pred --gpu 0 --n 30 --offset 0 \
    > $LOGDIR/x0pred.log 2>&1 &
PID2=$!
echo "  PID=$PID2"

echo "等待完成..."
wait $PID1 && echo "[sft_llada] 完成" || echo "[sft_llada] 失败"
wait $PID2 && echo "[x0pred]    完成" || echo "[x0pred]    失败"

echo "===== 全部结束 ====="
ls -lh $LOGDIR/*.json 2>/dev/null
