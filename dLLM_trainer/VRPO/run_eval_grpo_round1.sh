#!/bin/bash
# GRPO Round 1 模型 eval（dionysos，单 GPU）
# 与 Plan A(55%) / Plan B(54%) 对比

set -e
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
cd "$ROOT"

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

MODEL=/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/grpo/round1/model
INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_100.jsonl
OUTDIR=$ROOT/dLLM_trainer/VRPO/output/rl/eval/grpo_round1
OUTPUT=$OUTDIR/pred_100.jsonl
LOG=$OUTDIR/eval_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$OUTDIR"

echo "[grpo_round1 eval] 启动: $(date)" | tee "$LOG"
echo "模型: $MODEL" | tee -a "$LOG"
echo "数据: $INPUT (100题)" | tee -a "$LOG"

python my_eval/run_llada_eval.py \
    --model  "$MODEL" \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[grpo_round1 eval] 推理完成，开始打分: $(date)" | tee -a "$LOG"

python my_eval/cal_acc.py --data "$OUTPUT" 2>&1 | tee -a "$LOG"

echo "[grpo_round1 eval] 全部完成: $(date)" | tee -a "$LOG"
