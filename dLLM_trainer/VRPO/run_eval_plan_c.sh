#!/bin/bash
# Plan C RL 模型 eval（hercules，8×A40）
# 推理 200 题，LLM judge 打分，与 SFT baseline(42%/56%/54%) 对比

set -e
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
cd "$ROOT"

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_c/model/checkpoint-1
INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_100.jsonl
OUTPUT=$ROOT/dLLM_trainer/VRPO/output/rl/eval/plan_c/pred_100.jsonl
LOG=$ROOT/dLLM_trainer/VRPO/output/rl/eval/plan_c/eval_$(date +%Y%m%d_%H%M%S).log

mkdir -p $ROOT/dLLM_trainer/VRPO/output/rl/eval/plan_c

echo "[plan_c eval] 启动: $(date)" | tee "$LOG"
echo "模型: $MODEL" | tee -a "$LOG"
echo "数据: $INPUT (100题)" | tee -a "$LOG"

python my_eval/run_llada_eval.py \
    --model  "$MODEL" \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[plan_c eval] 推理完成，开始打分: $(date)" | tee -a "$LOG"

python my_eval/cal_acc.py --data "$OUTPUT" 2>&1 | tee -a "$LOG"

echo "[plan_c eval] 全部完成: $(date)" | tee -a "$LOG"
