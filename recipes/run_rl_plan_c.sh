#!/bin/bash
# Plan C: RL 训练全流程（rollout → DPO → eval）
# 用法: cd /research/cbim/vast/mz751/Projects/DLLM-Searcher && bash recipes/run_rl_plan_c.sh
set -e
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
ACCELERATE=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/accelerate
ACCEL_CFG=/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/recipes/accelerate_configs/zero3.yaml

MODEL="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
ROLLOUT_DIR="$ROOT/dLLM_trainer/VRPO/output/rl/plan_c/rollouts"
DPO_DATA="$ROOT/dLLM_trainer/VRPO/data/dpo_pairs_plan_c.jsonl"
TRAIN_OUT="$ROOT/dLLM_trainer/VRPO/output/rl/plan_c/model"
EVAL_OUT="$ROOT/dLLM_trainer/VRPO/output/rl/plan_c/eval_100.jsonl"
LOG_DIR="$ROOT/dLLM_trainer/VRPO/output/rl/plan_c/logs"

mkdir -p "$ROLLOUT_DIR" "$LOG_DIR" "$(dirname $DPO_DATA)" "$TRAIN_OUT"
echo "====== Plan C 开始 ======"

# Step 1: 收集 rollout
echo "[Step 1] 收集 rollout (8 GPU × 4 rolls/题)..."
PIDS=()
for rank in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$rank $PYTHON \
        "$ROOT/my_eval/collect_latent_dpo_rollouts.py" \
        --model   "$MODEL" --input "$INPUT" \
        --output  "$ROLLOUT_DIR" --n_rolls 4 \
        --rank $rank --world 8 \
        > "$LOG_DIR/rollout_rank${rank}.log" 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid || echo "[WARN] PID $pid failed"; done
echo "[Step 1] 完成"

# Step 2: LLM judge + 配对
echo "[Step 2] LLM judge + DPO 配对..."
$PYTHON "$ROOT/my_eval/make_dpo_pairs.py" \
    --rollout_dir "$ROLLOUT_DIR" --output "$DPO_DATA" --max_workers 20
wc -l "$DPO_DATA"

# Step 3: DPO 训练
echo "[Step 3] DPO 训练 (Plan C)..."
cd "$ROOT/dLLM_trainer/VRPO"
$ACCELERATE launch --config_file "$ACCEL_CFG" --num_processes 8 \
    my_train/llada_dpo_train_c.py \
    --model_path "$MODEL" --data_path "$DPO_DATA" \
    --output_dir "$TRAIN_OUT" \
    --beta 0.1 --lr 5e-7 --epochs 1 \
    --batch_size 1 --grad_accum 8 --num_mc 2 --block_length 64 \
    2>&1 | tee "$LOG_DIR/train.log"
echo "[Step 3] 完成"

# Step 4: Eval
echo "[Step 4] 评测..."
EVAL_PIDS=()
OFFSETS=(0 13 26 39 52 65 78 91)
COUNTS=(13 13 13 13 13 13 13 9)
mkdir -p "$(dirname $EVAL_OUT)/eval_parts"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model "$TRAIN_OUT" --input "$INPUT" \
        --output "$(dirname $EVAL_OUT)/eval_parts/gpu${i}.jsonl" \
        --offset "${OFFSETS[$i]}" --max_samples "${COUNTS[$i]}" \
        > "$LOG_DIR/eval_gpu${i}.log" 2>&1 &
    EVAL_PIDS+=($!)
done
for pid in "${EVAL_PIDS[@]}"; do wait $pid; done
cat "$(dirname $EVAL_OUT)/eval_parts"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$EVAL_OUT"
$PYTHON "$ROOT/my_eval/cal_acc.py" --data "$EVAL_OUT"
echo "====== Plan C 全部完成！======"
