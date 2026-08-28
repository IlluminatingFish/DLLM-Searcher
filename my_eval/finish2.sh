#!/bin/bash
# 等 GPU3/5（PID 3710666/3710668）完成，合并 ckpt_4/5，再跑 ckpt_6 全量
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
RESULTS="$ROOT/my_eval/results/bs128_v2"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
EVAL="$ROOT/my_eval/run_llada_eval.py"
CAL="$ROOT/my_eval/cal_acc.py"
MODEL_BASE="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_v2"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 等 GPU3/5 的 eval 进程完成（轮询文件大小）
log "等待 GPU3/5 进程（PID 3710666, 3710668）完成..."
wait 3710666 2>/dev/null
wait 3710668 2>/dev/null
log "GPU3/5 完成"

# 合并 ckpt_4（gpu1=12题，gpu2=55→100题，gpu3=38→100题，gpu5/6/7=100题各）
cat "$RESULTS/ckpt4_parallel"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt4_merged.jsonl" 2>/dev/null
N4=$(wc -l < "$RESULTS/ckpt4_merged.jsonl")
log "ckpt_4 合并: $N4 题"
$PYTHON "$CAL" --data "$RESULTS/ckpt4_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer rate"
echo ""

# 合并 ckpt_5
cat "$RESULTS/ckpt5_parallel"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt5_merged.jsonl" 2>/dev/null
N5=$(wc -l < "$RESULTS/ckpt5_merged.jsonl")
log "ckpt_5 合并: $N5 题"
$PYTHON "$CAL" --data "$RESULTS/ckpt5_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer rate"
echo ""

log "等 60s 让内存释放..."
sleep 60

# ckpt_6 全量 eval（6 GPU）
log "===== ckpt_6 全量 eval ====="
D6="$RESULTS/ckpt6_parallel"
mkdir -p "$D6"

GPUS=(1 2 3 5 6 7)
OFFSETS=(0 100 200 300 400 500)
PIDS6=()
for i in 0 1 2 3 4 5; do
    GPU=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL" \
        --model "$MODEL_BASE/ckpt_6/optimized" \
        --input "$INPUT" --output "$D6/gpu${GPU}.jsonl" \
        --offset "${OFFSETS[$i]}" --max_samples 100 \
        --block_size 128 --num_steps 128 \
        > "$D6/gpu${GPU}.log" 2>&1 &
    PIDS6+=($!)
    log "  GPU${GPU} → ckpt6 offset=${OFFSETS[$i]} PID=$!"
done

log "等待 ckpt_6 全部完成..."
for i in 0 1 2 3 4 5; do
    wait ${PIDS6[$i]} || log "[WARN] GPU${GPUS[$i]} 失败"
    N=$(wc -l < "$D6/gpu${GPUS[$i]}.jsonl" 2>/dev/null || echo 0)
    log "  GPU${GPUS[$i]}: $N 题完成"
done

cat "$D6"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt6_merged.jsonl"
N6=$(wc -l < "$RESULTS/ckpt6_merged.jsonl")
log "ckpt_6 合并: $N6 题"
$PYTHON "$CAL" --data "$RESULTS/ckpt6_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer rate"

log "===== 全部完成！====="
