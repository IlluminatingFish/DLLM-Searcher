#!/bin/bash
# 全力跑：补 ckpt_4 缺失 shard + ckpt_6 全量
# GPU 1/3/5 立刻跑 ckpt_4；等 ckpt_5 结束后全卡跑 ckpt_6

set -u
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL_BASE="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_v2"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
RESULTS="$ROOT/my_eval/results/bs128_v2"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
EVAL="$ROOT/my_eval/run_llada_eval.py"
CAL="$ROOT/my_eval/cal_acc.py"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$RESULTS"

# ── ① 立刻补跑 ckpt_4 GPU1/2/3 三个 shard（用物理 GPU 1/3/5，结果文件命名为 gpu1/2/3）──
log "===== 补跑 ckpt_4 缺失 shard（GPU 1,3,5）====="
D4="$RESULTS/ckpt4_parallel"
mkdir -p "$D4"
rm -f "$D4/gpu1.jsonl" "$D4/gpu2.jsonl" "$D4/gpu3.jsonl"

CUDA_VISIBLE_DEVICES=1 $PYTHON "$EVAL" --model "$MODEL_BASE/ckpt_4/optimized" \
    --input "$INPUT" --output "$D4/gpu1.jsonl" \
    --offset 0   --max_samples 100 --block_size 128 --num_steps 128 \
    > "$D4/gpu1_fix.log" 2>&1 &
P4_1=$!; log "  GPU1 → ckpt4 offset=0   PID=$P4_1"

CUDA_VISIBLE_DEVICES=3 $PYTHON "$EVAL" --model "$MODEL_BASE/ckpt_4/optimized" \
    --input "$INPUT" --output "$D4/gpu2.jsonl" \
    --offset 100 --max_samples 100 --block_size 128 --num_steps 128 \
    > "$D4/gpu2_fix.log" 2>&1 &
P4_2=$!; log "  GPU3 → ckpt4 offset=100 PID=$P4_2"

CUDA_VISIBLE_DEVICES=5 $PYTHON "$EVAL" --model "$MODEL_BASE/ckpt_4/optimized" \
    --input "$INPUT" --output "$D4/gpu3.jsonl" \
    --offset 200 --max_samples 100 --block_size 128 --num_steps 128 \
    > "$D4/gpu3_fix.log" 2>&1 &
P4_3=$!; log "  GPU5 → ckpt4 offset=200 PID=$P4_3"

# ── ② 等 ckpt_5 GPU2 跑完，合并 ckpt_5 ──
log "===== 等待 ckpt_5 GPU2 完成 ====="
wait "$P4_1" "$P4_2" "$P4_3"   # ckpt_4 补跑和 ckpt_5 收尾可以并行等

# 等 GPU2 上 ckpt_5 的进程结束（内存降下来）
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=2 2>/dev/null | tr -d ' MiB')
    [ "${MEM:-99999}" -lt 5000 ] && break
    log "  GPU2 仍在用 ${MEM}MiB，等 20s..."
    sleep 20
done
log "GPU2 已释放"
sleep 15

# 合并 ckpt_5
D5="$RESULTS/ckpt5_parallel"
cat "$D5"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt5_merged.jsonl" 2>/dev/null
N5=$(wc -l < "$RESULTS/ckpt5_merged.jsonl")
log "ckpt_5 合并: $N5 题"
log "--- ckpt_5 正确率 ---"
$PYTHON "$CAL" --data "$RESULTS/ckpt5_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer"

# 合并 ckpt_4（等补跑进程结束）
log "--- ckpt_4 正确率 ---"
cat "$D4"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt4_merged.jsonl" 2>/dev/null
N4=$(wc -l < "$RESULTS/ckpt4_merged.jsonl")
log "ckpt_4 合并: $N4 题"
$PYTHON "$CAL" --data "$RESULTS/ckpt4_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer"

log "等 60s 让 CUDA 内存释放..."
sleep 60

# ── ③ 全卡跑 ckpt_6（GPU 1/2/3/5/6/7）──
log "===== ckpt_6 全量 eval（6 GPU）====="
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
    log "  GPU$GPU → ckpt6 offset=${OFFSETS[$i]} PID=$!"
done

log "等待 ckpt_6 全部完成..."
FAILED=0
for i in 0 1 2 3 4 5; do
    wait ${PIDS6[$i]} || { log "[ERROR] GPU ${GPUS[$i]} 失败"; FAILED=1; }
    N=$(wc -l < "$D6/gpu${GPUS[$i]}.jsonl" 2>/dev/null || echo 0)
    log "  GPU${GPUS[$i]} 完成: $N 题"
done

[ $FAILED -eq 0 ] && cat "$D6"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt6_merged.jsonl"
N6=$(wc -l < "$RESULTS/ckpt6_merged.jsonl" 2>/dev/null || echo 0)
log "ckpt_6 合并: $N6 题"
log "--- ckpt_6 正确率 ---"
$PYTHON "$CAL" --data "$RESULTS/ckpt6_merged.jsonl" 2>/dev/null | grep -E "CEM-1|Total|Answer"

log "===== 全部完成！====="
