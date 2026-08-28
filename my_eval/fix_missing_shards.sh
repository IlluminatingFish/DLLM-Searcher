#!/bin/bash
# 补跑 ckpt_2 和 ckpt_3 缺失的 GPU 1/2/3 shard（各 100 题）
# 用 GPU 1,2,3（确认 ckpt_3 GPU7 完成后再跑）
set -u
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL_BASE="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_v2"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
RESULTS="$ROOT/my_eval/results/bs128_v2"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
EVAL_SCRIPT="$ROOT/my_eval/run_llada_eval.py"
CAL_ACC="$ROOT/my_eval/cal_acc.py"

# 等 GPU7 上 ckpt_3 的进程结束
echo "[$(date)] 等待 GPU 7 进程结束..."
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q ""; do
    GPU7_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=7 2>/dev/null | head -1 | tr -d ' MiB')
    if [ "${GPU7_MEM:-0}" -lt 5000 ] 2>/dev/null; then
        break
    fi
    echo "  GPU 7 仍在用 ${GPU7_MEM} MiB，等 30s..."
    sleep 30
done
echo "[$(date)] GPU 7 已释放，开始补跑..."
sleep 15  # 额外等待释放

# ckpt_2 和 ckpt_3 的 GPU 1/2/3 shard（offset=0/100/200，各 100 题）
FIX_GPUS=(1 2 3)
OFFSETS=(0 100 200)

for CKPT in 2 3; do
    MODEL="$MODEL_BASE/ckpt_${CKPT}/optimized"
    OUTDIR="$RESULTS/ckpt${CKPT}_parallel"

    echo ""
    echo "=== 补跑 ckpt_${CKPT} GPU 1/2/3 shard ==="
    PIDS=()
    for i in 0 1 2; do
        GPU=${FIX_GPUS[$i]}
        OUT="$OUTDIR/gpu${GPU}.jsonl"
        rm -f "$OUT"  # 清除之前的空文件/部分文件
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$EVAL_SCRIPT" \
            --model  "$MODEL" \
            --input  "$INPUT" \
            --output "$OUT" \
            --offset "${OFFSETS[$i]}" \
            --max_samples 100 \
            --block_size 128 \
            --num_steps  128 \
            > "$OUTDIR/gpu${GPU}_fix.log" 2>&1 &
        echo "  GPU $GPU → offset=${OFFSETS[$i]} PID=$!"
        PIDS+=($!)
    done

    echo "等待 3 个进程..."
    FAILED=0
    for i in 0 1 2; do
        wait ${PIDS[$i]} || { echo "[ERROR] GPU ${FIX_GPUS[$i]} 失败"; FAILED=1; }
        CNT=$(wc -l < "$OUTDIR/gpu${FIX_GPUS[$i]}.jsonl" 2>/dev/null || echo 0)
        echo "  GPU ${FIX_GPUS[$i]} 完成: $CNT 题"
    done

    if [ $FAILED -ne 0 ]; then
        echo "[ERROR] ckpt_${CKPT} 补跑失败"
        continue
    fi

    # 合并所有 6 个 GPU 的结果
    cat "$OUTDIR"/gpu{1,2,3,5,6,7}.jsonl > "$RESULTS/ckpt${CKPT}_merged.jsonl"
    TOTAL=$(wc -l < "$RESULTS/ckpt${CKPT}_merged.jsonl")
    echo "合并: $TOTAL 题 → $RESULTS/ckpt${CKPT}_merged.jsonl"

    echo "--- ckpt_${CKPT} 正确率 ---"
    $PYTHON "$CAL_ACC" --data "$RESULTS/ckpt${CKPT}_merged.jsonl" || true

    echo "等 60s 释放内存..."
    sleep 60
done

echo ""
echo "补跑完成！"
