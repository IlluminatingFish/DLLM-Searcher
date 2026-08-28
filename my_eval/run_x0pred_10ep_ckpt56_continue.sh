#!/bin/bash
# 续跑：补 ckpt_5 的 GPU3（题 225-299），然后顺序跑 ckpt_6
# 8 GPU 并行，每卡 75 题

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
BASE_MODEL_DIR="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_10ep"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
OUTROOT="$ROOT/dLLM_trainer/VRPO/output/llada_eval"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

GPUS=(0 1 2 3 4 5 6 7)
OFFSETS=(0 75 150 225 300 375 450 525)
COUNTS=(75 75 75 75 75 75 75 75)

# ── ckpt_5: 只补 GPU3（题 225-299）──────────────────────────────
echo "=============================="
echo "  补跑 ckpt_5 GPU3（题 225-299）"
echo "=============================="
CKPT=ckpt_5
MODEL="$BASE_MODEL_DIR/$CKPT/optimized"
OUTDIR="$OUTROOT/x0pred_10ep_${CKPT}_600_parallel"
MERGED="$OUTROOT/x0pred_10ep_${CKPT}_600.jsonl"

CUDA_VISIBLE_DEVICES=3 $PYTHON \
    "$ROOT/my_eval/run_llada_eval.py" \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTDIR/gpu3.jsonl" \
    --offset 225 \
    --max_samples 75 \
    > "$OUTDIR/gpu3.log" 2>&1
echo "ckpt_5 GPU3 完成"

cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
echo "ckpt_5 合并完成: $(wc -l < "$MERGED") 题 → $MERGED"
echo ""

# ── ckpt_6: 8 GPU 全跑 ──────────────────────────────────────────
CKPT=ckpt_6
MODEL="$BASE_MODEL_DIR/$CKPT/optimized"
OUTDIR="$OUTROOT/x0pred_10ep_${CKPT}_600_parallel"
MERGED="$OUTROOT/x0pred_10ep_${CKPT}_600.jsonl"

mkdir -p "$OUTDIR"
echo "=============================="
echo "  评测 ckpt_6"
echo "=============================="

PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    GPU=${GPUS[$i]}
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model "$MODEL" \
        --input "$INPUT" \
        --output "$OUTDIR/gpu${GPU}.jsonl" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
        > "$OUTDIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  PID=$!"
done

echo "等待 ckpt_6 完成..."
FAILED=0
for i in 0 1 2 3 4 5 6 7; do
    wait ${PIDS[$i]}; CODE=$?
    [ $CODE -ne 0 ] && echo "[ERROR] GPU ${GPUS[$i]} 退出码 $CODE" && FAILED=1
done
[ $FAILED -ne 0 ] && echo "ckpt_6 有进程失败" && exit 1

cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
echo "ckpt_6 完成: $(wc -l < "$MERGED") 题 → $MERGED"

echo "=============================="
echo "全部完成"
echo "=============================="
