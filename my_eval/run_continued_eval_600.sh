#!/bin/bash
# sft_llada_continued，600题（hotpot300+2wiki300），8GPU并行 eval
# 默认跑 ckpt_3；传入 CKPT=ckpt_4 可切换

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

CKPT=${CKPT:-ckpt_3}
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_continued/$CKPT/optimized"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
OUTDIR="$ROOT/dLLM_trainer/VRPO/output/llada_eval/continued_${CKPT}_600_parallel"
MERGED="$ROOT/dLLM_trainer/VRPO/output/llada_eval/continued_${CKPT}_600.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

mkdir -p "$OUTDIR"

echo "=========================================="
echo "sft_llada_continued $CKPT eval - 600题 - 8 GPU 并行"
echo "模型: $MODEL"
echo "输入: $INPUT"
echo "=========================================="

OFFSETS=(0   75  150  225  300  375  450  525)
COUNTS=(75   75   75   75   75   75   75   75)

PIDS=()
for i in $(seq 0 7); do
    GPU=$i
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    OUT="$OUTDIR/gpu${GPU}.jsonl"

    echo "GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  → $OUT"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model  "$MODEL" \
        --input  "$INPUT" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
        > "$OUTDIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "8 个进程已启动: ${PIDS[*]}"
echo "等待全部完成..."

FAILED=0
for i in $(seq 0 7); do
    wait ${PIDS[$i]}
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[ERROR] GPU $i 退出码 $CODE，查看: $OUTDIR/gpu${i}.log"
        FAILED=1
    else
        CNT=$(wc -l < "$OUTDIR/gpu${i}.jsonl" 2>/dev/null || echo 0)
        echo "GPU $i 完成: $CNT 题"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "有进程失败，跳过合并"
    exit 1
fi

echo ""
echo "合并结果到 $MERGED ..."
cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
TOTAL=$(wc -l < "$MERGED")
echo "合并完成: $TOTAL 题"
