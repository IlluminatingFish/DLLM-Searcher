#!/bin/bash
# sft_llada_continued ckpt_3 + sft_llada ckpt_5，600题，同时各用2张卡（共4张，留4张给别人）
# GPU 0-1: sft_llada_continued/ckpt_3
# GPU 2-3: sft_llada/ckpt_5

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

MODEL_A="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_continued/ckpt_3/optimized"
OUTDIR_A="$ROOT/dLLM_trainer/VRPO/output/llada_eval/continued_ckpt3_600_parallel"
MERGED_A="$ROOT/dLLM_trainer/VRPO/output/llada_eval/continued_ckpt3_600.jsonl"

MODEL_B="/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada/ckpt_5/optimized"
OUTDIR_B="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_llada_ckpt5_600_parallel"
MERGED_B="$ROOT/dLLM_trainer/VRPO/output/llada_eval/sft_llada_ckpt5_600.jsonl"

mkdir -p "$OUTDIR_A" "$OUTDIR_B"

echo "=========================================="
echo "双模型并行 eval（各2张卡，共4张）"
echo "A: sft_llada_continued/ckpt_3  → GPU 0,1"
echo "B: sft_llada/ckpt_5            → GPU 2,3"
echo "=========================================="

# 600题 / 2卡 = 300题/卡
OFFSETS_2=(0   300)
COUNTS_2=(300  300)

PIDS_A=()
for i in 0 1; do
    GPU=$i
    OFFSET=${OFFSETS_2[$i]}
    COUNT=${COUNTS_2[$i]}
    OUT="$OUTDIR_A/gpu${GPU}.jsonl"
    echo "[A] GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model  "$MODEL_A" \
        --input  "$INPUT" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
        > "$OUTDIR_A/gpu${GPU}.log" 2>&1 &
    PIDS_A+=($!)
done

PIDS_B=()
for i in 0 1; do
    GPU=$((i+2))
    OFFSET=${OFFSETS_2[$i]}
    COUNT=${COUNTS_2[$i]}
    OUT="$OUTDIR_B/gpu${GPU}.jsonl"
    echo "[B] GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        "$ROOT/my_eval/run_llada_eval.py" \
        --model  "$MODEL_B" \
        --input  "$INPUT" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --max_samples "$COUNT" \
        > "$OUTDIR_B/gpu${GPU}.log" 2>&1 &
    PIDS_B+=($!)
done

echo ""
echo "全部进程已启动，等待完成..."

FAILED=0
for i in 0 1; do
    wait ${PIDS_A[$i]}; CODE=$?
    [ $CODE -ne 0 ] && echo "[A ERROR] GPU $i 退出码 $CODE" && FAILED=1 || echo "[A] GPU $i 完成: $(wc -l < $OUTDIR_A/gpu$i.jsonl) 题"
done
for i in 0 1; do
    GPU=$((i+2))
    wait ${PIDS_B[$i]}; CODE=$?
    [ $CODE -ne 0 ] && echo "[B ERROR] GPU $GPU 退出码 $CODE" && FAILED=1 || echo "[B] GPU $GPU 完成: $(wc -l < $OUTDIR_B/gpu${GPU}.jsonl) 题"
done

if [ $FAILED -ne 0 ]; then echo "有进程失败"; exit 1; fi

echo ""
echo "合并 A → $MERGED_A"
cat "$OUTDIR_A"/gpu{0,1}.jsonl > "$MERGED_A"
echo "合并 B → $MERGED_B"
cat "$OUTDIR_B"/gpu{2,3}.jsonl > "$MERGED_B"
echo "A: $(wc -l < $MERGED_A) 题，B: $(wc -l < $MERGED_B) 题"
echo "完成！"
