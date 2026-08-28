#!/bin/bash
# 8-GPU 并行 latent eval
# 每张卡处理 ~13 题，共 100 题，结果合并到 OUTPUT_MERGED
# 用法: bash my_eval/run_latent_eval_parallel.sh
# 从 DLLM-Searcher 根目录运行

set -e
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

MODEL="dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
INPUT="dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
OUTDIR="dLLM_trainer/VRPO/output/llada_eval/latent_parallel"
MERGED="dLLM_trainer/VRPO/output/llada_eval/latent_ckpt6_100.jsonl"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python

mkdir -p "$OUTDIR"

# 100 题分 8 份：每份 13 题（最后一份 9 题）
# GPU 0: 0-12, GPU 1: 13-25, GPU 2: 26-38, GPU 3: 39-51
# GPU 4: 52-64, GPU 5: 65-77, GPU 6: 78-90, GPU 7: 91-99
OFFSETS=(0  13 26 39 52 65 78 91)
COUNTS=(13 13 13 13 13 13 13  9)

echo "=============================="
echo "LLaDA latent eval - 8 GPU 并行"
echo "模型: $MODEL"
echo "=============================="

PIDS=()
for i in $(seq 0 7); do
    GPU=$i
    OFFSET=${OFFSETS[$i]}
    COUNT=${COUNTS[$i]}
    OUT="$OUTDIR/gpu${GPU}.jsonl"

    echo "GPU $GPU: 题 $OFFSET ~ $((OFFSET+COUNT-1))  → $OUT"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
        my_eval/run_llada_latent_eval.py \
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

# 等待所有进程
FAILED=0
for i in $(seq 0 7); do
    wait ${PIDS[$i]}
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[ERROR] GPU $i 进程退出码 $CODE，查看 $OUTDIR/gpu${i}.log"
        FAILED=1
    else
        COUNT=$(wc -l < "$OUTDIR/gpu${i}.jsonl" 2>/dev/null || echo 0)
        echo "GPU $i 完成: $COUNT 题"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "有进程失败，跳过合并"
    exit 1
fi

# 合并 8 个文件（按 GPU 顺序保留题目顺序）
echo ""
echo "合并结果到 $MERGED ..."
cat "$OUTDIR"/gpu{0,1,2,3,4,5,6,7}.jsonl > "$MERGED"
TOTAL=$(wc -l < "$MERGED")
echo "合并完成: $TOTAL 题"

echo ""
echo "打分:"
$PYTHON my_eval/cal_acc.py --data "$MERGED"
