#!/bin/bash
# 需在 dLLM_trainer/VRPO 目录下运行：cd dLLM_trainer/VRPO && bash ../../my_eval/run_test.sh
# 多卡时避免与 DPO 训练端口冲突：PET_MASTER_PORT=29502 bash ../../my_eval/run_test.sh

export PET_MASTER_PORT="${PET_MASTER_PORT:-29502}"

MODEL_PATH="${MODEL_PATH:-/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_7/optimized}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/preact_eval}"

# 支持环境变量：DATASETS="hotpot" 或 DATASETS="hotpot,example"
if [ -n "${DATASETS}" ]; then
    IFS=',' read -ra DATASETS <<< "${DATASETS}"
else
    DATASETS=("example")
fi

NUM_ROLLS=1
BATCH_SIZE=2
GPU_NUM=${GPU_NUM:-8}

MAX_SAMPLES="${MAX_SAMPLES:-30}"           

# ===========================================
echo "========================================"

for dataset in "${DATASETS[@]}"; do

    
    INPUT_JSONL="${BASE_DATA_DIR}/${dataset}.jsonl"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${dataset}"
    
    if [ ! -f "$INPUT_JSONL" ]; then
        echo "⚠️  $INPUT_JSONL not valid, skip it"
        continue
    fi

    INPUT_ACTUAL="$INPUT_JSONL"
    if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
        INPUT_TMP=$(mktemp --suffix=.jsonl)
        head -n "$MAX_SAMPLES" "$INPUT_JSONL" > "$INPUT_TMP"
        INPUT_ACTUAL="$INPUT_TMP"
        echo "Quick test: $dataset limited to $MAX_SAMPLES samples"
    fi

    mkdir -p "$OUTPUT_DIR"

    if [ "$GPU_NUM" -eq 1 ]; then
        python ./my_train/my_test.py \
            --input "$INPUT_ACTUAL" \
            --output_dir "$OUTPUT_DIR" \
            --model_path "$MODEL_PATH" \
            --num_rolls $NUM_ROLLS \
            --batch_size $BATCH_SIZE
    else
        torchrun --nproc_per_node=$GPU_NUM ./my_train/my_test.py \
            --input "$INPUT_ACTUAL" \
            --output_dir "$OUTPUT_DIR" \
            --model_path "$MODEL_PATH" \
            --num_rolls $NUM_ROLLS \
            --batch_size $BATCH_SIZE
    fi

    EXIT_CODE=$?
    [ -n "${INPUT_TMP:-}" ] && [ -f "$INPUT_TMP" ] && rm -f "$INPUT_TMP"
    unset INPUT_TMP 2>/dev/null || true

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "✅  $dataset finished, output: $OUTPUT_DIR"
    else
        echo "❌ $dataset failed!"
    fi
done

echo -e "\n========================================"
echo "output dir: $BASE_OUTPUT_DIR"
echo "========================================"