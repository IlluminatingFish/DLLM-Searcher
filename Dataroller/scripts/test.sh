#!/bin/bash

export SEARCH_API_URL="your_search_api_url"
# export GOOGLE_SEARCH_KEY="xxx"
export MAX_LENGTH=$((1024 * 31 - 500))

MODELS=(
    "doubao-seed-1.8"
)

MODES=("base")

DATASETS=(
    "example"
)

# 用脚本所在路径定位 Dataroller，保证从任意目录运行都能找到 run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

for model in "${MODELS[@]}"; do
    echo -e "\n========================================"
    echo "start: $model"
    echo "========================================"

    for mode in "${MODES[@]}"; do
        echo -e "\n---------- current: $mode ----------"

        for dataset in "${DATASETS[@]}"; do
            echo -e "\n>>>> processing: $dataset"
            output_path=$mode
            
            bash run.sh "$model" "$dataset" "$output_path"

            echo "<<<< dataset $dataset finished(model: $model, mode: $mode)"
        done
    done

    echo -e "\n========================================"
    echo "model $model all finished"
    echo "========================================"
done

