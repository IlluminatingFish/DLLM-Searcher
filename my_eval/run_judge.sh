#!/bin/bash
# run_judge.sh — LLM Judge 评分脚本
#
# 用法:
#   bash my_eval/run_judge.sh --judge gpt    --data output/llada_eval/sft_baseline_ckpt6_100.jsonl
#   bash my_eval/run_judge.sh --judge doubao --data output/llada_eval/x0pred_ckpt6_100.jsonl
#   bash my_eval/run_judge.sh --judge both   --data output/llada_eval/sft_baseline_ckpt6_100.jsonl
#   bash my_eval/run_judge.sh --judge gpt    --data output/llada_eval/sft_baseline_ckpt6_100.jsonl --no_llm_judge
#
# --judge 选项:
#   gpt     : 只用 GPT-4o 评分（从 config.json judge_* 字段读取）
#   doubao  : 只用 Doubao-seed 评分（从 config.json doubao_* 字段读取）
#   both    : 先跑 GPT-4o，再跑 Doubao，并排输出两份结果

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="$SCRIPT_DIR/score_paper_metrics.py"

# ── 默认参数 ───────────────────────────────────────────────────────────────────
JUDGE="gpt"
DATA=""
NO_LLM_JUDGE=""

# ── 参数解析 ───────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --judge)   JUDGE="$2";   shift 2 ;;
        --data)    DATA="$2";    shift 2 ;;
        --no_llm_judge) NO_LLM_JUDGE="--no_llm_judge"; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "错误：必须指定 --data <pred.jsonl>"
    exit 1
fi

# ── 运行函数 ───────────────────────────────────────────────────────────────────
run_judge() {
    local judge="$1"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Judge: $judge   数据: $DATA"
    echo "════════════════════════════════════════════════════"
    python "$PY" --data "$DATA" --judge "$judge" $NO_LLM_JUDGE
}

# ── 主逻辑 ────────────────────────────────────────────────────────────────────
case "$JUDGE" in
    gpt)    run_judge gpt ;;
    doubao) run_judge doubao ;;
    both)
        run_judge gpt
        echo ""
        run_judge doubao
        ;;
    *)
        echo "错误：--judge 必须是 gpt / doubao / both"
        exit 1
        ;;
esac
