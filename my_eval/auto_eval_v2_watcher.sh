#!/bin/bash
# 自动监控 sft_llada_x0pred_bs128_v2 的每个 ckpt，一旦保存完成就在 dionysos 空闲 GPU 上跑 eval
# 用法: nohup bash auto_eval_v2_watcher.sh > auto_eval_v2_watcher.log 2>&1 &

set -u
export PATH=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin:$PATH

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
MODEL_BASE="$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_v2"
INPUT="$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl"
RESULTS="$ROOT/my_eval/results/bs128_v2"
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin/python
EVAL_SCRIPT="$ROOT/my_eval/run_llada_eval.py"
CAL_ACC="$ROOT/my_eval/cal_acc.py"

# 用 dionysos GPU 1,2,3,5,6,7（跳过 GPU 4 被占用，GPU 0 保留备用）
EVAL_GPUS=(1 2 3 5 6 7)
NUM_EVAL_GPUS=${#EVAL_GPUS[@]}  # 6

# 600题 / 6 GPU = 每 GPU 100题
OFFSETS=(0 100 200 300 400 500)
COUNTS=(100 100 100 100 100 100)

# block_size=128 num_steps=128（正式配置）
BLOCK_SIZE=128
NUM_STEPS=128

TOTAL_CKPTS=7  # ckpt_0 ~ ckpt_6

mkdir -p "$RESULTS"
EVAL_LOG="$RESULTS/eval_watcher.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$EVAL_LOG"
}

# 判断 ckpt 是否保存完成（检查最后一个 safetensors 分片）
is_ckpt_ready() {
    local ckpt=$1
    local sentinel="$MODEL_BASE/ckpt_${ckpt}/optimized/model-00004-of-00004.safetensors"
    [ -f "$sentinel" ]
}

# 判断 ckpt 是否已经 eval 过
is_ckpt_evaled() {
    local ckpt=$1
    local merged="$RESULTS/ckpt${ckpt}_merged.jsonl"
    [ -f "$merged" ] && [ "$(wc -l < "$merged")" -ge 590 ]
}

eval_ckpt() {
    local CKPT=$1
    local MODEL="$MODEL_BASE/ckpt_${CKPT}/optimized"
    local OUTDIR="$RESULTS/ckpt${CKPT}_parallel"
    local MERGED="$RESULTS/ckpt${CKPT}_merged.jsonl"

    log "===== 开始 eval ckpt_${CKPT} ====="
    log "模型: $MODEL"
    log "block_size=$BLOCK_SIZE  num_steps=$NUM_STEPS  600题 × ${NUM_EVAL_GPUS}GPU"

    mkdir -p "$OUTDIR"
    PIDS=()
    for i in $(seq 0 $((NUM_EVAL_GPUS-1))); do
        GPU=${EVAL_GPUS[$i]}
        OUT="$OUTDIR/gpu${GPU}.jsonl"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$EVAL_SCRIPT" \
            --model  "$MODEL" \
            --input  "$INPUT" \
            --output "$OUT" \
            --offset "${OFFSETS[$i]}" \
            --max_samples "${COUNTS[$i]}" \
            --block_size $BLOCK_SIZE \
            --num_steps  $NUM_STEPS \
            > "$OUTDIR/gpu${GPU}.log" 2>&1 &
        log "  GPU $GPU → offset=${OFFSETS[$i]} count=${COUNTS[$i]} PID=$!"
        PIDS+=($!)
    done

    log "等待 ${NUM_EVAL_GPUS} 个进程..."
    FAILED=0
    for i in $(seq 0 $((NUM_EVAL_GPUS-1))); do
        GPU=${EVAL_GPUS[$i]}
        wait ${PIDS[$i]} || { log "[ERROR] GPU ${GPU} 失败"; FAILED=1; }
        CNT=$(wc -l < "$OUTDIR/gpu${GPU}.jsonl" 2>/dev/null || echo 0)
        log "  GPU ${GPU} 完成: $CNT 题"
    done

    if [ $FAILED -ne 0 ]; then
        log "[ERROR] ckpt_${CKPT} eval 有进程失败，跳过合并"
        return 1
    fi

    # 合并（按 GPU 顺序）
    GPU_FILES=()
    for i in $(seq 0 $((NUM_EVAL_GPUS-1))); do
        GPU_FILES+=("$OUTDIR/gpu${EVAL_GPUS[$i]}.jsonl")
    done
    cat "${GPU_FILES[@]}" > "$MERGED"
    TOTAL=$(wc -l < "$MERGED")
    log "合并完成: $TOTAL 题 → $MERGED"

    log "--- 正确率 ---"
    $PYTHON "$CAL_ACC" --data "$MERGED" 2>&1 | tee -a "$EVAL_LOG" || true
    log "===== ckpt_${CKPT} eval 完成 ====="
    log "等待 60s 让 CUDA 内存释放..."
    sleep 60
}

log "启动自动 eval watcher：监控 ckpt_0 ~ ckpt_$((TOTAL_CKPTS-1))"
log "结果目录: $RESULTS"
log "eval GPU: ${EVAL_GPUS[*]}"

# 已处理的 ckpt 集合
PROCESSED=()

while true; do
    for ckpt in $(seq 0 $((TOTAL_CKPTS-1))); do
        # 是否已处理过
        already=0
        for p in "${PROCESSED[@]:-}"; do
            [ "$p" = "$ckpt" ] && already=1 && break
        done
        [ $already -eq 1 ] && continue

        # 检查是否已保存完成
        if ! is_ckpt_ready "$ckpt"; then
            continue
        fi

        # 检查是否已 eval 过（比如重启 watcher 时）
        if is_ckpt_evaled "$ckpt"; then
            log "ckpt_${ckpt} 已有结果，跳过"
            PROCESSED+=("$ckpt")
            continue
        fi

        log "检测到新 ckpt_${ckpt}，准备 eval..."
        eval_ckpt "$ckpt" && PROCESSED+=("$ckpt") || log "[WARN] ckpt_${ckpt} eval 失败，下次重试"
    done

    # 全部处理完则退出
    if [ "${#PROCESSED[@]}" -ge "$TOTAL_CKPTS" ]; then
        log "所有 ${TOTAL_CKPTS} 个 ckpt 已 eval 完毕，watcher 退出"
        break
    fi

    sleep 120  # 每 2 分钟检查一次
done
