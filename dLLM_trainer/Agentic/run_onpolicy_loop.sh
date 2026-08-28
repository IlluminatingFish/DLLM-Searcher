#!/bin/bash
# =============================================================================
# On-Policy RL Loop (hermes 8-GPU)
#
# 命名约定：Round N = 用 π_N 生成 D_N → ELBO-PG → π_{N+1}
#
# 已完成：
#   Round 0: π₀(SFT) → D₀ → π₁  (phase4_elbopg/reward_a/model)
#   Round 1: π₁      → D₁ → π₂  (phase4_elbopg_round2/reward_a/model)
#
# 本脚本从 Round 2 (π₂ 做 rollout) 开始，到 Round MAX_ROUND 为止
# 最终产出：π_{MAX_ROUND+1}
#
# 用法（在 hermes screen 里）：
#   screen -S onpolicy_loop bash run_onpolicy_loop.sh
# =============================================================================
set -uo pipefail

# ── 超参 ──────────────────────────────────────────────────────────────────────
START_ROUND=0        # π₀(SFT) 开始做 rollout  [LR=1e-6 重跑]
MAX_ROUND=5          # 最后一个做 rollout 的 policy（→ 产生 π₆）
POOL_BATCH=256       # 每轮新题数
N_ROLLS=8            # 每题 rollout 数
WORLD=8              # GPU 数（hermes A100×8）

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
POOL_FULL=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_full.jsonl
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
OUTBASE=$AGENTIC/output/onpolicy_loop
LOOP_LOG=$OUTBASE/loop.log

mkdir -p $OUTBASE

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export TORCH_EXTENSIONS_HOME=/tmp/torch_extensions_mz751
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOOP_LOG; }

# ── π_N 的模型路径 ────────────────────────────────────────────────────────────
# N=0 → SFT ckpt_6  (π₀)
# N=1 → phase4_elbopg/reward_a/model  (π₁)
# N=2 → phase4_elbopg_round2/reward_a/model  (π₂)
# N≥3 → onpolicy_loop/pi{N}/model  (π₃, π₄, ...)
get_model_path() {
    local N=$1
    case $N in
        0) echo "$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized" ;;
        *) echo "$OUTBASE/pi${N}/model" ;;
    esac
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
log "========================================================"
log "On-Policy Loop 启动：Round ${START_ROUND} → ${MAX_ROUND}  (目标: π₆)"
log "  命名: Round N = π_N → D_N → π_{N+1}"
log "========================================================"

for ROUND in $(seq $START_ROUND $MAX_ROUND); do
    NEXT_ROUND=$((ROUND + 1))
    BASE_MODEL=$(get_model_path $ROUND)
    ROUND_DIR=$OUTBASE/pi${NEXT_ROUND}     # 产出 π_{ROUND+1} 的工作目录
    POOL_FILE=$ROUND_DIR/pool.jsonl
    ROLLOUT_DIR=$ROUND_DIR/rollouts
    MERGED=$ROUND_DIR/merged.jsonl
    TRAIN_FILE=$ROUND_DIR/train_reward_a.jsonl
    OUT_MODEL=$ROUND_DIR/model             # π_{ROUND+1}
    EVAL_DIR=$ROUND_DIR/eval
    mkdir -p $ROUND_DIR $ROLLOUT_DIR $EVAL_DIR

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    log ""
    log "========================================================"
    log "Round $ROUND:  π_${ROUND} → D_${ROUND} → π_${NEXT_ROUND}"
    log "  Base model : $BASE_MODEL"
    log "  Output     : $OUT_MODEL"
    log "========================================================"

    # ── 检查基础模型存在（支持单文件和分片 safetensors）────────────────────────
    if [ ! -f "$BASE_MODEL/model.safetensors" ] && \
       [ ! -f "$BASE_MODEL/pytorch_model.bin" ] && \
       [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
        log "[ERROR] 基础模型不存在: $BASE_MODEL"
        exit 1
    fi
    log "  [OK] 基础模型验证通过"

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: 生成本轮 pool  [ROUND*256, (ROUND+1)*256)
    # ────────────────────────────────────────────────────────────────────────
    POOL_START=$((ROUND * POOL_BATCH))
    POOL_END=$((POOL_START + POOL_BATCH))
    log "Step 1/4: Pool  [${POOL_START}, ${POOL_END})..."

    if [ ! -f "$POOL_FILE" ]; then
        $PYTHON - << PYEOF
import json
pool = [json.loads(l) for l in open('$POOL_FULL') if l.strip()]
chunk = pool[$POOL_START:$POOL_END]
with open('$POOL_FILE', 'w') as f:
    for r in chunk:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Pool: {len(chunk)} 题  [{$POOL_START}, {$POOL_END})')
PYEOF
    else
        NPOOL=$(wc -l < "$POOL_FILE")
        log "  Pool 已存在 ($NPOOL 题)，跳过"
    fi

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: Rollout（8 GPU 并行，collect_round0_rollouts.py 输出到
    #         $ROLLOUT_DIR/round${ROUND}/rollouts_rank*.jsonl）
    # ────────────────────────────────────────────────────────────────────────
    log "Step 2/4: Rollout  π_${ROUND} × ${POOL_BATCH}题 × ${N_ROLLS}rolls..."

    ROLLOUT_DONE=true
    for RANK in $(seq 0 $((WORLD-1))); do
        RF="$ROLLOUT_DIR/round${ROUND}/rollouts_rank${RANK}.jsonl"
        if [ ! -f "$RF" ] || [ $(wc -l < "$RF" 2>/dev/null || echo 0) -lt $((POOL_BATCH / WORLD * N_ROLLS)) ]; then
            ROLLOUT_DONE=false
            break
        fi
    done

    if $ROLLOUT_DONE; then
        log "  Rollout 已完成，跳过"
    else
        LOG_DIR=$ROLLOUT_DIR/logs
        mkdir -p $LOG_DIR

        # 错开 20s 启动各 rank，避免 NFS 并发加载模型
        for RANK in $(seq 0 $((WORLD-1))); do
            SESSION="loop_r${ROUND}_rank${RANK}"
            INNER_SCRIPT=$OUTBASE/tmp_${SESSION}.sh
            cat > $INNER_SCRIPT << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${RANK}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751
${PYTHON} -u ${AGENTIC}/collect_round0_rollouts.py \
    --rank ${RANK} --world ${WORLD} \
    --model "${BASE_MODEL}" \
    --input "${POOL_FILE}" \
    --output "${ROLLOUT_DIR}" \
    --n_rolls ${N_ROLLS} \
    --round_id ${ROUND} \
    2>&1 | tee "${LOG_DIR}/rank${RANK}.log"
INNER
            chmod +x $INNER_SCRIPT
            screen -dmS "$SESSION" bash $INNER_SCRIPT
            log "  rank${RANK} 已启动 (GPU=${RANK})"
            [ $RANK -lt $((WORLD-1)) ] && sleep 20
        done

        # 等待所有 rank 完成（screen session 消失即完成）
        log "  等待 rollout 完成（每 2 分钟汇报一次）..."
        while true; do
            DONE=0
            for RANK in $(seq 0 $((WORLD-1))); do
                screen -list 2>/dev/null | grep -q "loop_r${ROUND}_rank${RANK}" || DONE=$((DONE+1))
            done
            log "  rollout $DONE/$WORLD rank 已完成"
            [ $DONE -eq $WORLD ] && break
            sleep 120
        done
        log "  Rollout 完成！"
    fi

    # 合并所有 rank 的输出
    if [ ! -f "$MERGED" ]; then
        log "  合并 rollout 文件..."
        $PYTHON - << PYEOF
import json
from pathlib import Path
recs = []
for f in sorted(Path('${ROLLOUT_DIR}/round${ROUND}').glob('rollouts_rank*.jsonl')):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    print(f'  {f.name}: {len(rows)}')
    recs.extend(rows)
with open('${MERGED}', 'w') as fout:
    for r in recs:
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'合并完成: {len(recs)} 条')
PYEOF
        NMERGED=$(wc -l < "$MERGED")
        log "  merged.jsonl: $NMERGED 条"
    else
        NMERGED=$(wc -l < "$MERGED")
        log "  merged.jsonl 已存在 ($NMERGED 条)"
    fi

    # ────────────────────────────────────────────────────────────────────────
    # Step 3: Reward 计算 + ELBO-PG 训练
    # ────────────────────────────────────────────────────────────────────────
    log "Step 3/4: Reward 计算..."
    if [ ! -f "$TRAIN_FILE" ]; then
        $PYTHON $AGENTIC/compute_rewards.py \
            --input "$MERGED" \
            --output "$ROUND_DIR"
    else
        NTRAIN=$(wc -l < "$TRAIN_FILE")
        log "  train_reward_a.jsonl 已存在 ($NTRAIN 条)"
    fi
    NTRAIN=$(wc -l < "$TRAIN_FILE")
    log "  训练数据: $NTRAIN 条"

    log "Step 3/4: ELBO-PG 训练  π_${ROUND} → π_${NEXT_ROUND}..."
    TRAIN_LOG=$ROUND_DIR/train_${TIMESTAMP}.log
    ROUND_CONFIG=$AGENTIC/configs/grpo_pi${NEXT_ROUND}_${TIMESTAMP}.yaml

    # 检查模型是否已存在（同时检查根目录和 checkpoint 子目录）
    MODEL_READY=false
    MODEL_SIZE=$(stat -c%s "$OUT_MODEL/model.safetensors" 2>/dev/null || echo 0)
    if [ -f "$OUT_MODEL/model.safetensors" ] && [ "$MODEL_SIZE" -gt 1000000000 ]; then
        MODEL_READY=true
    fi

    if ! $MODEL_READY; then
        # config 写到 NFS（hermes 可以读到）
        sed "s|model_name_or_path:.*|model_name_or_path: ${BASE_MODEL}|;
             s|output_dir:.*|output_dir: ${OUT_MODEL}|;
             s|run_name:.*|run_name: onpolicy_pi${NEXT_ROUND}_${TIMESTAMP}|;
             s|dataset_path:.*|dataset_path: ${TRAIN_FILE}|" \
            $AGENTIC/configs/grpo_phase4_reward_a.yaml > $ROUND_CONFIG

        cd $VRPO
        accelerate launch \
            --config_file $AGENTIC/configs/accel_zero3_8gpu.yaml \
            --num_processes 8 \
            my_train/llada_grpo_train.py \
            --config $ROUND_CONFIG \
            2>&1 | tee $TRAIN_LOG

        TRAIN_EXIT=${PIPESTATUS[0]}
        if [ $TRAIN_EXIT -ne 0 ]; then
            log "[ERROR] Round $ROUND 训练失败 (exit=$TRAIN_EXIT)，退出"
            exit 1
        fi

        # ZeRO-3 有时把完整权重保存在 checkpoint-*/model.safetensors
        # 而 output_dir 下的 model.safetensors 只是 stub（550K）
        # 自动修复：从最新 checkpoint 拷贝完整权重
        LATEST_CKPT=$(ls -dt $OUT_MODEL/checkpoint-*/model.safetensors 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_SIZE=$(stat -c%s "$LATEST_CKPT" 2>/dev/null || echo 0)
            if [ "$CKPT_SIZE" -gt 1000000000 ]; then
                log "  [ZeRO-3 fix] 从 checkpoint 复制完整权重 (${CKPT_SIZE} bytes)..."
                cp "$LATEST_CKPT" "$OUT_MODEL/model.safetensors"
                log "  [OK] 完整模型权重已复制到 $OUT_MODEL/model.safetensors"
            fi
        fi
        # ── 删除 checkpoint 子目录以释放磁盘（model.safetensors 已复制）──
        CKPT_DIRS=$(ls -dt $OUT_MODEL/checkpoint-* 2>/dev/null)
        if [ -n "$CKPT_DIRS" ]; then
            log "  [存储] 删除 checkpoint 子目录以释放磁盘..."
            rm -rf $OUT_MODEL/checkpoint-*
            log "  [存储] 已清理"
        fi
        log "  训练完成！π_${NEXT_ROUND} → $OUT_MODEL"
    else
        log "  模型已存在 ($(du -sh $OUT_MODEL/model.safetensors | cut -f1))，跳过训练"
    fi

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: eval_600  π_{NEXT_ROUND}（8 GPU，每卡 75 题，30s 错开加载）
    # ────────────────────────────────────────────────────────────────────────
    log "Step 4/4: eval_600  π_${NEXT_ROUND}..."
    SCORE_LOG=$EVAL_DIR/score.log

    if [ ! -f "$SCORE_LOG" ]; then
        for GPU in $(seq 0 $((WORLD-1))); do
            OFFSET=$((GPU * 75))
            SESSION="loop_r${ROUND}_eval${GPU}"
            INNER_SCRIPT=$OUTBASE/tmp_${SESSION}.sh
            cat > $INNER_SCRIPT << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751
${PYTHON} -u ${ROOT}/my_eval/run_llada_eval.py \
    --model    "${OUT_MODEL}" \
    --input    "${EVAL_INPUT}" \
    --output   "${EVAL_DIR}/shard${GPU}.jsonl" \
    --offset   ${OFFSET} \
    --max_samples 75 \
    2>&1 | tee "${EVAL_DIR}/shard${GPU}.log"
INNER
            chmod +x $INNER_SCRIPT
            screen -dmS "$SESSION" bash $INNER_SCRIPT
            log "  eval GPU${GPU} 已启动 (offset=${OFFSET})"
            [ $GPU -lt $((WORLD-1)) ] && sleep 30
        done

        log "  等待 eval 完成（每 1 分钟汇报）..."
        while true; do
            DONE=0
            for GPU in $(seq 0 $((WORLD-1))); do
                screen -list 2>/dev/null | grep -q "loop_r${ROUND}_eval${GPU}" || DONE=$((DONE+1))
            done
            log "  eval $DONE/$WORLD shard 完成"
            [ $DONE -eq $WORLD ] && break
            sleep 60
        done

        cat $EVAL_DIR/shard*.jsonl > $EVAL_DIR/merged.jsonl 2>/dev/null
        $PYTHON $ROOT/my_eval/cal_acc.py \
            --data $EVAL_DIR/merged.jsonl \
            2>&1 | tee $SCORE_LOG
        log "  eval 完成！"
    else
        log "  eval 已存在，跳过"
    fi

    # ── 打印最新学习曲线 ──────────────────────────────────────────────────────
    log ""
    log "========== Learning Curve (截至 π_${NEXT_ROUND}) =========="
    # 固定已知数据
    printf "%-6s %-10s %-10s\n" "Policy" "CEM-1" "LLM-Judge" | tee -a $LOOP_LOG
    printf "%-6s %-10s %-10s\n" "π₀ SFT" "43.33%"  "44.33%" | tee -a $LOOP_LOG
    for N in $(seq 1 $NEXT_ROUND); do
        SC=$OUTBASE/pi${N}/eval/score.log
        if [ -f "$SC" ]; then
            CEM=$(grep -oP "(?<=CEM-1[: ]+)\d+\.\d+" $SC 2>/dev/null || grep -oP "\d+\.\d+(?=%)" $SC | head -1)
            JDG=$(grep -oP "(?<=LLM.Judge[: ]+)\d+\.\d+" $SC 2>/dev/null || grep -oP "\d+\.\d+(?=%)" $SC | sed -n '2p')
            printf "%-6s %-10s %-10s\n" "π_${N}" "${CEM}%" "${JDG}%" | tee -a $LOOP_LOG
        fi
    done
    log "=================================================="
    log ""

done

log "========================================================"
log "On-Policy Loop 全部完成！Round ${START_ROUND} → ${MAX_ROUND}"
log "  最终模型：π_$((MAX_ROUND+1)) (π₆)"
log "========================================================"
