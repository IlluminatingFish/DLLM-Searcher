#!/bin/bash
# =============================================================================
# On-Policy RL Loop [bs128-v2] — dionysos 6-GPU
#
# 与 run_onpolicy_loop_bs128.sh 完全相同，唯一区别：
#   1. π₀ = sft_llada_x0pred_bs128_v2/ckpt_3 (49.83% CEM-1, 新 v2 最优 SFT)
#   2. CUDA_VISIBLE_DEVICES 默认 0,1,3,5,6,7（dionysos 6张空闲 GPU）
#   3. WORLD=6（默认）
#   4. ACCEL_CFG 默认用 accel_zero3_6gpu.yaml
#   5. 输出到 onpolicy_loop_bs128_v2/（不覆盖旧实验）
#
# 用法（在 dionysos tmux/screen 里）：
#   CUDA_VISIBLE_DEVICES=0,1,3,5,6,7 WORLD=6 \
#     screen -S onpolicy_v2 bash run_onpolicy_dionysos_v2.sh
# =============================================================================
set -uo pipefail

# ── 超参 ──────────────────────────────────────────────────────────────────────
START_ROUND=${START_ROUND:-0}   # π₀(v2/ckpt_3, 49.83%) 开始做 rollout
MAX_ROUND=${MAX_ROUND:-5}       # 最后一个做 rollout 的 policy（→ 产生 π₆）
POOL_BATCH=256       # 每轮新题数
N_ROLLS=8            # 每题 rollout 数
WORLD=${WORLD:-6}    # GPU 数（dionysos 6 空闲卡，可通过 WORLD=N 覆盖）
BLOCK_SIZE=128
NUM_STEPS=128

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
POOL_FULL=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_full.jsonl
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
OUTBASE=$AGENTIC/output/onpolicy_loop_bs128_v2
LOOP_LOG=$OUTBASE/loop.log

mkdir -p $OUTBASE

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
# dionysos 默认用 GPU 0,1,3,5,6,7（跳过被占的 GPU2/4）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,5,6,7}
# 解析物理 GPU 列表（用于 heredoc 里子进程的 CUDA_VISIBLE_DEVICES 映射）
IFS=',' read -ra PHYS_GPUS <<< "$CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_v2     # 独立缓存目录，避免与 bs128 冲突
export TORCH_EXTENSIONS_HOME=/tmp/torch_extensions_mz751_v2
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOOP_LOG; }

# ── π_N 的模型路径 ────────────────────────────────────────────────────────────
# N=0 → v2 SFT ckpt_3 (π₀, 49.83% CEM-1)
# N≥1 → onpolicy_loop_bs128_v2/pi{N}/model
get_model_path() {
    local N=$1
    case $N in
        0) echo "$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128_v2/ckpt_3/optimized" ;;
        *) echo "$OUTBASE/pi${N}/model" ;;
    esac
}

# ── 默认 accelerate 配置（6 GPU ZeRO-3）──────────────────────────────────────
ACCEL_CFG=${ACCEL_CFG:-$AGENTIC/configs/accel_zero3_6gpu.yaml}

log "========================================================"
log "On-Policy Loop [bs128-v2, dionysos] 启动：Round ${START_ROUND} → ${MAX_ROUND}"
log "  π₀ = sft_llada_x0pred_bs128_v2/ckpt_3 (49.83% CEM-1)"
log "  block_size=${BLOCK_SIZE}  num_steps=${NUM_STEPS}  WORLD=${WORLD}"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "  ACCEL_CFG=$ACCEL_CFG"
log "  输出: $OUTBASE"
log "========================================================"

for ROUND in $(seq $START_ROUND $MAX_ROUND); do
    NEXT_ROUND=$((ROUND + 1))
    BASE_MODEL=$(get_model_path $ROUND)
    ROUND_DIR=$OUTBASE/pi${NEXT_ROUND}
    POOL_FILE=$ROUND_DIR/pool.jsonl
    ROLLOUT_DIR=$ROUND_DIR/rollouts
    MERGED=$ROUND_DIR/merged.jsonl
    TRAIN_FILE=$ROUND_DIR/train_reward_a.jsonl
    OUT_MODEL=$ROUND_DIR/model
    EVAL_DIR=$ROUND_DIR/eval
    mkdir -p $ROUND_DIR $ROLLOUT_DIR $EVAL_DIR

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    log ""
    log "========================================================"
    log "Round $ROUND:  π_${ROUND} → D_${ROUND} → π_${NEXT_ROUND}"
    log "  Base model : $BASE_MODEL"
    log "  Output     : $OUT_MODEL"
    log "========================================================"

    # 检查基础模型存在
    if [ ! -f "$BASE_MODEL/model.safetensors" ] && \
       [ ! -f "$BASE_MODEL/pytorch_model.bin" ] && \
       [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
        log "[ERROR] 基础模型不存在: $BASE_MODEL"
        exit 1
    fi
    log "  [OK] 基础模型验证通过"

    # ── Step 1: Pool ──────────────────────────────────────────────────────────
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

    # ── Step 2: Rollout（WORLD GPU 并行，block_size=128）─────────────────────
    log "Step 2/4: Rollout  π_${ROUND} × ${POOL_BATCH}题 × ${N_ROLLS}rolls (block_size=${BLOCK_SIZE})..."

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

        for RANK in $(seq 0 $((WORLD-1))); do
            SESSION="v2_r${ROUND}_rank${RANK}"
            INNER_SCRIPT=$OUTBASE/tmp_${SESSION}.sh
            cat > $INNER_SCRIPT << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${PHYS_GPUS[$RANK]}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_v2
${PYTHON} -u ${AGENTIC}/collect_round0_rollouts.py \
    --rank ${RANK} --world ${WORLD} \
    --model "${BASE_MODEL}" \
    --input "${POOL_FILE}" \
    --output "${ROLLOUT_DIR}" \
    --n_rolls ${N_ROLLS} \
    --round_id ${ROUND} \
    --block_size ${BLOCK_SIZE} \
    --num_steps ${NUM_STEPS} \
    2>&1 | tee "${LOG_DIR}/rank${RANK}.log"
INNER
            chmod +x $INNER_SCRIPT
            screen -dmS "$SESSION" bash $INNER_SCRIPT
            log "  rank${RANK} 已启动 (GPU=${PHYS_GPUS[$RANK]})"
            [ $RANK -lt $((WORLD-1)) ] && sleep 20
        done

        log "  等待 rollout 完成（每 2 分钟汇报一次）..."
        while true; do
            DONE=0
            for RANK in $(seq 0 $((WORLD-1))); do
                screen -list 2>/dev/null | grep -q "v2_r${ROUND}_rank${RANK}" || DONE=$((DONE+1))
            done
            log "  rollout $DONE/$WORLD rank 已完成"
            [ $DONE -eq $WORLD ] && break
            sleep 120
        done
        log "  Rollout 完成！"
    fi

    # 合并
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

    # ── Step 3: Reward + 训练（50 steps，每 25 steps 保存一次）────────────────
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

    log "Step 3/4: ELBO-PG 训练  π_${ROUND} → π_${NEXT_ROUND}（50 steps，每 25 steps 保存）..."
    TRAIN_LOG=$ROUND_DIR/train_${TIMESTAMP}.log
    ROUND_CONFIG=$AGENTIC/configs/grpo_pi_bs128v2_${NEXT_ROUND}_${TIMESTAMP}.yaml
    # step-25 和 step-50 的模型目录
    OUT_STEP25=$ROUND_DIR/model_step25
    # OUT_STEP50 就是 OUT_MODEL（step-50 / 最终权重）

    MODEL_READY=false
    MODEL_SIZE=$(stat -c%s "$OUT_MODEL/model.safetensors" 2>/dev/null || echo 0)
    if [ -f "$OUT_MODEL/model.safetensors" ] && [ "$MODEL_SIZE" -gt 1000000000 ]; then
        MODEL_READY=true
    fi

    if ! $MODEL_READY; then
        sed "s|model_name_or_path:.*|model_name_or_path: ${BASE_MODEL}|;
             s|output_dir:.*|output_dir: ${OUT_MODEL}|;
             s|run_name:.*|run_name: onpolicy_bs128v2_pi${NEXT_ROUND}_${TIMESTAMP}|;
             s|dataset_path:.*|dataset_path: ${TRAIN_FILE}|" \
            $AGENTIC/configs/grpo_bs128_v2_50steps.yaml > $ROUND_CONFIG

        cd $VRPO
        accelerate launch \
            --config_file $ACCEL_CFG \
            --num_processes $WORLD \
            my_train/llada_grpo_train.py \
            --config $ROUND_CONFIG \
            2>&1 | tee $TRAIN_LOG

        TRAIN_EXIT=${PIPESTATUS[0]}
        if [ $TRAIN_EXIT -ne 0 ]; then
            log "[ERROR] Round $ROUND 训练失败 (exit=$TRAIN_EXIT)，退出"
            exit 1
        fi

        # ── 提取两个 checkpoint 权重 ──────────────────────────────────────────
        # ZeRO-3 每个 checkpoint 目录里都有完整的 model.safetensors

        # 1. step-25：复制配置文件 + step-25 权重 → model_step25/
        CKPT25=$OUT_MODEL/checkpoint-25
        if [ -f "$CKPT25/model.safetensors" ]; then
            log "  [step-25] 建立 model_step25/..."
            mkdir -p $OUT_STEP25
            # 复制所有非 checkpoint 子目录的文件（config/tokenizer 等）
            find "$OUT_MODEL" -maxdepth 1 -not -type d -exec cp {} "$OUT_STEP25/" \; 2>/dev/null || true
            # 用 step-25 的权重覆盖
            cp "$CKPT25/model.safetensors" "$OUT_STEP25/model.safetensors"
            log "  [step-25] 完整模型: $OUT_STEP25"
        else
            log "  [WARN] checkpoint-25 未找到，跳过 step-25 提取"
        fi

        # 2. step-50：从最新 checkpoint 复制权重 → OUT_MODEL/model.safetensors
        LATEST_CKPT=$(ls -dt $OUT_MODEL/checkpoint-*/model.safetensors 2>/dev/null | tail -1)  # 最大步数的
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_SIZE=$(stat -c%s "$LATEST_CKPT" 2>/dev/null || echo 0)
            if [ "$CKPT_SIZE" -gt 1000000000 ]; then
                log "  [step-50] 从 $(dirname $LATEST_CKPT | xargs basename) 复制完整权重..."
                cp "$LATEST_CKPT" "$OUT_MODEL/model.safetensors"
                log "  [step-50] 完整模型: $OUT_MODEL/model.safetensors"
            fi
        fi

        # 3. 清理所有 checkpoint 子目录（释放 optimizer states 占用的磁盘）
        if ls -d $OUT_MODEL/checkpoint-* 2>/dev/null | grep -q .; then
            log "  [存储] 删除 checkpoint 子目录（optimizer states）..."
            rm -rf $OUT_MODEL/checkpoint-*
            log "  [存储] 已清理"
        fi
        log "  训练完成！step-25 → $OUT_STEP25  step-50 → $OUT_MODEL"
    else
        log "  模型已存在，跳过训练"
        # 如果是重启，检查 model_step25 是否已建
        [ -f "$OUT_STEP25/model.safetensors" ] && log "  step-25 模型已存在" || log "  [WARN] step-25 模型缺失，本次 eval 只跑 step-50"
    fi

    # ── Step 4: eval_600 × 2（step-25 和 step-50 分别评估）─────────────────────
    # eval_one MODEL_DIR STEP_TAG SESSION_PREFIX SCORE_LOG
    eval_one() {
        local MDL=$1 TAG=$2 SFX=$3 SLOG=$4
        local EDIR=$EVAL_DIR/$TAG
        mkdir -p $EDIR

        if [ -f "$SLOG" ]; then
            log "  [$TAG] eval 已存在，跳过"
            return 0
        fi
        if [ ! -f "$MDL/model.safetensors" ] && [ ! -f "$MDL/model.safetensors.index.json" ]; then
            log "  [$TAG] 模型不存在，跳过 eval"
            return 1
        fi

        log "Step 4/4: eval_600 [$TAG]  model=$MDL"
        IFS=',' read -ra PHYS_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        N_EVAL_SHARDS=8
        local SHARD=0
        while [ $SHARD -lt $N_EVAL_SHARDS ]; do
            local BATCH_END=$((SHARD + WORLD - 1))
            [ $BATCH_END -ge $N_EVAL_SHARDS ] && BATCH_END=$((N_EVAL_SHARDS - 1))
            log "  [$TAG] eval batch shard ${SHARD}..${BATCH_END}"
            for IDX in $(seq $SHARD $BATCH_END); do
                local SLOT=$(( (IDX - SHARD) % WORLD ))
                local PHYS_GPU=${PHYS_GPUS[$SLOT]}
                local OFFSET=$((IDX * 75))
                local SESSION="${SFX}_eval${IDX}"
                local INNER_SCRIPT=$OUTBASE/tmp_${SESSION}.sh
                cat > $INNER_SCRIPT << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${PHYS_GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_v2
${PYTHON} -u ${ROOT}/my_eval/run_llada_eval.py \
    --model    "${MDL}" \
    --input    "${EVAL_INPUT}" \
    --output   "${EDIR}/shard${IDX}.jsonl" \
    --offset   ${OFFSET} \
    --max_samples 75 \
    --block_size ${BLOCK_SIZE} \
    --num_steps  ${NUM_STEPS} \
    2>&1 | tee "${EDIR}/shard${IDX}.log"
INNER
                chmod +x $INNER_SCRIPT
                screen -dmS "$SESSION" bash $INNER_SCRIPT
                log "  [$TAG] shard${IDX} 启动 (GPU=${PHYS_GPU}, offset=${OFFSET})"
                sleep 15
            done

            # 等待本批完成
            while true; do
                local DONE=0
                for IDX2 in $(seq $SHARD $BATCH_END); do
                    screen -list 2>/dev/null | grep -q "${SFX}_eval${IDX2}" || DONE=$((DONE+1))
                done
                local BATCH_SIZE=$((BATCH_END - SHARD + 1))
                log "  [$TAG] $DONE/$BATCH_SIZE shard 完成"
                [ $DONE -eq $BATCH_SIZE ] && break
                sleep 60
            done
            SHARD=$((BATCH_END + 1))
        done

        cat $EDIR/shard*.jsonl > $EDIR/merged.jsonl 2>/dev/null
        $PYTHON $ROOT/my_eval/cal_acc.py --data $EDIR/merged.jsonl 2>&1 | tee $SLOG
        log "  [$TAG] eval 完成！"
    }

    # 先评估 step-25，再评估 step-50（串行，避免同时占满 GPU）
    eval_one "$OUT_STEP25" "step25" "v2_r${ROUND}_s25" "$EVAL_DIR/score_step25.log"
    eval_one "$OUT_MODEL"  "step50" "v2_r${ROUND}_s50" "$EVAL_DIR/score_step50.log"

    # ── 打印学习曲线（每 round 列出 step-25 / step-50）──────────────────────
    log ""
    log "========== Learning Curve [bs128-v2] (截至 π_${NEXT_ROUND}) =========="
    printf "%-12s %-12s %-12s\n" "Policy" "step-25" "step-50" | tee -a $LOOP_LOG
    printf "%-12s %-12s %-12s\n" "π₀ SFT" "—" "49.83%" | tee -a $LOOP_LOG
    for N in $(seq 1 $NEXT_ROUND); do
        SC25=$OUTBASE/pi${N}/eval/score_step25.log
        SC50=$OUTBASE/pi${N}/eval/score_step50.log
        C25="—"; C50="—"
        [ -f "$SC25" ] && C25=$(grep -oP "\d+\.\d+(?=%)" $SC25 | head -1)%
        [ -f "$SC50" ] && C50=$(grep -oP "\d+\.\d+(?=%)" $SC50 | head -1)%
        printf "%-12s %-12s %-12s\n" "π_${N}" "$C25" "$C50" | tee -a $LOOP_LOG
    done
    log "=================================================="
    log ""

done

log "========================================================"
log "On-Policy Loop [bs128-v2] 全部完成！Round ${START_ROUND} → ${MAX_ROUND}"
log "  最终模型：π_$((MAX_ROUND+1))"
log "========================================================"
