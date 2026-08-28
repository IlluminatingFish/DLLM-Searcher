#!/bin/bash
# =============================================================================
# Offline Baseline Training [bs128]（dionysos）
#
# 与 run_offline_train.sh 完全相同，唯一区别：
#   1. 行为策略 = sft_llada_x0pred_bs128/ckpt_4 (block_size=128)
#   2. rollout 生成时加 --block_size 128 --num_steps 128
#   3. 训练 yaml 用 grpo_offline_150steps_bs128.yaml (block_length=128)
#   4. eval 时加 --block_size 128 --num_steps 128
#   5. 输出到 output/offline_loop_bs128/
#
# 前置：无（脚本自行生成 offline rollout）
#
# 用法（dionysos 上）：
#   screen -S offline_bs128 bash run_offline_train_bs128.sh
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

# bs128 固定行为策略（不随训练更新）
SFT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128/ckpt_4/optimized
BLOCK_SIZE=128
NUM_STEPS=128

OFFLINE_ROOT=$AGENTIC/output/offline_loop/offline_rollouts   # 复用原实验的 rollout 目录结构
OUTBASE=$AGENTIC/output/offline_loop_bs128                   # 独立输出目录
ALL_TRAIN=$OUTBASE/all_train_reward_a.jsonl
TRAIN_DIR=$OUTBASE/offline_train
MODEL_DIR=$TRAIN_DIR/model
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
POOL_FULL=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_full.jsonl
LOG=$OUTBASE/offline_train_bs128.log

WORLD=8
POOL_BATCH=1024   # 与原实验一致（256×6轮的量，一次性生成）
N_ROLLS=8

mkdir -p $OUTBASE $TRAIN_DIR

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_bs128
export TORCH_EXTENSIONS_HOME=/tmp/torch_ext_mz751_offline_bs128
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log "========================================================"
log "Offline Baseline [bs128] 启动  $(date)"
log "  行为策略 : $SFT_MODEL"
log "  block_size: $BLOCK_SIZE  num_steps: $NUM_STEPS"
log "  输出目录 : $OUTBASE"
log "========================================================"

# ── Step 0: 生成 offline rollout（bs128，固定行为策略）──────────────────────
ROUND=0
BS_ROLLOUT_DIR=$OUTBASE/offline_rollouts/round${ROUND}
POOL_FILE=$BS_ROLLOUT_DIR/pool.jsonl
ROLLOUT_DIR=$BS_ROLLOUT_DIR/rollouts
BS_MERGED=$BS_ROLLOUT_DIR/merged.jsonl
mkdir -p $BS_ROLLOUT_DIR $ROLLOUT_DIR/logs

log ""
log "Step 0: 生成 offline rollout (block_size=$BLOCK_SIZE)..."

# 生成 pool.jsonl
if [ ! -f "$POOL_FILE" ]; then
    $PYTHON - << PYEOF
import json
pool = [json.loads(l) for l in open('$POOL_FULL') if l.strip()]
chunk = pool[0:$POOL_BATCH]
with open('$POOL_FILE', 'w') as f:
    for r in chunk:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Pool: {len(chunk)} 题')
PYEOF
else
    log "  pool.jsonl 已存在 ($(wc -l < $POOL_FILE) 题)，跳过"
fi

# 检查 rollout 是否已完成
ROLLOUT_DONE=true
for RANK in $(seq 0 $((WORLD-1))); do
    RF="$ROLLOUT_DIR/round${ROUND}/rollouts_rank${RANK}.jsonl"
    CNT=$(wc -l < "$RF" 2>/dev/null || echo 0)
    EXPECTED=$((POOL_BATCH / WORLD * N_ROLLS))
    if [ "$CNT" -lt "$EXPECTED" ]; then ROLLOUT_DONE=false; break; fi
done

if $ROLLOUT_DONE; then
    log "  Rollout 已完成，跳过"
else
    for RANK in $(seq 0 $((WORLD-1))); do
        SESSION="offline_bs128_r${RANK}"
        INNER=$OUTBASE/tmp_${SESSION}.sh
        cat > $INNER << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${RANK}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline_bs128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_offline_bs128
${PYTHON} -u ${AGENTIC}/collect_round0_rollouts.py \
    --rank ${RANK} --world ${WORLD} \
    --model "${SFT_MODEL}" \
    --input "${POOL_FILE}" \
    --output "${ROLLOUT_DIR}" \
    --n_rolls ${N_ROLLS} \
    --round_id ${ROUND} \
    --block_size ${BLOCK_SIZE} \
    --num_steps ${NUM_STEPS} \
    2>&1 | tee "${ROLLOUT_DIR}/logs/rank${RANK}.log"
INNER
        chmod +x $INNER
        screen -dmS "$SESSION" bash $INNER
        log "  rank${RANK} 已启动 (GPU=${RANK})"
        [ $RANK -lt $((WORLD-1)) ] && sleep 20
    done

    log "  等待 rollout 完成（每 2 分钟汇报）..."
    while true; do
        DONE=0
        for RANK in $(seq 0 $((WORLD-1))); do
            screen -list 2>/dev/null | grep -q "offline_bs128_r${RANK}" || DONE=$((DONE+1))
        done
        log "  $DONE/$WORLD rank 完成"
        [ $DONE -eq $WORLD ] && break
        sleep 120
    done
    log "  Rollout 完成！"
fi

# ── Step 1: 合并 rollout + 计算 reward ───────────────────────────────────────
log ""
log "Step 1: 合并 offline rollout 并计算 reward..."

if [ ! -f "$ALL_TRAIN" ] || [ $(wc -l < "$ALL_TRAIN" 2>/dev/null || echo 0) -lt 10 ]; then
    # 合并 rollout
    if [ ! -f "$BS_MERGED" ]; then
        $PYTHON - << PYEOF
import json
from pathlib import Path
recs, bad = [], 0
for f in sorted(Path('${ROLLOUT_DIR}').glob('round*/rollouts_rank*.jsonl')):
    for line in open(f, errors='replace'):
        s = line.strip()
        if not s or '\x00' in s: bad += 1; continue
        try: recs.append(json.loads(s))
        except: bad += 1
    print(f'  {f.name}: OK')
with open('${BS_MERGED}', 'w') as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False)+'\n')
print(f'合并: {len(recs)} 条有效，跳过 {bad} 坏行')
PYEOF
    fi

    TOTAL=$(wc -l < "$BS_MERGED" 2>/dev/null || echo 0)
    log "  合计: $TOTAL 条 rollout"

    $PYTHON $AGENTIC/compute_rewards.py \
        --input "$BS_MERGED" \
        --output "$OUTBASE" \
        2>&1 | tee -a $LOG

    mv $OUTBASE/train_reward_a.jsonl $ALL_TRAIN 2>/dev/null || true
    mv $OUTBASE/train_reward_b.jsonl $OUTBASE/all_train_reward_b.jsonl 2>/dev/null || true

    NTRAIN=$(wc -l < "$ALL_TRAIN" 2>/dev/null || echo 0)
    log "  训练数据: $NTRAIN 条 → $ALL_TRAIN"
else
    NTRAIN=$(wc -l < "$ALL_TRAIN")
    log "  all_train_reward_a.jsonl 已存在 ($NTRAIN 条)，跳过"
fi

# ── Step 2: 训练 125 步（block_length=128）────────────────────────────────────
log ""
log "Step 2: 训练 125 步 (block_length=$BLOCK_SIZE, save every 25 steps)..."
TRAIN_LOG=$TRAIN_DIR/train_bs128.log
CONFIG=$AGENTIC/configs/grpo_offline_150steps_bs128.yaml

if [ ! -d "$MODEL_DIR/checkpoint-125" ]; then
    cd $VRPO
    accelerate launch \
        --config_file $AGENTIC/configs/accel_zero3_8gpu.yaml \
        --num_processes $WORLD \
        my_train/llada_grpo_train.py \
        --config $CONFIG \
        2>&1 | tee $TRAIN_LOG

    TRAIN_EXIT=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT -ne 0 ]; then
        log "[ERROR] 训练失败 (exit=$TRAIN_EXIT)"
        exit 1
    fi
    log "  训练完成！"
else
    log "  checkpoint-125 已存在，跳过训练"
fi

# ── Step 3: eval 各 checkpoint（block_size=128）───────────────────────────────
log ""
log "Step 3: eval checkpoints (block_size=$BLOCK_SIZE)..."
CHECKPOINTS="25 50 75 100 125"

for STEP in $CHECKPOINTS; do
    CKPT_DIR=$MODEL_DIR/checkpoint-${STEP}
    EVAL_DIR=$TRAIN_DIR/eval_step${STEP}
    SCORE_LOG=$EVAL_DIR/score.log
    mkdir -p $EVAL_DIR

    if [ -f "$SCORE_LOG" ]; then
        log "  step-${STEP} eval 已完成，跳过"
        continue
    fi

    # ZeRO-3 fix
    if [ ! -f "$CKPT_DIR/model.safetensors" ] && [ -f "$CKPT_DIR/pytorch_model.bin" ]; then
        log "  [step-${STEP}] ZeRO-3 fix..."
        $PYTHON -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('$CKPT_DIR', trust_remote_code=True)
m.save_pretrained('$CKPT_DIR')
print('done')
" 2>/dev/null || log "  [WARNING] 转换失败，跳过 step-${STEP}"
    fi

    if [ ! -f "$CKPT_DIR/model.safetensors" ]; then
        log "  [WARN] step-${STEP} 无 model.safetensors，跳过"
        continue
    fi

    log "  eval step-${STEP} 启动（8 GPU × 75 题）..."
    for GPU in $(seq 0 $((WORLD-1))); do
        OFFSET=$((GPU * 75))
        TC=/tmp/triton_offline_bs128_s${STEP}_g${GPU}
        mkdir -p $TC
        nohup bash -c "
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=${TC}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -u ${ROOT}/my_eval/run_llada_eval.py \
    --model  '${CKPT_DIR}' \
    --input  '${EVAL_INPUT}' \
    --output '${EVAL_DIR}/shard${GPU}.jsonl' \
    --offset ${OFFSET} \
    --max_samples 75 \
    --block_size ${BLOCK_SIZE} \
    --num_steps  ${NUM_STEPS} \
    2>&1 | tee '${EVAL_DIR}/shard${GPU}.log'
" > /dev/null 2>&1 &
        sleep 5
    done

    log "  等待 step-${STEP} eval 完成..."
    until [ $(ls $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l) -ge $WORLD ] && \
          [ $(cat $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l) -ge 590 ]; do
        LINES=$(cat $EVAL_DIR/shard*.jsonl 2>/dev/null | wc -l)
        log "    step-${STEP}: ${LINES}/600"
        sleep 120
    done

    cat $EVAL_DIR/shard*.jsonl > $EVAL_DIR/all.jsonl
    $PYTHON $ROOT/my_eval/cal_acc.py \
        --data $EVAL_DIR/all.jsonl \
        2>&1 | tee $SCORE_LOG
    log "  step-${STEP} 完成！$(grep 'CEM-1' $SCORE_LOG 2>/dev/null | head -1)"

    # 磁盘策略：eval 完后删除中间 checkpoint 权重（只保留 step-125）
    if [ "$STEP" -ne 125 ]; then
        log "  [存储] 删除 step-${STEP} model.safetensors..."
        rm -f $CKPT_DIR/model.safetensors
    fi
done

# ── Step 4: 学习曲线汇总 ──────────────────────────────────────────────────────
log ""
log "========================================================"
log "Offline Baseline [bs128] 学习曲线"
log "========================================================"
printf "%-12s %-10s\n" "Checkpoint" "CEM-1" | tee -a $LOG
printf "%-12s %-10s\n" "bs128 SFT ckpt_4" "49.2%" | tee -a $LOG
for STEP in $CHECKPOINTS; do
    SC=$TRAIN_DIR/eval_step${STEP}/score.log
    if [ -f "$SC" ]; then
        CEM=$(grep -oP '\d+\.\d+(?=%)' "$SC" | head -1)
        printf "%-12s %-10s\n" "step-${STEP}" "${CEM:-?}%" | tee -a $LOG
    fi
done
log "========================================================"
log "全部完成！最终模型: $MODEL_DIR/checkpoint-125"
