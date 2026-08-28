#!/bin/bash
# =============================================================================
# Offline Baseline: Pre-generate ALL rollout data from fixed SFT π₀
#
# 用固定 SFT π₀ (sft_x0pred/ckpt_6) 一次性生成全部 6 轮 offline 数据
# 输出：output/offline_loop/offline_Dk/  (k=0..5)
#
# 在 hermes 8 GPU 上运行（每轮顺序执行，共 6 轮）
# =============================================================================
set -uo pipefail

MAX_ROUND=0
POOL_BATCH=1024
N_ROLLS=8
WORLD=8

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
POOL_FULL=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_full.jsonl
SFT_MODEL=$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized
OUTBASE=$AGENTIC/output/offline_loop
OFFLINE_ROOT=$OUTBASE/offline_rollouts
LOG=$OUTBASE/offline_rollouts.log

mkdir -p $OFFLINE_ROOT

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline
export TORCH_EXTENSIONS_HOME=/tmp/torch_ext_mz751_offline
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log "========================================================"
log "Offline Rollout 生成：固定 SFT π₀，共 $((MAX_ROUND+1)) 轮"
log "  SFT model : $SFT_MODEL"
log "  输出根目录 : $OFFLINE_ROOT"
log "========================================================"

for ROUND in $(seq 0 $MAX_ROUND); do
    ROUND_DIR=$OFFLINE_ROOT/round${ROUND}
    POOL_FILE=$ROUND_DIR/pool.jsonl
    ROLLOUT_DIR=$ROUND_DIR/rollouts
    MERGED=$ROUND_DIR/merged.jsonl
    TRAIN_FILE=$ROUND_DIR/train_reward_a.jsonl
    mkdir -p $ROUND_DIR $ROLLOUT_DIR

    POOL_START=$((ROUND * POOL_BATCH))
    POOL_END=$((POOL_START + POOL_BATCH))

    log ""
    log "=== Round $ROUND: Q[${POOL_START}:${POOL_END}] via SFT π₀ ==="

    # ── Pool ──────────────────────────────────────────────────────────────────
    if [ ! -f "$POOL_FILE" ]; then
        $PYTHON - << PYEOF
import json
pool = [json.loads(l) for l in open('$POOL_FULL') if l.strip()]
chunk = pool[$POOL_START:$POOL_END]
with open('$POOL_FILE', 'w') as f:
    for r in chunk:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Pool R${ROUND}: {len(chunk)} 题  [{$POOL_START}, {$POOL_END})')
PYEOF
    else
        log "  pool.jsonl 已存在，跳过"
    fi

    # ── Rollout（SFT π₀，固定不变）──────────────────────────────────────────
    ROLLOUT_DONE=true
    for RANK in $(seq 0 $((WORLD-1))); do
        RF="$ROLLOUT_DIR/round${ROUND}/rollouts_rank${RANK}.jsonl"
        if [ ! -f "$RF" ] || [ $(wc -l < "$RF" 2>/dev/null || echo 0) -lt $((POOL_BATCH / WORLD * N_ROLLS)) ]; then
            ROLLOUT_DONE=false; break
        fi
    done

    if $ROLLOUT_DONE; then
        log "  Rollout R${ROUND} 已完成，跳过"
    else
        LOG_DIR=$ROLLOUT_DIR/logs
        mkdir -p $LOG_DIR

        for RANK in $(seq 0 $((WORLD-1))); do
            SESSION="offline_roll_r${ROUND}_rank${RANK}"
            INNER=$OUTBASE/tmp_${SESSION}.sh
            cat > $INNER << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${RANK}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_offline
${PYTHON} -u ${AGENTIC}/collect_round0_rollouts.py \
    --rank ${RANK} --world ${WORLD} \
    --model "${SFT_MODEL}" \
    --input "${POOL_FILE}" \
    --output "${ROLLOUT_DIR}" \
    --n_rolls ${N_ROLLS} \
    --round_id ${ROUND} \
    2>&1 | tee "${LOG_DIR}/rank${RANK}.log"
INNER
            chmod +x $INNER
            screen -dmS "$SESSION" bash $INNER
            log "  rank${RANK} 已启动 (GPU=${RANK})"
            [ $RANK -lt $((WORLD-1)) ] && sleep 20
        done

        log "  等待 Rollout R${ROUND} 完成..."
        while true; do
            DONE=0
            for RANK in $(seq 0 $((WORLD-1))); do
                screen -list 2>/dev/null | grep -q "offline_roll_r${ROUND}_rank${RANK}" || DONE=$((DONE+1))
            done
            log "  $DONE/$WORLD rank 完成"
            [ $DONE -eq $WORLD ] && break
            sleep 120
        done
        log "  Rollout R${ROUND} 完成！"
    fi

    # ── 合并 ──────────────────────────────────────────────────────────────────
    if [ ! -f "$MERGED" ]; then
        $PYTHON - << PYEOF
import json
from pathlib import Path
# 鲁棒合并：跳过 null bytes 行和 JSON 解析失败的行（防止 NFS/ThreadPool 写入污染）
recs = []
bad_total = 0
for f in sorted(Path('${ROLLOUT_DIR}').glob('round*/rollouts_rank*.jsonl')):
    file_ok = 0
    file_bad = 0
    try:
        for line in open(f, encoding='utf-8', errors='replace'):
            s = line.strip()
            if not s or '\x00' in s:
                file_bad += 1
                continue
            try:
                recs.append(json.loads(line))
                file_ok += 1
            except json.JSONDecodeError:
                file_bad += 1
        print(f'  {f.name}: {file_ok} OK, {file_bad} skip')
        bad_total += file_bad
    except Exception as e:
        print(f'  {f.name}: 读取失败 {e}')
with open('${MERGED}', 'w') as fout:
    for r in recs:
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'合并 R${ROUND}: {len(recs)} 条有效（跳过 {bad_total} 坏行）')
PYEOF
    fi

    # ── Reward 计算 ───────────────────────────────────────────────────────────
    if [ ! -f "$TRAIN_FILE" ]; then
        log "  计算 Reward..."
        $PYTHON $AGENTIC/compute_rewards.py \
            --input "$MERGED" \
            --output "$ROUND_DIR"
    fi

    NTRAIN=$(wc -l < "$TRAIN_FILE" 2>/dev/null || echo 0)
    log "  R${ROUND} 训练数据: $NTRAIN 条 → $TRAIN_FILE"
done

log ""
log "========================================================"
log "Offline Rollout 全部生成完毕！共 $((MAX_ROUND+1)) 轮"
log "========================================================"
