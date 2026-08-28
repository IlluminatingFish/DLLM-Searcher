#!/bin/bash
# 独立启动 5 对 ΔELBO analysis（GPU 0-4，与 eval 共享）
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
ANALYSIS_DIR=$AGENTIC/output/delta_elbo_analysis

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p $ANALYSIS_DIR

log "hostname: $(hostname)"
log "启动 5 对 ΔELBO analysis..."

declare -a LABELS=(
    "Round0_pi0_to_pi1"
    "Round1_pi1_to_pi2"
    "Round2_pi2_to_pi3"
    "Round3_pi3_to_pi4"
    "Round4_pi4_to_pi5"
)
declare -a BEFORES=(
    "$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
    "$AGENTIC/output/phase4_elbopg/reward_a/model"
    "$AGENTIC/output/phase4_elbopg_round2/reward_a/model"
    "$AGENTIC/output/onpolicy_loop/pi3/model"
    "$AGENTIC/output/onpolicy_loop/pi4/model"
)
declare -a AFTERS=(
    "$AGENTIC/output/phase4_elbopg/reward_a/model"
    "$AGENTIC/output/phase4_elbopg_round2/reward_a/model"
    "$AGENTIC/output/onpolicy_loop/pi3/model"
    "$AGENTIC/output/onpolicy_loop/pi4/model"
    "$AGENTIC/output/onpolicy_loop/pi5/model"
)
declare -a DATAS=(
    "$AGENTIC/output/round0_rollouts/train_reward_a.jsonl"
    "$AGENTIC/output/round1_rollouts/train_reward_a.jsonl"
    "$AGENTIC/output/onpolicy_loop/pi3/train_reward_a.jsonl"
    "$AGENTIC/output/onpolicy_loop/pi4/train_reward_a.jsonl"
    "$AGENTIC/output/onpolicy_loop/pi5/train_reward_a.jsonl"
)

for I in $(seq 0 4); do
    GPU=$I
    LABEL="${LABELS[$I]}"
    SESSION="delta_elbo_${I}"
    INNER=/tmp/hermes_analysis_${I}.sh
    cat > $INNER << INNER_EOF
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_ana_${I}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_ana_${I}
cd ${VRPO}
${PYTHON} -u ${AGENTIC}/analyze_delta_elbo.py \
    --before      "${BEFORES[$I]}" \
    --after       "${AFTERS[$I]}" \
    --data        "${DATAS[$I]}" \
    --output      "${ANALYSIS_DIR}/${LABEL}.jsonl" \
    --label       "${LABEL}" \
    --gpu         0 \
    --n_mc        4 \
    --max_samples 256 \
    2>&1 | tee "${ANALYSIS_DIR}/${LABEL}.log"
INNER_EOF
    chmod +x $INNER
    screen -dmS "$SESSION" bash $INNER
    log "  [$I] $LABEL → GPU ${GPU} 已启动"
    sleep 5   # 错开一点，避免同时拉 transformers 缓存
done

log ""
log "全部已启动:"
screen -list 2>/dev/null | grep delta_elbo

log ""
log "等待 analysis 完成..."
while true; do
    DONE=0
    for I in $(seq 0 4); do
        screen -list 2>/dev/null | grep -q "delta_elbo_${I}" || DONE=$((DONE+1))
    done
    log "  analysis: $DONE/5 对完成"
    [ $DONE -eq 5 ] && break
    sleep 120
done

log ""
log "===== ΔELBO 全部完成 ====="
for I in $(seq 0 4); do
    LABEL="${LABELS[$I]}"
    echo "── $LABEL ──"
    grep -A 20 "=====" $ANALYSIS_DIR/${LABEL}.log 2>/dev/null | head -22
    echo ""
done
