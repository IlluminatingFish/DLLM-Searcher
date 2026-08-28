#!/bin/bash
# 3对 ΔELBO analysis（π₁ 不存在，只跑 Round2-4）
# GPU 5-7，与 eval GPU 0-7 共享（各有 ~23GB 余量，够用 16GB 的 analysis）
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
log "启动 3 对 ΔELBO analysis (Round2~4, GPU 5-7)..."

# π₁ 不存在，Round0/1 跳过
declare -a LABELS=(
    "Round2_pi2_to_pi3"
    "Round3_pi3_to_pi4"
    "Round4_pi4_to_pi5"
)
declare -a BEFORES=(
    "$AGENTIC/output/phase4_elbopg_round2/reward_a/model"
    "$AGENTIC/output/onpolicy_loop/pi3/model"
    "$AGENTIC/output/onpolicy_loop/pi4/model"
)
declare -a AFTERS=(
    "$AGENTIC/output/onpolicy_loop/pi3/model"
    "$AGENTIC/output/onpolicy_loop/pi4/model"
    "$AGENTIC/output/onpolicy_loop/pi5/model"
)
declare -a DATAS=(
    "$AGENTIC/output/onpolicy_loop/pi3/train_reward_a.jsonl"
    "$AGENTIC/output/onpolicy_loop/pi4/train_reward_a.jsonl"
    "$AGENTIC/output/onpolicy_loop/pi5/train_reward_a.jsonl"
)
declare -a GPUS=(5 6 7)

for I in 0 1 2; do
    GPU="${GPUS[$I]}"
    LABEL="${LABELS[$I]}"
    SESSION="delta_elbo_${I}"
    INNER=/tmp/hermes_ana3_${I}.sh
    cat > $INNER << INNER_EOF
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_ana3_${I}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_ana3_${I}
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
    sleep 5
done

log ""
log "已启动:"
screen -list 2>/dev/null | grep delta_elbo

log ""
log "等待 3 对 analysis 完成..."
while true; do
    DONE=0
    for I in 0 1 2; do
        screen -list 2>/dev/null | grep -q "delta_elbo_${I}" || DONE=$((DONE+1))
    done
    log "  analysis: $DONE/3 对完成"
    [ $DONE -eq 3 ] && break
    sleep 120
done

log ""
log "===== ΔELBO 全部完成 ====="
for I in 0 1 2; do
    LABEL="${LABELS[$I]}"
    echo "── $LABEL ──"
    grep -A 20 "=====" $ANALYSIS_DIR/${LABEL}.log 2>/dev/null | head -22
    echo ""
done
