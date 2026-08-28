#!/bin/bash
# run_hermes_eval_analysis.sh
# 在 hermes 上同时跑：
#   GPU 0-7: π₆ eval_600（8GPU × 75题）
#   GPU 0-4: ΔELBO 分析 5对（与 eval 共享，顺序加载峰值 ~16GB）
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
OUTBASE=$AGENTIC/output/onpolicy_loop
ANALYSIS_DIR=$AGENTIC/output/delta_elbo_analysis

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_main
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p $ANALYSIS_DIR

PI6_MODEL=$OUTBASE/pi6/model
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl
EVAL_DIR=$OUTBASE/pi6/eval
mkdir -p $EVAL_DIR

log "=== hermes: π₆ eval_600 + ΔELBO 分析同时启动 ==="
log "hostname: $(hostname)"

# ── eval: GPU 0-7，每张 75 题（8×75=600）
log ""
log "=== 启动 π₆ eval_600 (GPU 0-7, 75题/GPU) ==="
for GPU in $(seq 0 7); do
    OFFSET=$((GPU * 75))
    SESSION="pi6_eval_${GPU}"
    INNER=/tmp/hermes_${SESSION}.sh
    cat > $INNER << INNER_EOF
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_eval_${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_eval_${GPU}
${PYTHON} -u ${ROOT}/my_eval/run_llada_eval.py \
    --model        "${PI6_MODEL}" \
    --input        "${EVAL_INPUT}" \
    --output       "${EVAL_DIR}/shard${GPU}.jsonl" \
    --offset       ${OFFSET} \
    --max_samples  75 \
    2>&1 | tee "${EVAL_DIR}/shard${GPU}.log"
INNER_EOF
    chmod +x $INNER
    screen -dmS "$SESSION" bash $INNER
    log "  eval GPU${GPU} 启动 (offset=${OFFSET})"
    sleep 30
done

# ── analysis: GPU 0-4，5对，等 eval 模型加载后启动
log ""
log "=== 等 2 分钟后启动 ΔELBO 分析 (GPU 0-4) ==="

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

# 等待 eval 开始加载（GPU 0 已启动后约 2 分钟）
sleep 120

for I in $(seq 0 4); do
    GPU=$I
    LABEL="${LABELS[$I]}"
    SESSION="delta_elbo_${I}"
    INNER=/tmp/hermes_${SESSION}.sh
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
    log "  ΔELBO $LABEL → GPU ${GPU}"
    sleep 30
done

log ""
log "=== 全部已启动 ==="
screen -list 2>/dev/null | grep -E "pi6_eval|delta_elbo"

# ── 等 eval 完成
log ""
log "等待 eval 完成..."
while true; do
    DONE=0
    for GPU in $(seq 0 7); do
        screen -list 2>/dev/null | grep -q "pi6_eval_${GPU}" || DONE=$((DONE+1))
    done
    log "  eval: $DONE/8 shard 完成"
    [ $DONE -eq 8 ] && break
    sleep 60
done

log "eval 全部完成！合并打分..."
cat $EVAL_DIR/shard*.jsonl > $EVAL_DIR/merged.jsonl
$PYTHON $ROOT/my_eval/cal_acc.py --data $EVAL_DIR/merged.jsonl 2>&1 | tee $EVAL_DIR/score.log
log ""
grep -E "CEM-1|LLM Judge|Answer rate|Total" $EVAL_DIR/score.log

# ── 等 analysis 完成
log ""
log "等待 ΔELBO 分析完成..."
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
