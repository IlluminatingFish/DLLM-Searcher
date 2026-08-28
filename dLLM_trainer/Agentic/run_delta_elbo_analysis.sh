#!/bin/bash
# run_delta_elbo_analysis.sh
# 等待 π₆ 完成，然后并行跑 3 对 (π_n, π_{n+1}) 的 ΔELBO 分析
# 每对用一张 GPU，每对约 30-40 min
#
# 分析对：
#   Round 0: π₀ → π₁  (最大提升轮，黄金参考)
#   Round 1: π₁ → π₂  (提升后第一次下降)
#   Round 2: π₂ → π₃  (on-policy 首轮，有完整 advantage 日志)
#
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python
OUTBASE=$AGENTIC/output/delta_elbo_analysis

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUTBASE

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 等待 π₆ eval 完成 ─────────────────────────────────────────────────────────
log "等待 π₆ eval 完成..."
SCORE6=$AGENTIC/output/onpolicy_loop/pi6/eval/score.log
until [ -s "$SCORE6" ]; do sleep 60; done
log "π₆ eval 完成！"
log ""
grep -E "CEM-1|LLM Judge" $SCORE6
log ""

# ── 定义分析对 ────────────────────────────────────────────────────────────────
declare -a LABELS=("Round0_pi0→pi1"  "Round1_pi1→pi2"  "Round2_pi2→pi3")
declare -a BEFORES=(
  "$ROOT/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
  "$AGENTIC/output/phase4_elbopg/reward_a/model"
  "$AGENTIC/output/phase4_elbopg_round2/reward_a/model"
)
declare -a AFTERS=(
  "$AGENTIC/output/phase4_elbopg/reward_a/model"
  "$AGENTIC/output/phase4_elbopg_round2/reward_a/model"
  "$AGENTIC/output/onpolicy_loop/pi3/model"
)
declare -a DATAS=(
  "$AGENTIC/output/round0_rollouts/train_reward_a.jsonl"
  "$AGENTIC/output/round1_rollouts/train_reward_a.jsonl"
  "$AGENTIC/output/onpolicy_loop/pi3/train_reward_a.jsonl"
)
declare -a GPUS=(0 1 2)

# ── 并行启动 ─────────────────────────────────────────────────────────────────
log "并行启动 3 对 ΔELBO 分析（GPU 0/1/2）..."

for I in 0 1 2; do
  LABEL="${LABELS[$I]}"
  BEFORE="${BEFORES[$I]}"
  AFTER="${AFTERS[$I]}"
  DATA="${DATAS[$I]}"
  GPU="${GPUS[$I]}"
  OUT="$OUTBASE/${LABEL}.jsonl"
  SESSION="delta_elbo_${I}"

  INNER=/tmp/${SESSION}.sh
  cat > $INNER << INNER_EOF
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_${GPU}
cd ${VRPO}
${PYTHON} -u ${AGENTIC}/analyze_delta_elbo.py \
  --before   "${BEFORE}" \
  --after    "${AFTER}" \
  --data     "${DATA}" \
  --output   "${OUT}" \
  --label    "${LABEL}" \
  --gpu      0 \
  --n_mc     4 \
  --max_samples 256 \
  2>&1 | tee "${OUTBASE}/${LABEL}.log"
INNER_EOF
  chmod +x $INNER
  screen -dmS "$SESSION" bash $INNER
  log "  [$I] $LABEL → GPU $GPU（screen: $SESSION）"
done

log ""
log "等待所有分析完成..."
while true; do
  DONE=0
  for I in 0 1 2; do
    screen -list 2>/dev/null | grep -q "delta_elbo_${I}" || DONE=$((DONE+1))
  done
  log "  $DONE/3 对已完成"
  [ $DONE -eq 3 ] && break
  sleep 120
done

log ""
log "===== ΔELBO Analysis 完整结果 ====="
for I in 0 1 2; do
  LABEL="${LABELS[$I]}"
  LOG="$OUTBASE/${LABEL}.log"
  echo ""
  echo "── $LABEL ──"
  grep -A 20 "=====" $LOG 2>/dev/null | head -25
done

log ""
log "原始数据已保存到: $OUTBASE/"
log "用 python 读取 .jsonl 做进一步分析（散点图、分布等）"
