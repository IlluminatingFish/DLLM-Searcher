#!/bin/bash
# run_delta_elbo_all.sh — 3 对 ΔELBO 分析，GPU 0/1/2 并行
# Round3: π₃→π₄ (GPU 0)
# Round4: π₄→π₅ (GPU 1)
# Round5: π₅→π₆ (GPU 2)

set -e
ADIR=/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic
OUTBASE=$ADIR/output/onpolicy_loop
RESDIR=$ADIR/results
SCRIPT=$ADIR/analyze_delta_elbo.py

mkdir -p $RESDIR
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
mkdir -p $TRITON_CACHE_DIR

echo "======================================================"
echo "ΔELBO Analysis — 3 对 (Round3/4/5)"
echo "======================================================"

# ── Round 3: π₃ → π₄ (GPU 0) ──────────────────────────────
nohup bash -c "
  TRITON_CACHE_DIR=/tmp/triton_r3_\$\$
  mkdir -p \$TRITON_CACHE_DIR
  export TRITON_CACHE_DIR
  python $SCRIPT \
    --before  $OUTBASE/pi3/model \
    --after   $OUTBASE/pi4/model \
    --data    $OUTBASE/pi4/train_reward_a.jsonl \
    --output  $RESDIR/delta_elbo_round3.jsonl \
    --label   'Round3_pi3→pi4' \
    --gpu     0 \
    --n_mc    4 \
    --max_samples 256 \
    2>&1 | tee $RESDIR/round3.log
  echo '[Round3 done]'
" > /dev/null 2>&1 &
PID3=$!
echo "Round3 启动: PID=$PID3 (GPU 0)"

# ── Round 4: π₄ → π₅ (GPU 1) ──────────────────────────────
nohup bash -c "
  TRITON_CACHE_DIR=/tmp/triton_r4_\$\$
  mkdir -p \$TRITON_CACHE_DIR
  export TRITON_CACHE_DIR
  python $SCRIPT \
    --before  $OUTBASE/pi4/model \
    --after   $OUTBASE/pi5/model \
    --data    $OUTBASE/pi5/train_reward_a.jsonl \
    --output  $RESDIR/delta_elbo_round4.jsonl \
    --label   'Round4_pi4→pi5' \
    --gpu     1 \
    --n_mc    4 \
    --max_samples 256 \
    2>&1 | tee $RESDIR/round4.log
  echo '[Round4 done]'
" > /dev/null 2>&1 &
PID4=$!
echo "Round4 启动: PID=$PID4 (GPU 1)"

# ── Round 5: π₅ → π₆ (GPU 2) ──────────────────────────────
nohup bash -c "
  TRITON_CACHE_DIR=/tmp/triton_r5_\$\$
  mkdir -p \$TRITON_CACHE_DIR
  export TRITON_CACHE_DIR
  python $SCRIPT \
    --before  $OUTBASE/pi5/model \
    --after   $OUTBASE/pi6/model \
    --data    $OUTBASE/pi6/train_reward_a.jsonl \
    --output  $RESDIR/delta_elbo_round5.jsonl \
    --label   'Round5_pi5→pi6' \
    --gpu     2 \
    --n_mc    4 \
    --max_samples 256 \
    2>&1 | tee $RESDIR/round5.log
  echo '[Round5 done]'
" > /dev/null 2>&1 &
PID5=$!
echo "Round5 启动: PID=$PID5 (GPU 2)"

echo ""
echo "3 个任务并行跑中。查看进度:"
echo "  tail -f $RESDIR/round3.log"
echo "  tail -f $RESDIR/round4.log"
echo "  tail -f $RESDIR/round5.log"
echo ""
echo "等待全部完成..."
wait $PID3 && echo "[Round3 ✓]" || echo "[Round3 ✗]"
wait $PID4 && echo "[Round4 ✓]" || echo "[Round4 ✗]"
wait $PID5 && echo "[Round5 ✓]" || echo "[Round5 ✗]"

echo ""
echo "======================================================"
echo "全部完成！汇总:"
for r in 3 4 5; do
    echo ""
    echo "--- Round $r ---"
    tail -20 $RESDIR/round${r}.log 2>/dev/null | grep -E "Spearman|Sign-align|mean ΔELBO|Label|N :"
done
