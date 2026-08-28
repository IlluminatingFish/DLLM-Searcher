#!/bin/bash
# =============================================================================
# Phase 2b: Round 1 Sanity Check Rollout
#   π₁ (checkpoint-25) × 32题 × 8 rollouts = 256 trajectories
#   用和 Round 0 相同的前 32 题（rl_pool_256.jsonl），
#   目的：检测 search collapse / format collapse，对比 π₀ 行为
#
#   注意：--max_questions 4 → 每卡处理 shard 中前 4 题，8卡共 32题
#         （rl_pool_256.jsonl 按 rank::world 分片，每卡 32题，取前 4 = 全局前 32题）
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

# π₁ = ELBO-PG Round 1 checkpoint
MODEL=$AGENTIC/output/phase4_elbopg/reward_a/model

INPUT=$ROOT/dLLM_trainer/VRPO/data/rl_pool/rl_pool_256.jsonl   # 与 Round 0 相同，便于对比
OUTPUT=$AGENTIC/output/round1_sanity
LOG_DIR=$OUTPUT/logs
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Phase 2b: Round 1 Sanity Check  $(date)"
echo "Policy  : π₁ (checkpoint-25 ELBO-PG)"
echo "Model   : $MODEL"
echo "Input   : $INPUT (32 题，与 Round 0 相同)"
echo "Output  : $OUTPUT"
echo "================================================================"

# 交错启动 8 个 rank（每卡 4 题 × 8 rollouts = 32 trajectories/卡）
for RANK in $(seq 0 7); do
    SESSION="r1sanity_rank${RANK}"
    LOG="$LOG_DIR/rank${RANK}.log"

    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$RANK
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_${USER}

echo "=== rank $RANK 开始 \$(date) ==="
$PYTHON -u $AGENTIC/collect_round0_rollouts.py \
    --rank          $RANK \
    --world         8 \
    --model         "$MODEL" \
    --input         "$INPUT" \
    --output        "$OUTPUT" \
    --n_rolls       8 \
    --round_id      1 \
    --max_questions 4 \
    2>&1 | tee "$LOG"
echo "=== rank $RANK 完成 exit=\$? \$(date) ==="
INNER
    chmod +x /tmp/${SESSION}.sh

    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  rank $RANK launched (GPU=$RANK)  log=$LOG"

    # 每 2 个 rank 后等 90s，避免并发加载模型撞 NFS
    if [ $RANK -eq 1 ] || [ $RANK -eq 3 ] || [ $RANK -eq 5 ]; then
        echo "  等待 90s..."
        sleep 90
    fi
done

echo ""
echo "================================================================"
echo "全部 8 个 rank 已启动（π₁ sanity check）"
echo "监控: screen -list | grep r1sanity"
echo "进度: tail -f $LOG_DIR/rank0.log"
echo "================================================================"

# 等待所有 rank 完成
echo ""
echo "等待所有 rank 完成..."
while true; do
    DONE=0
    for RANK in $(seq 0 7); do
        screen -list | grep -q "r1sanity_rank${RANK}" || DONE=$((DONE+1))
    done
    echo "  $(date)  完成 rank 数: $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 120
done

echo ""
echo "所有 rank 完成！合并并对比 π₀ vs π₁..."
$PYTHON - << 'EOF'
import json, statistics
from collections import Counter
from pathlib import Path

# ── 加载 π₁ sanity rollouts ────────────────────────────────────────────────
OUT  = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round1_sanity/round1")
recs_pi1 = []
for f in sorted(OUT.glob("rollouts_rank*.jsonl")):
    with open(f) as fp:
        rows = [json.loads(l) for l in fp if l.strip()]
    print(f"  {f.name}: {len(rows)} records")
    recs_pi1.extend(rows)

merged_file = OUT.parent / "round1_sanity_merged.jsonl"
with open(merged_file, "w") as f:
    for r in recs_pi1:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── 加载 π₀ round0 对应的同一批题 ─────────────────────────────────────────
pi1_qids = set(r["question_id"] for r in recs_pi1)
recs_pi0 = []
with open("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round0_rollouts/round0_merged.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r["question_id"] in pi1_qids:
                recs_pi0.append(r)

def stats(recs, label):
    n = len(recs)
    search_rate = sum(1 for r in recs if r.get("num_searches", 0) > 0) / n
    avg_search  = sum(r.get("num_searches", 0) for r in recs) / n
    ans_rate    = sum(1 for r in recs if r.get("valid_format")) / n
    term_dist   = Counter(r.get("terminated_by", "?") for r in recs)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  search rate       : {search_rate*100:.1f}%")
    print(f"  avg search calls  : {avg_search:.2f}")
    print(f"  answer rate       : {ans_rate*100:.1f}%")
    print(f"  terminated_by     : {dict(term_dist)}")
    return search_rate, avg_search, ans_rate

sr0, as0, ar0 = stats(recs_pi0, "π₀ (SFT ckpt_6) — same 32 questions")
sr1, as1, ar1 = stats(recs_pi1, "π₁ (ELBO-PG ckpt-25)")

print(f"\n=== π₀ → π₁ 变化 ===")
print(f"  search rate   : {sr0*100:.1f}% → {sr1*100:.1f}%  ({(sr1-sr0)*100:+.1f}pp)")
print(f"  avg search    : {as0:.2f} → {as1:.2f}  ({as1-as0:+.2f})")
print(f"  answer rate   : {ar0*100:.1f}% → {ar1*100:.1f}%  ({(ar1-ar0)*100:+.1f}pp)")

# ── 诊断：search collapse / format collapse ────────────────────────────────
print(f"\n=== Collapse 检测 ===")
if sr1 < 0.5:
    print("  [WARN] search collapse 风险：π₁ search rate < 50%！")
else:
    print(f"  ✓ search rate 正常 ({sr1*100:.1f}%)")
if ar1 < 0.8:
    print("  [WARN] format collapse 风险：π₁ answer rate < 80%！")
else:
    print(f"  ✓ format 正常 ({ar1*100:.1f}%)")

qid_cnt = Counter(r["question_id"] for r in recs_pi1)
incomplete = [q for q, c in qid_cnt.items() if c != 8]
if incomplete:
    print(f"  [WARN] {len(incomplete)} 个问题不足 8 rollouts")
else:
    print(f"  ✓ 所有 {len(qid_cnt)} 题严格 8 rollouts")

print(f"\n输出: {merged_file}")
print("Sanity check 完成。如无 collapse，可继续启动正式 Round 1 rollout。")
EOF

echo ""
echo "Sanity check 完成: $(date)"
echo "查看结果后，如一切正常，运行: bash run_round1_rollout.sh"
