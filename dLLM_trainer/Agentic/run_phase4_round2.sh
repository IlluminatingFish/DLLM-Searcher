#!/bin/bash
# =============================================================================
# Phase 4 Round 2: π₁ + D₁ → ELBO-PG → π₂
#
#   输入：round1_rollouts/train_reward_a.jsonl（π₁ 在新 256题上的 rollout + reward）
#   基础模型：π₁（ELBO-PG checkpoint-25）
#   输出：π₂ 模型 → output/phase4_elbopg_round2/reward_a/model
#   之后自动触发 eval_600
#
#   完成后对比 π₀ / π₁ / π₂ 的 learning curve。
# =============================================================================
set -uo pipefail

ROOT=/research/cbim/vast/mz751/Projects/DLLM-Searcher
AGENTIC=$ROOT/dLLM_trainer/Agentic
VRPO=$ROOT/dLLM_trainer/VRPO
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TORCH_EXTENSIONS_HOME=/tmp/torch_extensions_${USER}
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export USE_ZERO_STAGE=3
export NCCL_DEBUG=WARN
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_HOME

# ── 路径 ──────────────────────────────────────────────────────────────────────
# 基础模型：π₁ (Round 1 ELBO-PG)
BASE_MODEL=$AGENTIC/output/phase4_elbopg/reward_a/model

# 训练数据：π₁ 在新 256题上采样的 D₁
TRAIN_FILE=$AGENTIC/output/round1_rollouts/train_reward_a.jsonl

# 输出目录
ROUND2_DIR=$AGENTIC/output/phase4_elbopg_round2
OUT_MODEL=$ROUND2_DIR/reward_a/model
mkdir -p "$ROUND2_DIR/reward_a"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROUND2_DIR/reward_a/train_${TIMESTAMP}.log"

echo "================================================================"
echo "Phase 4 Round 2: ELBO-PG  $(date)"
echo "Base model : π₁  $BASE_MODEL"
echo "Train data : D₁  $TRAIN_FILE ($(wc -l < $TRAIN_FILE) records)"
echo "Output     : π₂  $OUT_MODEL"
echo "================================================================"

# ── 检查前提 ──────────────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] 训练数据不存在: $TRAIN_FILE"
    echo "  请先运行: bash run_round1_rewards.sh"
    exit 1
fi

N=$(wc -l < "$TRAIN_FILE")
echo "  训练数据: $N records"
if [ "$N" -lt 500 ]; then
    echo "[WARN] 训练数据过少 ($N < 500)，可能 effective groups 不足，继续但请留意"
fi

# ── 生成临时训练 config（从 Round 1 config 派生，改 model_name_or_path）───────
ROUND2_CONFIG="/tmp/grpo_round2_reward_a_${TIMESTAMP}.yaml"
BASE_CONFIG="$AGENTIC/configs/grpo_phase4_reward_a.yaml"

# 替换 model_name_or_path → π₁，output_dir → round2，dataset_path → D₁
sed "s|model_name_or_path:.*|model_name_or_path: $BASE_MODEL|;
     s|output_dir:.*|output_dir: $OUT_MODEL|;
     s|run_name:.*|run_name: phase4_elbopg_round2_reward_a_${TIMESTAMP}|;
     s|dataset_path:.*|dataset_path: $TRAIN_FILE|" \
    "$BASE_CONFIG" > "$ROUND2_CONFIG"

echo "  临时 config: $ROUND2_CONFIG"
echo ""

# ── 训练 ──────────────────────────────────────────────────────────────────────
cd "$VRPO"

echo "[$(date)] 启动 ELBO-PG Round 2 训练..."
set +e
accelerate launch \
    --config_file "$AGENTIC/configs/accel_zero3_8gpu.yaml" \
    --num_processes 8 \
    my_train/llada_grpo_train.py \
    --config "$ROUND2_CONFIG" \
    2>&1 | tee "$LOG"
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Round 2 训练失败 (exit=$TRAIN_EXIT)"
    exit $TRAIN_EXIT
fi

echo ""
echo "[OK] Round 2 训练完成: $(date)"
echo "  π₂ 保存到: $OUT_MODEL"

# ── eval_600（8 GPU，每 GPU 75 题）──────────────────────────────────────────
echo ""
echo "================================================================"
echo "π₂ eval_600 开始  $(date)"
echo "================================================================"

EVAL_DIR="$ROUND2_DIR/reward_a/eval_$(date +%H%M%S)"
mkdir -p "$EVAL_DIR"
EVAL_INPUT=$ROOT/dLLM_trainer/VRPO/data/eval_600.jsonl

for GPU in $(seq 0 7); do
    OFFSET=$((GPU * 75))
    SESSION="r2eval_gpu${GPU}"
    cat > /tmp/${SESSION}.sh << INNER
#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:\$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -u $ROOT/my_eval/run_llada_eval.py \
    --model       "$OUT_MODEL" \
    --input       "$EVAL_INPUT" \
    --output      "$EVAL_DIR/shard${GPU}.jsonl" \
    --offset      $OFFSET \
    --max_samples 75 \
    2>&1 | tee "$EVAL_DIR/shard${GPU}.log"
INNER
    chmod +x /tmp/${SESSION}.sh
    screen -dmS "$SESSION" bash /tmp/${SESSION}.sh
    echo "  GPU$GPU launched (offset=$OFFSET)"
    [ $GPU -lt 7 ] && sleep 60
done

echo "  等待 8 个 eval 进程完成..."
while true; do
    DONE=0
    for GPU in $(seq 0 7); do
        screen -list | grep -q "r2eval_gpu${GPU}" || DONE=$((DONE+1))
    done
    echo "  $(date)  完成 $DONE/8"
    [ $DONE -eq 8 ] && break
    sleep 120
done

# 合并 + 打分
cat "$EVAL_DIR/"shard*.jsonl > "$EVAL_DIR/merged.jsonl"
$PYTHON $ROOT/my_eval/cal_acc.py \
    --data "$EVAL_DIR/merged.jsonl" \
    2>&1 | tee "$EVAL_DIR/score.log"

# ── 打印三轮对比 ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Learning Curve: π₀ → π₁ → π₂"
echo "================================================================"
$PYTHON - << 'EOF'
import json, pathlib

BASE = pathlib.Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher")

def read_score(score_log):
    if not pathlib.Path(score_log).exists():
        return None
    lines = pathlib.Path(score_log).read_text().splitlines()
    d = {}
    for l in lines:
        if "CEM-1" in l and "short" in l.lower():
            import re
            m = re.search(r"(\d+\.\d+)%", l)
            if m: d["cem1"] = float(m.group(1))
        if "LLM Judge" in l:
            import re
            m = re.search(r"(\d+\.\d+)%", l)
            if m: d["judge"] = float(m.group(1))
        if "Token F1 mean" in l:
            import re
            m = re.search(r"(\d+\.\d+)%", l)
            if m: d["f1"] = float(m.group(1))
    return d

# 从上一轮 phase4_elbopg 里找最近的 score.log
def find_latest_score(base_dir):
    logs = sorted(pathlib.Path(base_dir).glob("eval_*/score.log"))
    return str(logs[-1]) if logs else None

s0_log = find_latest_score(BASE / "dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6")
s1_log = find_latest_score(BASE / "dLLM_trainer/Agentic/output/phase4_elbopg/reward_a")
s2_log = find_latest_score(BASE / "dLLM_trainer/Agentic/output/phase4_elbopg_round2/reward_a")

models = [("π₀ SFT ckpt_6", s0_log), ("π₁ ELBO-PG R1", s1_log), ("π₂ ELBO-PG R2", s2_log)]
print(f"{'Model':<20} {'CEM-1':>8} {'Judge':>8} {'Token F1':>10}")
print("-" * 50)
for name, log in models:
    if log:
        s = read_score(log)
        if s:
            print(f"{name:<20} {s.get('cem1','-'):>7.2f}% {s.get('judge','-'):>7.2f}% {s.get('f1','-'):>9.2f}%")
        else:
            print(f"{name:<20} (score.log 解析失败: {log})")
    else:
        print(f"{name:<20} (无 score.log)")
EOF

echo ""
echo "================================================================"
echo "下一步："
echo "  若 π₂ 稳定，继续: π₂ → fresh rollout D₂ → ELBO-PG → π₃"
echo "  脚本模板：复制 run_round1_rollout.sh，改 MODEL 和 round_id=2"
echo "================================================================"
echo "Phase 4 Round 2 完成: $(date)"
