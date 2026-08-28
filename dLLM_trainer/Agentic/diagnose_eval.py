#!/usr/bin/env python3
"""
诊断脚本 ②：SFT vs Round 1 模型 eval 对比
复用 collect_agentic_rollouts.py 的生成逻辑，不重新实现。

用法（在 hercules 上运行）：
  # 在 tmux 中同时运行：
  CUDA_VISIBLE_DEVICES=3 python diagnose_eval.py --model sft
  CUDA_VISIBLE_DEVICES=7 python diagnose_eval.py --model round1

  # 两个都跑完后汇总对比：
  python diagnose_eval.py --compare
"""

import argparse, json, re, string, os, sys, subprocess
from pathlib import Path

ROOT      = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher")
SFT_PATH  = ROOT / "dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
R1_PATH   = ROOT / "dLLM_trainer/Agentic/output/plan_a/models/round1/model"
EVAL_FILE = ROOT / "dLLM_trainer/VRPO/data/eval_600.jsonl"
AGENTIC   = ROOT / "dLLM_trainer/Agentic"
OUT_DIR   = AGENTIC / "output/plan_a/logs/diag_eval"
ROLLOUT_SCRIPT = AGENTIC / "collect_agentic_rollouts.py"
PYTHON    = "/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python"

N_QUESTIONS = 30
SEED        = 42

# ── F1/EM 函数 ─────────────────────────────────────────────────────────────────
def _normalize(s):
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())

def _bool_map(s): return {"True": "yes", "False": "no"}.get(s, s)

def _first_sentence(s):
    s = s.strip()
    for sep in [". ", ".\n", "\n\n", "\n"]:
        idx = s.find(sep)
        if idx > 10: return s[:idx+1]
    return s[:200]

def _yes_no(s):
    s = s.strip().lower()
    if re.match(r"^(yes[,\s]|yes$)", s): return "yes"
    if re.match(r"^(no[,\s]|no$)", s):   return "no"
    if re.search(r"\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b", s): return "no"
    if re.search(r"\bthe same\b|\bboth\b|\byes\b", s): return "yes"
    return None

def token_f1(pred, gt):
    if not pred or not str(pred).strip(): return 0.0
    gt_k   = _normalize(_bool_map(_first_sentence(str(gt))))
    pred_n = _normalize(_bool_map(str(pred)))
    gt_t   = set(gt_k.split())
    pd_t   = set(pred_n.split())
    if not gt_t: return 0.0
    common = gt_t & pd_t
    if not common: return 0.0
    p = len(common)/max(len(pd_t),1)
    r = len(common)/len(gt_t)
    return 2*p*r/(p+r+1e-9)

def norm_em(pred, gt):
    if not pred: return False
    return _normalize(_bool_map(_first_sentence(str(gt)))) == _normalize(_bool_map(str(pred)))

def correctness(pred, gt):
    if not pred: return 0.0
    gt_yn   = _yes_no(str(gt))
    pred_yn = _yes_no(str(pred))
    if gt_yn and pred_yn:
        return 1.0 if gt_yn == pred_yn else 0.0
    return token_f1(pred, gt)

# ── 生成 eval 问题子集 ─────────────────────────────────────────────────────────
def make_eval_subset():
    import random
    random.seed(SEED)
    with open(EVAL_FILE) as f:
        qs = [json.loads(l) for l in f if l.strip()]
    random.shuffle(qs)
    subset = qs[:N_QUESTIONS]
    # collect_agentic_rollouts.py 需要有 question / answer 字段
    normalized = []
    for q in subset:
        ans = q.get("answer", "")
        if not ans and isinstance(q.get("answers"), list):
            ans = q["answers"][0]
        normalized.append({"question": q["question"], "answer": ans})
    return normalized

# ── 运行单个模型的 rollout ────────────────────────────────────────────────────
def run_model(model_name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 写问题子集到临时 JSONL
    subset = make_eval_subset()
    input_file = OUT_DIR / f"eval_subset_{N_QUESTIONS}q.jsonl"
    with open(input_file, "w") as f:
        for q in subset:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"评测子集: {len(subset)} 题 → {input_file}")

    model_path = str(SFT_PATH if model_name == "sft" else R1_PATH)
    rollout_out = OUT_DIR / f"rollout_{model_name}"
    rollout_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[{model_name}] 开始 rollout，模型: {model_path}")
    print(f"[{model_name}] 输出目录: {rollout_out}")

    # 2. 调用 collect_agentic_rollouts.py (rank=0, world=1, n_rolls=1)
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{os.environ.get('USER','mz751')}_{model_name}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        PYTHON, str(ROLLOUT_SCRIPT),
        "--model",       model_path,
        "--input",       str(input_file),
        "--output",      str(rollout_out),
        "--n_questions", str(N_QUESTIONS),
        "--n_rolls",     "1",
        "--rank",        "0",
        "--world",       "1",
        "--plan",        "A",
    ]
    print(f"[{model_name}] CMD: {' '.join(cmd)}")
    ret = subprocess.run(cmd, env=env, cwd=str(AGENTIC))
    if ret.returncode != 0:
        print(f"[{model_name}] rollout 失败 (exit={ret.returncode})")
        return

    # 3. 读取结果并计算 F1/EM
    rollout_file = rollout_out / "plan_a" / "rank0.jsonl"
    if not rollout_file.exists():
        # 有时输出在不同子目录，搜索
        candidates = list(rollout_out.rglob("rank0.jsonl"))
        if candidates:
            rollout_file = candidates[0]
        else:
            print(f"[{model_name}] 找不到 rollout 输出文件")
            return

    records = []
    with open(rollout_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 用本地 F1/EM 重新计算（不依赖 reward）
    results = []
    for r in records:
        pred = r.get("prediction", "")
        gt   = r.get("answer", "")
        f1   = correctness(pred, gt)
        em   = norm_em(pred, gt)
        results.append({
            "question":   r.get("question", "")[:80],
            "gold":       _first_sentence(str(gt))[:60],
            "prediction": str(pred)[:80] if pred else "(None)",
            "f1":         round(f1, 4),
            "em":         em,
            "reward":     r.get("reward", 0.0),
            "n_searches": r.get("n_searches", 0),
        })

    # 保存详细结果
    detail_file = OUT_DIR / f"results_{model_name}.jsonl"
    with open(detail_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印汇总
    n = len(results)
    f1s = [r["f1"] for r in results]
    ems = [r["em"] for r in results]
    srch = [r["n_searches"] for r in results]
    rwds = [r["reward"] for r in results]

    print(f"\n{'='*60}")
    print(f"模型: {model_name}  题数: {n}")
    print(f"  F1 均值         : {sum(f1s)/n:.4f}")
    print(f"  F1 > 0.5        : {sum(1 for f in f1s if f > 0.5)}/{n} "
          f"({100*sum(1 for f in f1s if f>0.5)/n:.1f}%)")
    print(f"  F1 > 0.8        : {sum(1 for f in f1s if f > 0.8)}/{n} "
          f"({100*sum(1 for f in f1s if f>0.8)/n:.1f}%)")
    print(f"  Exact Match     : {sum(ems)}/{n} ({100*sum(ems)/n:.1f}%)")
    print(f"  平均搜索次数    : {sum(srch)/n:.2f}")
    print(f"  reward>0（旧）  : {sum(1 for r in rwds if r>0)}/{n}")
    print(f"{'='*60}")
    print(f"详细结果: {detail_file}")


# ── 对比两个模型结果 ──────────────────────────────────────────────────────────
def compare():
    sft_file = OUT_DIR / "results_sft.jsonl"
    r1_file  = OUT_DIR / "results_round1.jsonl"
    if not sft_file.exists() or not r1_file.exists():
        print("需要先跑完两个模型的 eval")
        return

    def load(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    sft = load(sft_file)
    r1  = load(r1_file)

    print("\n" + "="*80)
    print("SFT vs Round 1 对比")
    print("="*80)

    # 按问题对齐（假设顺序一致）
    n = min(len(sft), len(r1))
    better, worse, same = 0, 0, 0

    print(f"\n{'#':>3}  {'F1-SFT':>8}  {'F1-R1':>8}  {'Δ':>7}  Q")
    print("-"*80)
    for i in range(n):
        s_f1 = sft[i]["f1"]
        r_f1 = r1[i]["f1"]
        delta = r_f1 - s_f1
        if delta > 0.05:  better += 1
        elif delta < -0.05: worse += 1
        else: same += 1
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        print(f"{i+1:>3}  {s_f1:>8.3f}  {r_f1:>8.3f}  {delta:>+7.3f} {arrow}  "
              f"{sft[i]['question'][:45]}")

    sft_f1s = [s["f1"] for s in sft[:n]]
    r1_f1s  = [r["f1"] for r in r1[:n]]
    sft_ems = [s["em"] for s in sft[:n]]
    r1_ems  = [r["em"] for r in r1[:n]]

    print("\n" + "="*80)
    print(f"{'指标':<20} {'SFT':>10} {'Round1':>10} {'变化':>10}")
    print("-"*80)
    print(f"{'F1 均值':<20} {sum(sft_f1s)/n:>10.4f} {sum(r1_f1s)/n:>10.4f} "
          f"{sum(r1_f1s)/n - sum(sft_f1s)/n:>+10.4f}")
    print(f"{'F1>0.5 占比':<20} {sum(1 for f in sft_f1s if f>0.5)/n:>10.2%} "
          f"{sum(1 for f in r1_f1s if f>0.5)/n:>10.2%} "
          f"{(sum(1 for f in r1_f1s if f>0.5) - sum(1 for f in sft_f1s if f>0.5))/n:>+10.2%}")
    print(f"{'Exact Match':<20} {sum(sft_ems)/n:>10.2%} {sum(r1_ems)/n:>10.2%} "
          f"{(sum(r1_ems)-sum(sft_ems))/n:>+10.2%}")
    print(f"{'F1提升题数':<20} {better:>10} {'/'} {n}")
    print(f"{'F1下降题数':<20} {worse:>10} {'/'} {n}")
    print(f"{'F1基本不变':<20} {same:>10} {'/'} {n}")
    print("="*80)


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sft", "round1"],
                        help="运行单个模型的 rollout eval")
    parser.add_argument("--compare", action="store_true",
                        help="对比两个已有结果")
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.model:
        run_model(args.model)
    else:
        parser.print_help()
