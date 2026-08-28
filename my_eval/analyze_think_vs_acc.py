"""
Per-sample analysis: think length vs. ACC_R, search behavior, turn count.
Generates JSON + prints a summary for each model.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

OUTDIR = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/llada_eval")

MODELS = [
    ("sft_llada ckpt_6 原始",       OUTDIR / "sft_llada_ckpt6_600_shortgt.jsonl"),
    ("sft_llada ckpt_6 强制",       OUTDIR / "sft_llada_ckpt6_forced_600_shortgt.jsonl"),
    ("x0pred 原版 ckpt_6",          OUTDIR / "x0pred_ckpt6_600_shortgt.jsonl"),
    ("x0pred 10ep ckpt_4 强制",     OUTDIR / "x0pred_10ep_ckpt_4_600_shortgt.jsonl"),
    ("x0pred 10ep ckpt_5 强制",     OUTDIR / "x0pred_10ep_ckpt_5_600_shortgt.jsonl"),
    ("x0pred 10ep ckpt_5 原始",     OUTDIR / "x0pred_10ep_ckpt_5_noforceanswer_shortgt.jsonl"),
    ("x0pred 10ep ckpt_6 强制",     OUTDIR / "x0pred_10ep_ckpt_6_600_shortgt.jsonl"),
]

# ---------- helpers ----------

def extract_think_len(text):
    """word count inside <think>...</think> blocks, return list per block"""
    blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return [len(b.split()) for b in blocks]

def has_tool_call(text):
    return bool(re.search(r'<tool_call>', text))

def is_correct(row):
    """substring-containment check (ACC_R logic).
    File format: answer=short_GT, prediction=model_output
    """
    gt   = str(row.get("answer", "")).strip()
    pred = str(row.get("prediction", "")).strip()
    if not gt or not pred:
        return None
    return gt.lower() in pred.lower()

def get_turns(row):
    """extract list of (role, content) from messages field"""
    turns = []
    for m in row.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, list):
            # flatten list-of-dicts content
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", str(part)))
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        turns.append((m.get("role",""), str(content)))
    return turns

def analyze_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    for row in rows:
        turns = get_turns(row)
        # assistant turns only
        asst_turns = [(r,c) for r,c in turns if r == "assistant"]

        # per-assistant-turn think lengths
        round_thinks = []
        for _, c in asst_turns:
            lens = extract_think_len(c)
            total = sum(lens)
            round_thinks.append(total)

        # did_search = at least one tool_call in any assistant turn
        did_search = any(has_tool_call(c) for _,c in asst_turns)

        n_turns = len(asst_turns)
        correct  = is_correct(row)
        results.append({
            "round_thinks": round_thinks,
            "did_search":   did_search,
            "n_asst_turns": n_turns,
            "correct":      correct,
        })
    return results

def group_stat(items, key_fn, val_fn):
    """items → dict of group→{n, n_correct, n_valid}"""
    groups = defaultdict(lambda: {"n":0,"n_correct":0,"n_valid":0})
    for it in items:
        k = key_fn(it)
        v = val_fn(it)
        groups[k]["n"] += 1
        if v is not None:
            groups[k]["n_valid"] += 1
            if v:
                groups[k]["n_correct"] += 1
    return groups

def acc(g):
    if g["n_valid"] == 0: return float("nan")
    return 100.0 * g["n_correct"] / g["n_valid"]

def pct(a, b):
    if b == 0: return 0.0
    return 100.0 * a / b

# ---------- main ----------

all_model_data = {}

for name, path in MODELS:
    if not path.exists():
        print(f"[SKIP] {name}: {path} not found")
        continue

    results = analyze_file(path)
    n = len(results)

    # --- overall ---
    n_valid   = sum(1 for r in results if r["correct"] is not None)
    n_correct = sum(1 for r in results if r["correct"])
    overall_acc = 100.0 * n_correct / n_valid if n_valid else float("nan")

    # --- think presence/absence split (R1) ---
    has_r1_think   = [r for r in results if r["round_thinks"] and r["round_thinks"][0] > 0]
    no_r1_think    = [r for r in results if not r["round_thinks"] or r["round_thinks"][0] == 0]

    def sub_acc(lst):
        v = [r["correct"] for r in lst if r["correct"] is not None]
        if not v: return float("nan"), 0
        return 100.0 * sum(v) / len(v), len(v)

    acc_has_think, n_has = sub_acc(has_r1_think)
    acc_no_think,  n_no  = sub_acc(no_r1_think)

    # --- search split ---
    did_search   = [r for r in results if r["did_search"]]
    no_search    = [r for r in results if not r["did_search"]]
    acc_search,  n_s  = sub_acc(did_search)
    acc_nosearch, n_ns = sub_acc(no_search)

    # --- think length buckets (R1) ---
    def r1_len(r):
        return r["round_thinks"][0] if r["round_thinks"] else 0

    buckets = {"0":[],  "1-10":[], "11-20":[], "21+": []}
    for r in results:
        l = r1_len(r)
        if l == 0:          buckets["0"].append(r)
        elif l <= 10:       buckets["1-10"].append(r)
        elif l <= 20:       buckets["11-20"].append(r)
        else:               buckets["21+"].append(r)

    bucket_stats = {}
    for k, lst in buckets.items():
        a, nv = sub_acc(lst)
        bucket_stats[k] = {"n": len(lst), "acc": round(a,1)}

    # --- turn count distribution ---
    turn_dist = defaultdict(int)
    for r in results:
        turn_dist[r["n_asst_turns"]] += 1

    # --- R2 think split (only samples that reached R2) ---
    r2_samples = [r for r in results if len(r["round_thinks"]) >= 2]
    has_r2_think = [r for r in r2_samples if r["round_thinks"][1] > 0]
    no_r2_think  = [r for r in r2_samples if r["round_thinks"][1] == 0]
    acc_r2_has, n_r2h = sub_acc(has_r2_think)
    acc_r2_no,  n_r2n = sub_acc(no_r2_think)

    model_data = {
        "name": name,
        "n": n,
        "overall_acc": round(overall_acc, 1),
        "think_split": {
            "has_r1_think": {"n": n_has, "acc": round(acc_has_think,1)},
            "no_r1_think":  {"n": n_no,  "acc": round(acc_no_think, 1)},
        },
        "search_split": {
            "did_search":  {"n": n_s,  "acc": round(acc_search,   1)},
            "no_search":   {"n": n_ns, "acc": round(acc_nosearch, 1)},
        },
        "r1_think_length_buckets": bucket_stats,
        "r2_think_split": {
            "has_r2_think": {"n": n_r2h, "acc": round(acc_r2_has,1)},
            "no_r2_think":  {"n": n_r2n, "acc": round(acc_r2_no, 1)},
        },
        "turn_dist": dict(sorted(turn_dist.items())),
    }
    all_model_data[name] = model_data

    # print summary
    print(f"\n{'='*60}")
    print(f"  {name}   (n={n}, ACC_R={overall_acc:.1f}%)")
    print(f"{'='*60}")
    print(f"  [R1 Think 有/无] 有think n={n_has} ACC={acc_has_think:.1f}% | 无think n={n_no} ACC={acc_no_think:.1f}%")
    print(f"  [Search 有/无]   搜索   n={n_s}  ACC={acc_search:.1f}%  | 不搜索  n={n_ns} ACC={acc_nosearch:.1f}%")
    print(f"  [R1 think词数分桶]")
    for k in ["0","1-10","11-20","21+"]:
        bs = bucket_stats[k]
        print(f"    {k:6s}: n={bs['n']:4d}  ACC={bs['acc']:.1f}%")
    print(f"  [R2 Think 有/无] 有think n={n_r2h} ACC={acc_r2_has:.1f}% | 无think n={n_r2n} ACC={acc_r2_no:.1f}%")
    print(f"  [Turn分布] {dict(sorted(turn_dist.items()))}")

# save JSON
out_path = OUTDIR / "think_acc_analysis.json"
with open(out_path, "w") as f:
    json.dump(all_model_data, f, ensure_ascii=False, indent=2)
print(f"\n\n[saved] {out_path}")
