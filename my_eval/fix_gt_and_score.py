#!/usr/bin/env python3
"""
用原始数据集的短答案替换 eval output 中冗长的 GPT-4o GT，然后计算 CEM-1 和 LLM Judge。
用法:
  python my_eval/fix_gt_and_score.py --data output/llada_eval/sft_llada_ckpt6_600.jsonl
"""
import argparse, json, os, re, string, sys
import rich
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── API config ──────────────────────────────────────────────────────────────
def _load_config():
    cfg = {}
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass
    return cfg

_cfg = _load_config()
API_KEY      = os.getenv("JUDGE_API_KEY") or _cfg.get("judge_api_key") or os.getenv("OPENAI_API_KEY") or _cfg.get("openai_api_key") or "YOUR_API_KEY"
API_BASE_URL = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or os.getenv("OPENAI_API_BASE") or _cfg.get("openai_api_base") or "YOUR_API_BASE_URL"
MODEL_NAME   = os.getenv("JUDGE_MODEL") or _cfg.get("judge_model") or "gpt-4o"
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Metrics ──────────────────────────────────────────────────────────────────
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'","'","´","`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)
    def lower(text): return text.lower()
    def replace_underscore(text): return text.replace("_", " ")
    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))

def bool_mapping(s):
    sl = s.strip().lower()
    if sl in ("yes","true","correct","right"): return "yes"
    if sl in ("no","false","incorrect","wrong"): return "no"
    return s

def cem1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction or "")).split()
    gt_list  = normalize_answer(bool_mapping(ground_truth or "")).split()
    return all(w in pre_list for w in gt_list)

def llm_judge_single(question, reference, prediction):
    pred_str = (prediction or "") or "(no answer)"
    prompt = f"""You are an expert evaluator for question answering systems.

Question: {question}
Reference Answer: {reference}
Model Prediction: {pred_str}

Judge whether the Model Prediction correctly answers the question based on the Reference Answer.
The prediction doesn't need to be word-for-word identical; it just needs to convey the correct answer.
Reply with only "correct" or "incorrect"."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=10,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        return "correct" in verdict
    except Exception as e:
        rich.print(f"[yellow]LLM Judge error: {e}[/yellow]")
        return False

# ── Build short-answer lookup maps ───────────────────────────────────────────
def build_hotpot_map():
    rich.print("[cyan]Loading HotpotQA train split...[/cyan]")
    ds = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)
    return {ex["question"].strip(): ex["answer"] for ex in ds}

def build_2wiki_map():
    rich.print("[cyan]Loading 2WikiMultiHopQA train split...[/cyan]")
    try:
        ds = load_dataset("wiki_hop", "original", split="train", trust_remote_code=True)
        return {ex["question"].strip(): ex["answer"] for ex in ds}
    except Exception:
        pass
    try:
        ds = load_dataset("THUDM/2WikiMultihopQA", split="train", trust_remote_code=True)
        return {ex["question"].strip(): ex["answer"] for ex in ds}
    except Exception as e:
        rich.print(f"[yellow]2wiki load failed: {e}, skipping[/yellow]")
        return {}

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True, help="eval output jsonl path")
    parser.add_argument("--skip_llm", action="store_true", help="skip LLM judge (faster)")
    args = parser.parse_args()

    hotpot_map = build_hotpot_map()
    wiki2_map  = build_2wiki_map()

    records = []
    missing = 0
    with open(args.data) as f:
        for line in f:
            d = json.loads(line)
            q = d["question"].strip()
            src = d.get("source", "")
            old_ans = d.get("answer", "")

            if src == "hotpotqa" and q in hotpot_map:
                d["answer"] = hotpot_map[q]
            elif src == "2wikimultihopqa" and q in wiki2_map:
                d["answer"] = wiki2_map[q]
            else:
                # 找不到短答案：保留原始，但尝试从冗长答案中提取关键实体
                missing += 1
            records.append(d)

    rich.print(f"总共 {len(records)} 题，找不到短答案的: {missing}")

    cem1_count = 0
    llm_count  = 0
    total      = len(records)

    def process(r):
        c = cem1(r.get("prediction") or "", r["answer"])
        if args.skip_llm:
            return c, False
        j = llm_judge_single(r["question"], r["answer"], r.get("prediction") or "")
        return c, j

    rich.print("开始评测...")
    with ThreadPoolExecutor(max_workers=50) as ex:
        futures = {ex.submit(process, r): r for r in records}
        for fut in tqdm(as_completed(futures), total=total):
            c, j = fut.result()
            if c: cem1_count += 1
            if j: llm_count  += 1

    rich.print(f"\n{'='*50}")
    rich.print(f"[bold]数据集: {args.data}[/bold]")
    rich.print(f"CEM-1  (短答案覆盖): {cem1_count}/{total} = {cem1_count/total*100:.2f}%")
    if not args.skip_llm:
        rich.print(f"LLM Judge           : {llm_count}/{total} = {llm_count/total*100:.2f}%")

if __name__ == "__main__":
    main()
