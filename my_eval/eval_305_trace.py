#!/usr/bin/env python3
"""对 305 道错题的 trace 结果做 CEM-1 + LLM Judge（含详细分析），输出 jsonl"""
import json, os, re, string, time, argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── API 配置 ───────────────────────────────────────────────────────────────────
def _load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        return json.load(open(cfg_path))
    except:
        return {}

_cfg = _load_config()
API_KEY      = os.getenv("JUDGE_API_KEY") or _cfg.get("judge_api_key") or _cfg.get("openai_api_key", "")
API_BASE_URL = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or _cfg.get("openai_api_base", "")
MODEL_NAME   = os.getenv("JUDGE_MODEL")    or _cfg.get("judge_model", "gpt-4o")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── CEM-1 ──────────────────────────────────────────────────────────────────────
def normalize(s):
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    excl = set(string.punctuation + "".join(["'", "'", "´", "`"]))
    s = "".join(c if c not in excl else " " for c in s)
    return " ".join(s.split())

def cem1(pred, gt):
    if not pred or not str(pred).strip():
        return False
    pred_words = normalize(str(pred)).split()
    gt_words   = normalize(str(gt)).split()
    return all(w in pred_words for w in gt_words)

# ── LLM Judge（返回 verdict + 分析）─────────────────────────────────────────────
JUDGE_PROMPT = """\
You are a strict answer evaluator. Given a Question, a Gold Answer (short reference), and a Predicted Answer, judge whether the prediction is correct.

Rules:
- The prediction is CORRECT if it contains the key information of the gold answer, even if worded differently.
- The prediction is WRONG if it gives a different entity, contradicts the gold answer, or is empty.
- Respond in this exact JSON format (no markdown, no extra text):
{{"verdict": "CORRECT" or "WRONG", "reason": "<1-2 sentences explaining your judgment>"}}

Question: {question}
Gold Answer: {gold}
Predicted Answer: {pred}"""

def llm_judge(question, gold, pred):
    pred_str = (str(pred) if pred else "") or "(no answer)"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, gold=gold, pred=pred_str[:500]
                )}],
                stream=False,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            # 去掉可能的 markdown 代码块
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            obj = json.loads(raw)
            verdict = obj.get("verdict", "WRONG").upper()
            reason  = obj.get("reason", "")
            return verdict == "CORRECT", reason
        except json.JSONDecodeError:
            # 尝试从原始文本里提取
            correct = "correct" in raw.lower() and "wrong" not in raw.lower()
            return correct, raw[:200]
        except Exception as e:
            if attempt == 2:
                return False, f"Judge error: {e}"
            time.sleep(1)
    return False, "Judge failed"

# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", default="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/bs128/trace/305")
    parser.add_argument("--output",    default="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/bs128/trace/305_eval.jsonl")
    parser.add_argument("--workers",   type=int, default=60)
    args = parser.parse_args()

    # 读取 305 题
    all_data = []
    for r in range(7):
        f = f"{args.trace_dir}/rank{r}.jsonl"
        for line in open(f):
            d = json.loads(line)
            all_data.append(d)
    print(f"共 {len(all_data)} 题")

    def process(d):
        q    = d["question"]
        gt   = d["gt"]
        pred = d.get("prediction", "")
        c1   = cem1(pred, gt)
        lj_correct, lj_reason = llm_judge(q, gt, pred)
        return {
            "question":        q,
            "gt":              gt,
            "prediction":      str(pred) if pred else "",
            "cat":             d.get("cat", ""),
            "termination_reason": d.get("termination_reason", ""),
            "regen_correct_orig": d.get("regen_correct", False),  # trace 脚本自己算的
            "cem1":            c1,
            "llm_correct":     lj_correct,
            "llm_reason":      lj_reason,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, d): d for d in all_data}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Error: {e}")

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    n = len(results)
    c1_n  = sum(1 for r in results if r["cem1"])
    lj_n  = sum(1 for r in results if r["llm_correct"])
    ans_n = sum(1 for r in results if r["prediction"].strip())
    print(f"\n{'='*50}")
    print(f"总题数      : {n}")
    print(f"有答案      : {ans_n}/{n} = {ans_n/n*100:.1f}%")
    print(f"CEM-1       : {c1_n}/{n} = {c1_n/n*100:.2f}%")
    print(f"LLM Judge   : {lj_n}/{n} = {lj_n/n*100:.2f}%")
    print(f"结果已写入  : {args.output}")

if __name__ == "__main__":
    main()
