#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import rich
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
# LLM Judge 可单独配置；未设置则用 openai_*
API_KEY = os.getenv("JUDGE_API_KEY") or _cfg.get("judge_api_key") or os.getenv("OPENAI_API_KEY") or _cfg.get("openai_api_key") or "YOUR_API_KEY"
API_BASE_URL = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or os.getenv("OPENAI_API_BASE") or _cfg.get("openai_api_base") or "YOUR_API_BASE_URL"
MODEL_NAME = os.getenv("JUDGE_MODEL") or _cfg.get("judge_model") or "gpt-4o"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ==================== Answer Normalization ====================
def normalize_answer(s):
    """Normalize answer: lowercase, remove punctuation, remove articles, fix whitespace"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    """Convert boolean string to yes/no"""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def cover_exact_match_score_1(prediction, ground_truth):
    """CEM-1: All ground truth words appear in prediction (order-independent)"""
    if prediction is None or not str(prediction).strip():
        return False
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    return all(ground in pre_list for ground in ground_list)


# ==================== LLM Judge ====================
def llm_judge_single(question, reference, prediction):
    """Use LLM to judge if the predicted answer is correct"""
    pred_str = (prediction if prediction is not None else "") or "(no answer)"
    prompt = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. 
    The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. 
    Respond with ONLY True if the prediction is correct and ONLY False otherwise.
    Question: {question}
    Golden Answer: {reference}
    Predicted Answer: {prediction}
    '''
    
    llm_correct = False
    try:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": prompt.format(
                            question=question,
                            reference=reference,
                            prediction=pred_str
                        )
                    }],
                    stream=False,
                )
                result = response.choices[0].message.content.strip()
                llm_correct = (result.lower() == "true")
                break
            except Exception as e:
                if attempt == 1:
                    rich.print(f"Judge failed (question: {question[:30]}...): {str(e)}")
                time.sleep(1)
    except Exception as e:
        rich.print(f"Judge error (question: {question[:30]}...): {str(e)}")
    
    return llm_correct


def token_f1(prediction, reference):
    """Token-level F1 between prediction and reference."""
    if not prediction or not reference:
        return 0.0
    pred_tokens = normalize_answer(bool_mapping(str(prediction))).split()
    ref_tokens  = normalize_answer(bool_mapping(str(reference))).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(data_path):
    """Evaluate data file.

    Metrics:
    - CEM-1        : uses `short_answer` field when available, else `answer`
    - LLM Judge    : always uses `answer` (long reference is better for judge)
    - Token F1     : uses `short_answer` when available, else `answer`
    - Answer rate  : fraction of non-empty predictions
    - F1 > 0.5 / > 0.8 thresholds
    """
    query_info = {}

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data["question"]
                # short_answer: CEM-1 / token-F1 / LLM Judge 统一用短GT
                short_gt  = data.get("short_answer") or data.get("answer") or ""
                query_info[question] = {
                    "short_gt":   short_gt,
                    "prediction": data.get("prediction") or "",
                    "source":     data.get("source", "unknown"),
                }
            except json.JSONDecodeError:
                print(f"Error parsing line: {line[:100]}...")
                continue

    rich.print(f"Valid data entries: {len(query_info)}")
    key_mask = ("*" + API_KEY[-4:]) if API_KEY and len(API_KEY) > 4 and API_KEY != "YOUR_API_KEY" else "(unset)"
    rich.print(f"[dim]LLM Judge: {API_BASE_URL} | model={MODEL_NAME} | key={key_mask}[/dim]")

    cem1_count  = 0
    llm_count   = 0
    f1_scores   = []
    answer_count = 0

    items = list(query_info.items())

    def process_item(item):
        question, info = item
        pred = info["prediction"]
        has_answer = bool(pred and str(pred).strip())
        f1  = token_f1(pred, info["short_gt"])
        cem1 = cover_exact_match_score_1(pred, info["short_gt"])
        llm  = llm_judge_single(question, info["short_gt"], pred)
        return cem1, llm, f1, has_answer

    rich.print("Starting evaluation...")
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            cem1, llm, f1, has_ans = future.result()
            if cem1:      cem1_count  += 1
            if llm:       llm_count   += 1
            if has_ans:   answer_count += 1
            f1_scores.append(f1)

    total = len(query_info)
    mean_f1  = sum(f1_scores) / total if total else 0
    f1_gt05  = sum(1 for s in f1_scores if s > 0.5)
    f1_gt08  = sum(1 for s in f1_scores if s > 0.8)

    rich.print(f"\n{'='*50}")
    rich.print(f"Total questions     : {total}")
    rich.print(f"Answer rate         : {answer_count}/{total} = {answer_count/total*100:.1f}%")
    rich.print(f"CEM-1  (short GT)   : {cem1_count}/{total} = {cem1_count/total*100:.2f}%")
    rich.print(f"Token F1 mean       : {mean_f1*100:.2f}%")
    rich.print(f"Token F1 > 0.5      : {f1_gt05}/{total} = {f1_gt05/total*100:.1f}%")
    rich.print(f"Token F1 > 0.8      : {f1_gt08}/{total} = {f1_gt08/total*100:.1f}%")
    rich.print(f"LLM Judge           : {llm_count}/{total} = {llm_count/total*100:.2f}%")

    # 按数据集来源分组汇报
    sources = {}
    for q, info in query_info.items():
        s = info["source"]
        if s not in sources:
            sources[s] = []
        sources[s].append(info)
    if len(sources) > 1:
        rich.print(f"\n--- By source ---")
        for src, infos in sorted(sources.items()):
            src_preds = [i["prediction"] for i in infos]
            src_ans = sum(1 for p in src_preds if p and str(p).strip())
            rich.print(f"  {src}: {len(infos)}题  answer_rate={src_ans/len(infos)*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default=None, help="Path to rollout output jsonl")
    args = parser.parse_args()
    data_path = args.data or os.getenv("EVAL_DATA") or "YOUR_DATA_PATH.jsonl"
    if data_path == "YOUR_DATA_PATH.jsonl":
        rich.print("[yellow]请用 --data 指定 rollout 输出的 jsonl 路径[/yellow]")
        rich.print("例: python cal_acc.py --data ../Dataroller/base/Llama-3.1-8B_sglang/example/iter1.jsonl")
        exit(1)
    evaluate(data_path)