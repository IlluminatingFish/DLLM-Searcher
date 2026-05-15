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


def evaluate(data_path):
    """Evaluate data file, compute CEM-1 and LLM Judge scores"""
    query_info = {}
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data["question"]
                query_info[question] = {
                    "reference": data["answer"],
                    "prediction": data.get("prediction") or "",  # JSON null -> ""
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}...")
                continue
    
    rich.print(f"Valid data entries: {len(query_info)}")
    key_mask = ("*" + API_KEY[-4:]) if API_KEY and len(API_KEY) > 4 and API_KEY != "YOUR_API_KEY" else "(unset)"
    rich.print(f"[dim]LLM Judge: {API_BASE_URL} | model={MODEL_NAME} | key={key_mask}[/dim]")

    cem1_count = 0
    llm_count = 0
    
    items = list(query_info.items())
    
    def process_item(item):
        question, info = item
        reference = info["reference"]
        prediction = info["prediction"]
        
        cem1 = cover_exact_match_score_1(prediction, reference)
        llm_correct = llm_judge_single(question, reference, prediction)
        
        return cem1, llm_correct
    
    rich.print("Starting evaluation...")
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            cem1, llm_correct = future.result()
            if cem1:
                cem1_count += 1
            if llm_correct:
                llm_count += 1
    
    total = len(query_info)
    rich.print(f"\n{'='*50}")
    rich.print(f"CEM-1 (word coverage): {cem1_count}/{total} = {cem1_count/total*100:.2f}%")
    rich.print(f"LLM as Judge        : {llm_count}/{total} = {llm_count/total*100:.2f}%")


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