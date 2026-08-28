#!/usr/bin/env python3
"""
DLLM-Searcher 论文标准评分脚本：ACC_R + ACC_L

严格按照论文 Section 5.1.2 的定义：
  - ACC_R: 标准答案（normalize后）是否被包含在预测输出中（substring containment）
           参考 R1-Searcher [28] / Search-R1 [32] 的实现
  - ACC_L: LLM-as-Judge，判断预测语义是否正确（论文用 Doubao，我们用 GPT-4o）
  - normalize: lowercase, remove punctuation, remove articles (a/an/the), fix whitespace
  - prediction 取 <|box_start|>...<|box_end|> 中的内容，若无 box 取 prediction 字段

注意：论文明确指出 EM 不适合（输出通常是长段落），因此不用 EM/F1。

用法:
  python my_eval/score_paper_metrics.py --data path/to/pred.jsonl
  python my_eval/score_paper_metrics.py --data path/to/pred.jsonl --no_llm_judge
"""

import argparse
import json
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────
def _load_config():
    for p in [
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
    ]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

_cfg = _load_config()

# 默认从 config.json / 环境变量读（GPT-4o）
JUDGE_API_KEY  = os.getenv("JUDGE_API_KEY")  or _cfg.get("judge_api_key")  or ""
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or "https://api.openai.com/v1"
JUDGE_MODEL    = os.getenv("JUDGE_MODEL")    or _cfg.get("judge_model")    or "gpt-4o"

DOUBAO_API_KEY  = os.getenv("DOUBAO_API_KEY")  or _cfg.get("doubao_api_key")  or ""
DOUBAO_API_BASE = os.getenv("DOUBAO_API_BASE") or _cfg.get("doubao_api_base") or "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL    = os.getenv("DOUBAO_MODEL")    or _cfg.get("doubao_model")    or "doubao-seed-1-6-250615"


def select_judge(judge_name: str):
    """根据 --judge 参数切换 API 配置（修改模块级全局变量）。"""
    global JUDGE_API_KEY, JUDGE_API_BASE, JUDGE_MODEL, _judge_client
    if judge_name == "doubao":
        JUDGE_API_KEY  = DOUBAO_API_KEY
        JUDGE_API_BASE = DOUBAO_API_BASE
        JUDGE_MODEL    = DOUBAO_MODEL
        _judge_client  = None  # 重置 client
        print(f"[Judge] 使用 Doubao  model={JUDGE_MODEL}  base={JUDGE_API_BASE}")
    else:
        print(f"[Judge] 使用 GPT-4o  model={JUDGE_MODEL}  base={JUDGE_API_BASE}")


# ── Normalization ─────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    """标准 normalize：lowercase, 去标点, 去冠词, 合并空格"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def acc_r(prediction: str, ground_truth: str) -> int:
    """
    ACC_R（论文 Section 5.1.2）：
    normalize 后，判断标准答案是否被包含在预测输出中（substring containment）。
    与 R1-Searcher [28] / Search-R1 [32] 的实现一致。
    """
    if not prediction:
        return 0
    norm_pred = normalize_answer(prediction)
    norm_gt   = normalize_answer(ground_truth)
    return int(norm_gt in norm_pred)


def extract_prediction(data: dict) -> str:
    """从 box 或 prediction 字段提取最终答案。"""
    # 优先从 response/messages 里找 box
    for field in ["response", "full_response"]:
        text = data.get(field, "")
        if text:
            m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
            if m:
                return m.group(1).strip()
    # 其次直接用 prediction 字段
    pred = data.get("prediction") or ""
    if pred:
        # prediction 字段本身可能是 box 内容（run_llada_eval.py 已提取）
        return str(pred).strip()
    return ""


# ── LLM Judge ─────────────────────────────────────────────────────────────────
_judge_client = None

def get_judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_API_BASE)
    return _judge_client


JUDGE_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.
Respond with ONLY True if the prediction is correct and ONLY False otherwise.
Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}"""


def llm_judge(question: str, reference: str, prediction: str) -> bool:
    if not prediction:
        return False
    client = get_judge_client()
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference,
        prediction=prediction or "(no answer)"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip().lower() == "true"
        except Exception as e:
            if attempt == 2:
                print(f"  [judge error] {e}")
            time.sleep(2 ** attempt)
    return False


# ── Main ───────────────────────────────────────────────────────────────────────
def evaluate(data_path: str, no_llm_judge: bool = False):
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    print(f"共 {len(records)} 条记录  |  文件: {data_path}")
    if not records:
        return

    accr_scores, judge_scores = [], []
    no_pred = 0

    # ACC_R（确定性，直接计算）
    for d in records:
        pred = extract_prediction(d)
        gt   = str(d.get("answer", "")).strip()
        if not pred:
            no_pred += 1
        accr_scores.append(acc_r(pred, gt))
        judge_scores.append(None)  # 占位，后面 LLM Judge 填入

    # ACC_L：LLM Judge（并行）
    if not no_llm_judge:
        print(f"运行 LLM Judge（ACC_L，model={JUDGE_MODEL}）...")
        def judge_item(idx):
            d    = records[idx]
            pred = extract_prediction(d)
            gt   = str(d.get("answer", "")).strip()
            return idx, llm_judge(d.get("question", ""), gt, pred)

        with ThreadPoolExecutor(max_workers=50) as ex:
            futures = {ex.submit(judge_item, i): i for i in range(len(records))}
            for fut in tqdm(as_completed(futures), total=len(records)):
                idx, ok = fut.result()
                judge_scores[idx] = ok
    else:
        judge_scores = [None] * len(records)

    n    = len(records)
    ar   = sum(accr_scores) / n * 100
    al   = sum(1 for s in judge_scores if s) / n * 100 if not no_llm_judge else None

    print(f"\n{'='*55}")
    print(f"  样本数:                    {n}  （无答案: {no_pred}）")
    print(f"  ACC_R (containment):       {ar:.1f}%")
    if al is not None:
        print(f"  ACC_L (LLM Judge):         {al:.1f}%")
    print(f"{'='*55}")

    # 保存详细结果
    out_path = Path(data_path).with_suffix(".scored.jsonl")
    with open(out_path, "w") as f:
        for d, ar_s, al_s in zip(records, accr_scores, judge_scores):
            pred = extract_prediction(d)
            f.write(json.dumps({
                "question":  d.get("question"),
                "answer":    d.get("answer"),
                "prediction": pred,
                "acc_r":     ar_s,
                "acc_l":     al_s,
            }, ensure_ascii=False) + "\n")
    print(f"  详细结果 → {out_path}")
    return {"acc_r": ar, "acc_l": al, "n": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="pred.jsonl 路径")
    parser.add_argument("--no_llm_judge", action="store_true", help="跳过 LLM Judge")
    parser.add_argument("--judge", default="gpt", choices=["gpt", "doubao"],
                        help="LLM judge 选择：gpt（GPT-4o）或 doubao（Doubao-seed）")
    args = parser.parse_args()
    select_judge(args.judge)
    evaluate(args.data, no_llm_judge=args.no_llm_judge)
