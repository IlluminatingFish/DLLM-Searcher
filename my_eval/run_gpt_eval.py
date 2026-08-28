#!/usr/bin/env python3
"""
Run GPT as a baseline model on eval datasets.
Sends each question directly to GPT (no agentic search tools),
outputs a JSONL with {question, answer, prediction} for cal_acc.py.

Usage:
  python my_eval/run_gpt_eval.py \
    --input  data/dpo_train_as_eval_200.jsonl \
    --output output/gpt_eval/dpo_train_200.jsonl \
    --model  gpt-4o

  python my_eval/run_gpt_eval.py \
    --input  data/sft_train_as_eval_200.jsonl \
    --output output/gpt_eval/sft_train_200.jsonl \
    --model  gpt-4o
"""

import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import rich
from openai import OpenAI
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
def _load_config():
    cfg = {}
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
    return cfg

_cfg = _load_config()
API_KEY      = os.getenv("JUDGE_API_KEY") or _cfg.get("judge_api_key") or \
               os.getenv("OPENAI_API_KEY") or _cfg.get("openai_api_key") or "YOUR_KEY"
API_BASE_URL = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or \
               "https://api.openai.com/v1"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the user's question concisely and accurately. "
    "Give a direct, factual answer in 1-3 sentences."
)


# ── GPT call ──────────────────────────────────────────────────────────────────
def ask_gpt(client, model, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                temperature=0,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                rich.print(f"[red]GPT failed for: {question[:60]}... → {e}[/red]")
                return None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,  help="Input eval JSONL (question + answer)")
    parser.add_argument("--output", required=True,  help="Output JSONL with predictions")
    parser.add_argument("--model",  default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--workers", type=int, default=16, help="Parallel API workers")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0=all)")
    args = parser.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    rich.print(f"[bold]GPT Eval[/bold]  model={args.model}  samples={len(samples)}")
    rich.print(f"API: {API_BASE_URL}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = [None] * len(samples)

    def process(idx_sample):
        idx, sample = idx_sample
        q = sample["question"]
        pred = ask_gpt(client, args.model, q)
        return idx, {
            "question":   q,
            "answer":     sample["answer"],
            "prediction": pred,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, (i, s)): i
                   for i, s in enumerate(samples)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"GPT ({args.model})"):
            idx, result = future.result()
            results[idx] = result

    with open(args.output, "w") as f:
        answered = 0
        for r in results:
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["prediction"]:
                answered += 1

    rich.print(f"\n[green]Done![/green] Answered: {answered}/{len(samples)}")
    rich.print(f"Output → {args.output}")
    rich.print(f"\nRun accuracy: python my_eval/cal_acc.py --data {args.output}")


if __name__ == "__main__":
    main()
