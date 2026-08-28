#!/usr/bin/env python3
"""
SFT data generation pipeline for DLLM-Searcher.
Uses GPT-4o as teacher + Serper Google Search to generate agentic trajectories.

Pipeline:
  1. Load HotpotQA / 2WikiMultiHopQA / Musique training sets (2048 each)
  2. Rollout with GPT-4o + real search
  3. Filter: LLM judge (answer correct) + format valid
  4. Save as SFT training format {prompt, response}

Usage:
  python generate_sft_data.py --output_dir ./sft_output --max_workers 20
  python generate_sft_data.py --output_dir ./sft_output --datasets hotpotqa --samples_per_dataset 500
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
def _load_config():
    for path in [
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

_cfg = _load_config()

# GPT-4o for teacher rollout AND LLM judge
TEACHER_API_KEY  = os.getenv("JUDGE_API_KEY")  or _cfg.get("judge_api_key")  or ""
TEACHER_API_BASE = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base") or "https://api.openai.com/v1"
TEACHER_MODEL    = "gpt-4o"

# Serper for search
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key") or ""
SEARCH_URL = "https://google.serper.dev/search"

MAX_TURNS   = 8
MAX_WORKERS = 20

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins."""

USER_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }
    },
    "required": ["query"]
  }
}
</tools>

The assistant starts with one or more cycles of (thinking -> tool call -> tool response), and ends with the answer.

Format:
<think>
reasoning here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query 1", "query 2"]}}
</tool_call>
<tool_response>
search results here
</tool_response>
<think>
more reasoning
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["another query"]}}
</tool_call>
<tool_response>
search results here
</tool_response>
<|box_start|>
final answer here
<|box_end|>

The answer MUST be placed between <|box_start|> and <|box_end|>. Use <tool_call> for searches.

User: """

JUDGE_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.
Respond with ONLY True if the prediction is correct and ONLY False otherwise.
Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}"""

# ── Search ────────────────────────────────────────────────────────────────────
def google_search(query: str) -> str:
    headers = {"X-API-KEY": GOOGLE_SEARCH_KEY, "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(SEARCH_URL, headers=headers,
                                 json={"q": query, "num": 10}, timeout=15)
            if resp.status_code != 200:
                return f"Search error {resp.status_code}: {resp.text[:100]}"
            data = resp.json()
            if "organic" not in data:
                return f"No results for '{query}'."
            snippets = []
            for i, page in enumerate(data["organic"], 1):
                date = f"\nDate: {page['date']}" if "date" in page else ""
                snippet = f"\n{page.get('snippet', '')}"
                snippets.append(f"{i}. [{page['title']}]({page['link']}){date}{snippet}")
            return f"Google search for '{query}' found {len(snippets)} results:\n\n" + "\n\n".join(snippets)
        except Exception as e:
            if attempt == 2:
                return f"Search failed: {e}"
            time.sleep(1)
    return "Search failed."


def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n=======\n\n".join(results)


# ── GPT-4o call ───────────────────────────────────────────────────────────────
_teacher_client = None

def get_teacher_client():
    global _teacher_client
    if _teacher_client is None:
        _teacher_client = OpenAI(api_key=TEACHER_API_KEY, base_url=TEACHER_API_BASE)
    return _teacher_client


def call_gpt4o(messages: list, max_retries: int = 3) -> str:
    client = get_teacher_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                temperature=1.0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"GPT-4o error: {e}"
            time.sleep(2 ** attempt)
    return "GPT-4o failed."


# ── Parse helpers ─────────────────────────────────────────────────────────────
def parse_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        return obj.get("arguments", {}).get("query", [])
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass
    return None


def extract_answer(text: str):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def has_valid_format(messages: list) -> bool:
    """Check: has at least one real tool_call and a final answer."""
    full = " ".join(m["content"] for m in messages if m["role"] == "assistant")
    has_tool = bool(re.search(r"<tool_call>.*?</tool_call>", full, re.DOTALL))
    has_answer = bool(re.search(r"<\|box_start\|>.*?<\|box_end\|>", full, re.DOTALL))
    return has_tool and has_answer


def has_real_search_results(messages: list) -> bool:
    """Check: at least one tool_response contains real search results (not an error)."""
    error_patterns = [
        "Search error", "Not enough credits", "Rate limit",
        "Search failed", "No results", "credits",
    ]
    tool_responses = [
        m["content"] for m in messages
        if m["role"] == "user" and "<tool_response>" in m["content"]
    ]
    if not tool_responses:
        return False
    for resp in tool_responses:
        content = resp.replace("<tool_response>", "").replace("</tool_response>", "")
        if not any(pat in content for pat in error_patterns):
            return True  # 至少一条搜索成功
    return False


# ── Rollout ───────────────────────────────────────────────────────────────────
def rollout_one(question: str, answer: str) -> dict:
    """Run one ReAct trajectory for a question using GPT-4o + search."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    prediction = None

    for turn in range(MAX_TURNS):
        content = call_gpt4o(messages)

        # Strip anything after <tool_response> if model hallucinates it
        if "<tool_response>" in content:
            content = content[:content.find("<tool_response>")]

        messages.append({"role": "assistant", "content": content.strip()})

        # Check for final answer
        prediction = extract_answer(content)
        if prediction:
            break

        # Check for tool call
        queries = parse_tool_call(content)
        if queries:
            search_result = run_search(queries)
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{search_result}\n</tool_response>"
            })
        elif turn >= 4:
            # Force answer
            messages.append({
                "role": "user",
                "content": "Based on your research above, provide your final answer "
                           "in the format <|box_start|>answer<|box_end|>."
            })

    return {
        "question": question,
        "answer": answer,
        "messages": messages,
        "prediction": prediction,
        "termination": "answer" if prediction else "max_turns",
    }


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def llm_judge(question: str, reference: str, prediction: str) -> bool:
    if not prediction or not prediction.strip():
        return False
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference,
        prediction=prediction or "(no answer)"
    )
    for attempt in range(2):
        try:
            result = call_gpt4o([{"role": "user", "content": prompt}])
            return result.strip().lower() == "true"
        except Exception:
            time.sleep(1)
    return False


# ── Convert to SFT training format ───────────────────────────────────────────
def to_sft_format(result: dict, tokenizer_apply=None) -> dict:
    """
    Convert a rollout result to {prompt, response} SFT training format.
    prompt  = system + user turn (chat template applied)
    response = full assistant trajectory (all think/tool_call/tool_response/answer)
    """
    messages = result["messages"]
    # Build prompt: everything up to (not including) first assistant turn
    prompt_messages = [m for m in messages if m["role"] in ("system", "user")][:2]

    # Build response: concatenate all assistant turns + tool responses in order
    response_parts = []
    for m in messages[2:]:  # skip system + first user
        if m["role"] == "assistant":
            response_parts.append(m["content"])
        elif m["role"] == "user" and "<tool_response>" in m["content"]:
            response_parts.append(m["content"])
        # skip the force-answer user messages

    # Build prompt string using chat template tokens manually
    system = prompt_messages[0]["content"]
    user   = prompt_messages[1]["content"]
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response = "\n".join(response_parts)

    return {"prompt": prompt, "response": response}


# ── Dataset loading ───────────────────────────────────────────────────────────
# data/hotpot.jsonl 是 HotpotQA validation 集，仅供参考，不再用于训练。
# 训练数据统一从 HuggingFace train split 加载；测试数据从 validation split 加载。

def load_dataset_samples(dataset_name: str, n_samples: int, split: str = "train", seed: int = 42) -> list:
    """
    Load n_samples questions from the specified split of a dataset.
    split: "train" for SFT data generation, "validation" for eval set creation.
    """
    print(f"Loading {dataset_name} split={split} ({n_samples} samples)...")
    import random
    random.seed(seed)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
        items = [{"question": x["question"], "answer": x["answer"]} for x in ds]

    elif dataset_name == "2wikimultihopqa":
        ds = load_dataset("xanhho/2WikiMultiHopQA", split=split, trust_remote_code=True)
        items = [{"question": x["question"], "answer": x["answer"]} for x in ds]

    elif dataset_name == "musique":
        loaded = False
        for hf_name in ["drt-zl/musique", "allenai/musique", "musique"]:
            try:
                ds = load_dataset(hf_name, split=split, trust_remote_code=True)
                items = [{"question": x["question"], "answer": x["answer"]} for x in ds]
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            print(f"WARNING: Could not load musique from HuggingFace, skipping.")
            return []

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if n_samples >= len(items):
        sampled = items
    else:
        sampled = random.sample(items, n_samples)
    print(f"  Loaded {len(sampled)} samples from {dataset_name}/{split}")
    return sampled


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SFT data generation / eval set creation")
    parser.add_argument("--mode",        default="sft",
                        choices=["sft", "eval", "merge"],
                        help="sft: GPT-4o rollout for training data; "
                             "eval: extract val split as test set (no rollout); "
                             "merge: combine shard files and apply filter")
    parser.add_argument("--output_dir",  default="./sft_output",
                        help="Directory for output files")
    parser.add_argument("--datasets",    default="hotpotqa,2wikimultihopqa",
                        help="Comma-separated dataset names")
    parser.add_argument("--samples_per_dataset", type=int, default=2048,
                        help="Questions per dataset for SFT rollout")
    parser.add_argument("--eval_samples_per_dataset", type=int, default=500,
                        help="Questions per dataset for eval set (--mode eval)")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help="Parallel rollout workers")
    parser.add_argument("--skip_judge",  action="store_true",
                        help="Skip LLM judge filter")
    parser.add_argument("--rank",        type=int, default=0,
                        help="Process rank for multi-process sharding (0-indexed)")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="Total number of parallel processes")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for dataset sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE: eval — extract question/answer from validation splits, no rollout
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "eval":
        print("=== MODE: eval (building test set from validation splits) ===")
        all_items = []
        for name in args.datasets.split(","):
            name = name.strip()
            items = load_dataset_samples(name, args.eval_samples_per_dataset,
                                         split="validation", seed=args.seed)
            for item in items:
                item["source"] = name
            all_items.extend(items)

        out_file = output_dir / "eval_test_set.jsonl"
        with open(out_file, "w") as f:
            for item in all_items:
                f.write(json.dumps({"question": item["question"],
                                    "answer": item["answer"],
                                    "source": item.get("source", "")},
                                   ensure_ascii=False) + "\n")
        print(f"\n{'='*50}")
        print(f"Test set saved → {out_file}")
        print(f"Total: {len(all_items)} questions "
              f"({args.eval_samples_per_dataset} per dataset)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE: merge — combine shards, filter, convert to SFT format
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "merge":
        print("=== MODE: merge (combining shard files and filtering) ===")
        shard_files = sorted(output_dir.glob("raw_rollouts_shard*.jsonl"))
        if not shard_files:
            print("No shard files found (raw_rollouts_shard*.jsonl). Nothing to merge.")
            return
        print(f"Found {len(shard_files)} shard files")

        all_results = []
        seen_questions = set()
        for sf in shard_files:
            with open(sf) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        q = d.get("question", "")
                        if q and q not in seen_questions:
                            seen_questions.add(q)
                            all_results.append(d)
                    except Exception:
                        pass
        print(f"Total unique rollouts: {len(all_results)}")

        assert TEACHER_API_KEY, "Set judge_api_key in config.json or JUDGE_API_KEY env var"
        _run_filter_and_save(all_results, output_dir, timestamp, args)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE: sft — GPT-4o rollout (default, supports multi-process sharding)
    # ══════════════════════════════════════════════════════════════════════════
    assert TEACHER_API_KEY, "Set judge_api_key in config.json or JUDGE_API_KEY env var"
    assert GOOGLE_SEARCH_KEY, "Set google_search_key in config.json or GOOGLE_SEARCH_KEY env var"

    print(f"=== MODE: sft (rank={args.rank}/{args.world_size}) ===")

    # ── Step 1: Load datasets (train split) ──────────────────────────────────
    all_items = []
    for name in args.datasets.split(","):
        name = name.strip()
        items = load_dataset_samples(name, args.samples_per_dataset,
                                     split="train", seed=args.seed)
        all_items.extend(items)
    print(f"\nTotal questions (all ranks): {len(all_items)}")

    # Shard by rank
    shard_items = all_items[args.rank::args.world_size]
    print(f"This rank handles: {len(shard_items)} questions")

    # Output file per rank (rank 0 with world_size=1 → single file)
    if args.world_size > 1:
        raw_file = output_dir / f"raw_rollouts_shard{args.rank:02d}_of{args.world_size:02d}_{timestamp}.jsonl"
    else:
        raw_file = output_dir / f"raw_rollouts_{timestamp}.jsonl"

    # Resume: skip already-processed questions
    processed = set()
    if raw_file.exists():
        with open(raw_file) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["question"])
                except Exception:
                    pass
        print(f"Resuming: {len(processed)} already done")

    todo = [x for x in shard_items if x["question"] not in processed]
    print(f"Remaining: {len(todo)}")

    # ── Step 2: Rollout ───────────────────────────────────────────────────────
    write_lock = threading.Lock()

    def process_item(item):
        return rollout_one(item["question"], item["answer"])

    with open(raw_file, "a") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(process_item, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Rollout rank{args.rank}"):
                try:
                    result = future.result()
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                except Exception as e:
                    item = futures[future]
                    print(f"Error on '{item['question'][:50]}': {e}")

    print(f"\nRaw rollouts saved → {raw_file}")

    # If multi-process, skip filter (run merge mode afterwards to combine all shards)
    if args.world_size > 1:
        print(f"\nRank {args.rank} done. After all ranks finish, run:")
        print(f"  python generate_sft_data.py --mode merge --output_dir {output_dir}")
        return

    # Single-process: filter and save immediately
    all_results = []
    with open(raw_file) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass
    _run_filter_and_save(all_results, output_dir, timestamp, args)


def _run_filter_and_save(all_results, output_dir, timestamp, args):
    """Filter rollouts and save final SFT training data."""
    print(f"\nFiltering {len(all_results)} rollouts...")

    has_answer  = [r for r in all_results if r.get("prediction")]
    print(f"Has answer:          {len(has_answer)}")

    valid_format = [r for r in has_answer if has_valid_format(r["messages"])]
    print(f"Valid format:        {len(valid_format)}")

    real_search  = [r for r in valid_format if has_real_search_results(r["messages"])]
    print(f"Has real search:     {len(real_search)}")
    valid_format = real_search

    if args.skip_judge:
        correct = valid_format
        print(f"LLM judge skipped, using all {len(correct)} valid-format samples")
    else:
        print(f"Running LLM judge on {len(valid_format)} samples...")
        correct = []
        judge_lock = threading.Lock()

        def judge_item(r):
            ok = llm_judge(r["question"], r["answer"], r["prediction"])
            return r, ok

        with ThreadPoolExecutor(max_workers=min(getattr(args, 'max_workers', 20), 50)) as ex:
            futures = [ex.submit(judge_item, r) for r in valid_format]
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
                r, ok = future.result()
                if ok:
                    with judge_lock:
                        correct.append(r)
        print(f"Correct (LLM judge): {len(correct)}")

    sft_data   = [to_sft_format(r) for r in correct]
    final_file = output_dir / f"sft_train_{timestamp}.json"
    with open(final_file, "w") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"SFT training data saved → {final_file}")
    print(f"Total samples: {len(sft_data)}")


if __name__ == "__main__":
    main()
