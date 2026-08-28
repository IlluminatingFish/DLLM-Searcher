#!/usr/bin/env python3
"""
LLaDA baseline 评测脚本 —— 移植 toolcall_pre_rl 原理到 LLaDA tokenizer。

原版原理（scheduler.py toolcall_pre_rl）：
  step 0：强制写入 <tool_call> 和 </tool_call> 的结构框架
  step >0：tool_call 内容区域 confidence +0.5，优先揭示

LLaDA 适配（<tool_call> 是4个token，而非单个特殊token）：
  step 0：强制 positions 0-3  = <tool_call>  [27, 28347, 32074, 29]
           强制 positions 60-63 = </tool_call> [1263, 28347, 32074, 29]
  step >0：confidence[4:60] += 0.5，优先揭示中间的 JSON 内容

用法:
  python my_eval/run_baseline_llada_eval.py \
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
    --output dLLM_trainer/VRPO/output/llada_eval/baseline_ckpt6_100.jsonl \
    --max_samples 100
"""

import argparse
import json
import os
import re
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 推理超参 ──────────────────────────────────────────────────────────────────
BLOCK_SIZE = 64
NUM_STEPS  = 64
MAX_BLOCKS = 16
MAX_TURNS  = 5

# LLaDA tokenizer 里的 <tool_call> / </tool_call> token 序列（4个token各）
TOOL_CALL_TOKENS     = [27, 28347, 32074, 29]   # <tool_call>
TOOL_CALL_END_TOKENS = [1263, 28347, 32074, 29]  # </tool_call>
CONTENT_START = 4    # tool_call 内容起始位置（跳过 <tool_call> 的4个token）
CONTENT_END   = 60   # tool_call 内容结束位置（为 </tool_call> 留4个位置）

DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/output/llada_eval/baseline_ckpt6_100.jsonl"
)


# ── Search API ────────────────────────────────────────────────────────────────
def _load_config():
    for path in [
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"


def google_search(query: str) -> str:
    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(SEARCH_URL, headers=headers,
                             json={"q": query, "num": 5}, timeout=10)
        if resp.status_code != 200:
            return f"Search error {resp.status_code}"
        data = resp.json()
        if "organic" not in data:
            return f"No results for '{query}'."
        lines = [
            f"{i}. [{p['title']}] {p.get('snippet', '')}"
            for i, p in enumerate(data["organic"][:5], 1)
        ]
        return f"Results for '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)


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
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question.

Example response:
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query string 1", "query string 2"]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think>
thinking process here
</think>
<|box_start|>
answer here
<|box_end|>
The assistant must strictly abide by the above format.
User: """


# ── toolcall_pre_rl 原理的 LLaDA 适配版 ──────────────────────────────────────
@torch.no_grad()
def denoise_block_toolcall_priority(model, input_ids, block_start, block_end,
                                    mask_id, num_steps, is_first_block=False):
    """
    对第一个生成 block 应用 toolcall_pre_rl 原理：
      step 0：强制写入 <tool_call>(0-3) 和 </tool_call>(60-63) 的结构框架
      step >0：中间内容区(4-59) confidence +0.5，优先揭示 JSON body

    后续 block（已有 tool_call 结构后）使用普通置信度排序。
    """
    block_len = block_end - block_start

    for step in range(num_steps):
        logits = model(input_ids=input_ids).logits
        block_logits = logits[0, block_start:block_end]

        probs      = torch.softmax(block_logits.float(), dim=-1)
        pred_ids   = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0:
            break

        if is_first_block and step == 0:
            # 强制写入结构框架（toolcall_pre_rl 原理）
            for i, tok in enumerate(TOOL_CALL_TOKENS):
                if i < block_len:
                    input_ids[0, block_start + i] = tok
            for i, tok in enumerate(TOOL_CALL_END_TOKENS):
                pos = CONTENT_END + i
                if pos < block_len:
                    input_ids[0, block_start + pos] = tok
        else:
            n_reveal = max(1, round(block_len / num_steps))
            conf_m   = confidence.clone()
            conf_m[~still_masked] = -1.0
            if is_first_block:
                # 内容区 confidence boost
                conf_m[CONTENT_START:CONTENT_END] += 0.5
            n_reveal = min(n_reveal, still_masked.sum().item())
            top_idx  = conf_m.topk(n_reveal).indices
            for idx in top_idx:
                input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


@torch.no_grad()
def llada_generate_blocks(model, tokenizer, prompt_ids, mask_id,
                           block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                           max_blocks=MAX_BLOCKS, stop_strings=None,
                           force_toolcall=False):
    """
    force_toolcall=True：对第一个 block 应用 toolcall_priority 策略
    force_toolcall=False：所有 block 用普通置信度 denoising（用于答案轮）
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids = prompt_ids.to(device).clone()
    generated_ids = []

    for block_idx in range(max_blocks):
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        # 仅当 force_toolcall=True 且是第一个 block 时应用 toolcall_priority
        apply_priority = force_toolcall and (block_idx == 0)
        input_ids = denoise_block_toolcall_priority(
            model, input_ids, block_start, block_end,
            mask_id, num_steps, is_first_block=apply_priority
        )

        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop_str in stop_strings:
            if stop_str in current_text:
                idx = current_text.index(stop_str) + len(stop_str)
                return current_text[:idx]

        tail = block_tokens[-8:]
        if all(t == eos_id for t in tail):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 单题 agentic rollout ──────────────────────────────────────────────────────
def run_one(model, tokenizer, mask_id, question: str) -> dict:
    stop_strings = ["</tool_call>", "<|box_end|>"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prediction         = None
    termination_reason = "max_turns"
    num_turns          = 0

    for turn in range(MAX_TURNS):
        num_turns = turn + 1

        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        # 前两轮强制 toolcall 结构；第三轮起让模型自由生成（可输出答案）
        force_toolcall = (turn < 2)

        new_text = llada_generate_blocks(
            model, tokenizer, prompt_ids,
            mask_id=mask_id,
            block_size=BLOCK_SIZE,
            num_steps=NUM_STEPS,
            max_blocks=MAX_BLOCKS,
            stop_strings=stop_strings,
            force_toolcall=force_toolcall,
        )

        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction         = answer
            termination_reason = "answer"
            break

        queries = parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            messages.append({
                "role":    "user",
                "content": f"<tool_response>\n{search_result}\n</tool_response>",
            })
            tool_response = (
                f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            full_context += new_text + tool_response
        else:
            full_context += new_text

    return {
        "prediction":         prediction,
        "num_turns":          num_turns,
        "termination_reason": termination_reason,
        "messages":           messages,
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--input",       default=DEFAULT_INPUT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--offset",      type=int, default=0, help="从第几题开始（用于多进程并行）")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.offset > 0:
        samples = samples[args.offset:]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"eval 数据: {len(samples)} 题  来自 {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_questions = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_questions.add(json.loads(line)["question"])
                except Exception:
                    pass
        print(f"已完成 {len(done_questions)} 题，继续剩余部分")

    todo = [s for s in samples if s["question"] not in done_questions]
    if not todo:
        print("全部题目已完成！")
        return

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    mask_id = model.config.mask_token_id
    print(f"模型加载完成  mask_id={mask_id}")
    print(f"策略: toolcall_priority (LLaDA 适配)  block_size={BLOCK_SIZE}")
    print(f"待评测: {len(todo)} 题\n")

    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="baseline eval"):
            question  = sample["question"]
            gt_answer = sample["answer"]

            try:
                result = run_one(model, tokenizer, mask_id, question)
            except Exception as e:
                tqdm.write(f"[ERROR] {question[:60]}: {e}")
                result = {
                    "prediction":         None,
                    "num_turns":          0,
                    "termination_reason": "error",
                    "messages":           [],
                }

            record = {
                "question":           question,
                "answer":             gt_answer,
                "prediction":         result["prediction"],
                "num_turns":          result["num_turns"],
                "termination_reason": result["termination_reason"],
                "messages":           result["messages"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\n完成！结果保存至 {out_path}")
    print(f"打分: python my_eval/cal_acc.py --data {out_path}")


if __name__ == "__main__":
    main()
