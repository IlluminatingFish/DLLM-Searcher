#!/usr/bin/env python3
"""
LLaDA baseline 推理评测脚本 —— 最大程度还原原版推理策略。

原版关键特性（来自 jetengine/scheduler.py + my_test.py）：
  1. block_size = 128，denoising_steps = 128
  2. toolcall_pre_rl remasking 策略：
       step 0：强制位置 63 = <tool_call>(151657)，位置 126 = </tool_call>(151658)
       step >0：confidence[63:] += 0.5，按置信度揭示 token
  3. Prompt padding：prompt 长度预先 pad 到 128 的整数倍，保证 block 边界对齐
  4. 模型：sft_llada/ckpt_6（标准 SFT baseline，非 x0pred）

用法:
  python my_eval/run_llada_baseline_eval.py \
    --model dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized \
    --input dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
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

# ── 超参（与原版 my_test.py 对齐） ────────────────────────────────────────────
BLOCK_SIZE  = 128
NUM_STEPS   = 128
MAX_BLOCKS  = 8
MAX_TURNS   = 5

# toolcall_pre_rl 固定位置的 token id（来自 scheduler.py 注释）
TOOL_CALL_START_ID = 151657   # <tool_call>
TOOL_CALL_END_ID   = 151658   # </tool_call>
TOOL_CALL_POS      = 63       # 强制 <tool_call> 位置
TOOL_CALL_END_POS  = 126      # 强制 </tool_call> 位置

# 用于 prompt padding 的填充文本（来自 my_test.py format_prompt）
PADDING_TEMPLATE = (
    "Note that I need you to keep the content as concise as possible, "
    "limit it to no more than 128 tokens, but near to 128 tokens."
) * 10

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
GOOGLE_KEY  = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL  = "https://google.serper.dev/search"


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


# ── Prompt template ───────────────────────────────────────────────────────────
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


# ── Prompt padding（对齐 block 边界，来自 my_test.py format_prompt） ──────────
def pad_prompt_to_block_boundary(tokenizer, prompt_text: str, block_size: int = BLOCK_SIZE) -> str:
    """把 prompt 长度 pad 到 block_size 的整数倍，在 prompt 前面插入填充文本。"""
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    padding_len = block_size - (prompt_len % block_size)
    if padding_len == block_size:
        return prompt_text  # 已经对齐
    # 取前 padding_len 个 padding token，解码后插到 prompt 前面
    pad_ids = tokenizer.encode(PADDING_TEMPLATE, add_special_tokens=False)[:padding_len]
    pad_text = tokenizer.decode(pad_ids)
    return pad_text + prompt_text


# ── toolcall_pre_rl block diffusion ──────────────────────────────────────────
@torch.no_grad()
def denoise_block_toolcall_pre_rl(
    model, input_ids, block_start, block_end, mask_id, num_steps
):
    """
    还原 scheduler.py toolcall_pre_rl 策略：
      step 0：强制 pos 63 = <tool_call>，pos 126 = </tool_call>
      step >0：confidence[63:] += 0.5，按置信度揭示 token
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

        if step == 0:
            # 强制写入结构 token
            if TOOL_CALL_POS < block_len:
                input_ids[0, block_start + TOOL_CALL_POS]     = TOOL_CALL_START_ID
            if TOOL_CALL_END_POS < block_len:
                input_ids[0, block_start + TOOL_CALL_END_POS] = TOOL_CALL_END_ID
        else:
            n_reveal = max(1, round(block_len / num_steps))
            conf_m   = confidence.clone()
            conf_m[~still_masked] = -1.0
            # 位置 63+ 加 +0.5 boost（优先揭示 tool_call 内容）
            conf_m[TOOL_CALL_POS:] = conf_m[TOOL_CALL_POS:] + 0.5
            n_reveal = min(n_reveal, still_masked.sum().item())
            top_idx  = conf_m.topk(n_reveal).indices
            for idx in top_idx:
                input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


@torch.no_grad()
def llada_generate_toolcall_pre_rl(
    model, tokenizer, prompt_ids, mask_id,
    block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
    max_blocks=MAX_BLOCKS, stop_strings=None
):
    device       = next(model.parameters()).device
    eos_id       = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids    = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block   = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block_toolcall_pre_rl(
            model, input_ids, block_start, block_end, mask_id, num_steps
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

        # Pad prompt 到 block 边界（还原 my_test.py format_prompt 逻辑）
        padded_context = pad_prompt_to_block_boundary(tokenizer, full_context)
        prompt_ids = tokenizer(
            padded_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_text = llada_generate_toolcall_pre_rl(
            model, tokenizer, prompt_ids,
            mask_id=mask_id,
            stop_strings=stop_strings,
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
    parser.add_argument("--offset",      type=int, default=0)
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
    print(f"策略: toolcall_pre_rl  block_size={BLOCK_SIZE}  num_steps={NUM_STEPS}")
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
