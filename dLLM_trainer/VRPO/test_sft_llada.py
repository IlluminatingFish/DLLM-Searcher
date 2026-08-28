#!/usr/bin/env python3
"""
测试 SFT 后的 LLaDA 模型推理效果，带真实搜索工具。
用法: python test_sft_llada.py [--model /path/to/ckpt]

LLaDA 推理：按 block_size=64 逐块去噪，每块运行 num_steps 步迭代解噪。
"""
import argparse
import json
import os
import re
import sys
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 配置 ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
)
BLOCK_SIZE   = 64    # 与训练一致
NUM_STEPS    = 64    # 每块去噪步数；=block_size 保证每个 token 都有机会被揭露
MAX_BLOCKS   = 16    # 最多生成 16×64=1024 token
MAX_TURNS    = 5

def _load_config():
    for path in [
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
        os.path.join(os.path.dirname(__file__), "../../config.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL  = "https://google.serper.dev/search"

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

# ── 搜索工具 ──────────────────────────────────────────────────────────────────
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
        lines = []
        for i, p in enumerate(data["organic"][:5], 1):
            lines.append(f"{i}. [{p['title']}] {p.get('snippet','')}")
        return f"Results for '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"

def run_search(queries):
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

def parse_tool_call(text):
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

def extract_answer(text):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ── Block-by-block LLaDA 生成 ─────────────────────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps, eos_id):
    """
    对 input_ids 中 [block_start:block_end] 位置迭代去噪。
    前面的 token 已经固定（prompt + 已生成的块）。
    返回去噪后的完整 input_ids（仅 block 部分被更新）。
    """
    B, L = input_ids.shape
    block_len = block_end - block_start

    for step in range(num_steps):
        logits = model(input_ids=input_ids).logits   # (1, L, V)
        block_logits = logits[0, block_start:block_end]  # (block_len, V)

        probs      = torch.softmax(block_logits.float(), dim=-1)
        pred_ids   = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0:
            break

        # 线性 schedule：每步揭露 block_len/num_steps 个
        n_reveal = max(1, round(block_len / num_steps))
        conf_m = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, n_still)
        top_idx = conf_m.topk(n_reveal).indices

        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


@torch.no_grad()
def llada_generate_blocks(model, tokenizer, prompt_ids, mask_id,
                           block_size=64, num_steps=10, max_blocks=16,
                           stop_strings=None):
    """
    逐块 LLaDA 生成，遇到 stop_strings 中的字符串就停止。
    返回生成的文本（不含 prompt）。
    """
    device  = next(model.parameters()).device
    eos_id  = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids = prompt_ids.to(device).clone()
    generated_ids = []

    for block_idx in range(max_blocks):
        # 追加一个全 mask 的新块
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        # 迭代去噪该块
        input_ids = denoise_block(model, input_ids, block_start, block_end,
                                  mask_id, num_steps, eos_id)

        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)

        # 解码当前全部生成内容，检查 stop_string
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 去掉 EOS 之后的内容
        for stop_str in stop_strings:
            if stop_str in current_text:
                # 截断到 stop_str 结束位置
                idx = current_text.index(stop_str) + len(stop_str)
                current_text = current_text[:idx]
                return current_text

        # 如果最后几个 token 都是 EOS，也停止
        tail = block_tokens[-8:]
        if all(t == eos_id for t in tail):
            # 截断到第一个 EOS
            full = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return full

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 主推理逻辑 ────────────────────────────────────────────────────────────────
def run_inference(model_path, questions):
    print(f"\n加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    mask_id = model.config.mask_token_id
    print(f"模型加载完成，mask_token_id={mask_id}\n")

    # stop strings
    stop_strings = ["</tool_call>", "<|box_end|>"]

    for question in questions:
        print("=" * 70)
        print(f"问题: {question}")
        print("=" * 70)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
        ]
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_context = context
        answer = None

        for turn in range(MAX_TURNS):
            print(f"\n[Turn {turn+1}] 生成中...", flush=True)
            prompt_ids = tokenizer(
                full_context, return_tensors="pt", add_special_tokens=False
            ).input_ids

            new_text = llada_generate_blocks(
                model, tokenizer, prompt_ids,
                mask_id=mask_id,
                block_size=BLOCK_SIZE,
                num_steps=NUM_STEPS,
                max_blocks=MAX_BLOCKS,
                stop_strings=stop_strings,
            )
            print(f"[输出]: {new_text[:500]}{'...' if len(new_text)>500 else ''}")

            answer = extract_answer(new_text)
            if answer:
                print(f"\n最终答案: {answer}")
                break

            queries = parse_tool_call(new_text)
            if queries:
                print(f"[搜索]: {queries}")
                search_result = run_search(queries)
                print(f"[搜索结果前300字]: {search_result[:300]}...")
                tool_response = (
                    f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                    f"<|im_end|>\n<|im_start|>assistant\n"
                )
                full_context += new_text + tool_response
            else:
                full_context += new_text

        if not answer:
            print("(未找到答案，已达最大轮次)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    questions = [
        "What is the birthplace of the director of the movie Inception?",
        "Which country won the most gold medals at the 2020 Tokyo Olympics?",
        "What river flows through the city that hosted the 2016 Summer Olympics?",
    ]
    run_inference(args.model, questions)
