#!/usr/bin/env python3
"""
Interactive demo: run the SDAR DPO model with live search.
Usage:
    python demo.py
    python demo.py --model /path/to/checkpoint
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/base_sdar_eval"
)

def _load_config():
    cfg = {}
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    for path in [config_path,
                 os.path.join(os.path.dirname(__file__), "config.json"),
                 "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json"]:
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            break
    return cfg

_cfg = _load_config()
GOOGLE_SEARCH_KEY = (os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key") or "")
SEARCH_URL = "https://google.serper.dev/search"

BLOCK_LENGTH    = 128
DENOISING_STEPS = 128
TEMPERATURE     = 1.0
MAX_TURNS       = 5

# ── Prompts (identical to my_test.py) ────────────────────────────────────────
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

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question. The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query string 1", "query string 2", ...]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["another query string"]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<|box_start|>
answer here
<|box_end|>
The assistant must strictly abide by the above format. Tool calls should be placed between <tool_call> and </tool_call>, tool responses between <tool_response> and </tool_response>, thinking processes between <think> and </think>, and answers between <|box_start|> and <|box_end|>.
User: """

ANSWER_INSTRUCTION = (
    "<|im_start|>user\n"
    "Based on your research above, provide your final answer in the format "
    "<|box_start|>answer<|box_end|>.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ── Search tool ───────────────────────────────────────────────────────────────
def google_search(query: str) -> str:
    headers = {"X-API-KEY": GOOGLE_SEARCH_KEY, "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(SEARCH_URL, headers=headers,
                                 json={"q": query, "num": 10}, timeout=10)
            if resp.status_code != 200:
                return f"Search error {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            if "organic" not in data:
                return f"No results for '{query}'."
            snippets = []
            for i, page in enumerate(data["organic"], 1):
                date = f"\nDate: {page['date']}" if "date" in page else ""
                snippet = f"\n{page.get('snippet','')}" if "snippet" in page else ""
                snippets.append(f"{i}. [{page['title']}]({page['link']}){date}{snippet}")
            return f"Google search for '{query}' found {len(snippets)} results:\n\n" + "\n\n".join(snippets)
        except Exception as e:
            if attempt == 2:
                return f"Search failed: {e}"
    return "Search failed."


def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n=======\n\n".join(results)


# ── Parse tool call from model output ────────────────────────────────────────
def parse_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        return obj.get("arguments", {}).get("query", [])
    except Exception:
        # Try extracting query array directly
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass
    return None


def extract_answer(text: str):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ── Padding to block boundary (matches my_test.py) ───────────────────────────
def pad_to_block(prompt: str, tokenizer) -> str:
    PADDING_PROMPT = ("Note that I need you to keep the content as concise as possible, "
                      "limit it to no more than 128 tokens, but near to 128 tokens.") * 10
    prompt_len = len(tokenizer.encode(prompt))
    padding_len = BLOCK_LENGTH - (prompt_len % BLOCK_LENGTH)
    if padding_len == BLOCK_LENGTH:
        padding_len = 0
    if padding_len > 0:
        pad_tokens = tokenizer.decode(tokenizer.encode(PADDING_PROMPT)[:padding_len])
        prompt = pad_tokens + prompt
    return prompt


# ── Main interactive loop ─────────────────────────────────────────────────────
def run_demo(model_path: str):
    print(f"\nLoading model from: {model_path}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "my_train"))
    from transformers import AutoTokenizer
    from jetengine import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        mask_token_id=tokenizer.mask_token_id,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        max_model_len=8192,
        block_length=BLOCK_LENGTH,
        max_num_seqs=1,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=512,
        block_length=BLOCK_LENGTH,
        denoising_steps=DENOISING_STEPS,
        remasking_strategy="low_confidence_static",
        dynamic_threshold=0.9,
        stop_words=[151645, 151658],  # <|im_end|>, </tool_call>
        topk=1,
    )

    print("\nModel loaded. Type your question (or 'quit' to exit).\n")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        # Build initial context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
        ]
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        context = pad_to_block(context, tokenizer)

        print("\n" + "-" * 60)
        answer = None

        for turn in range(MAX_TURNS):
            print(f"\n[Turn {turn+1}] Generating...", flush=True)

            if turn >= 3:
                context += ANSWER_INSTRUCTION
                sampling_params.stop_words = [151645, 151658, 151649]  # + <|box_end|>

            padded = pad_to_block(context, tokenizer)
            outputs = list(llm.generate_streaming([padded], sampling_params, max_active=1))
            new_text = outputs[0]['text'] if outputs else ""
            print(f"[Model output]: {new_text[:300]}{'...' if len(new_text)>300 else ''}")

            # Check for answer
            answer = extract_answer(new_text)
            if answer:
                context += new_text
                break

            # Check for tool call
            queries = parse_tool_call(new_text)
            if queries:
                print(f"[Search queries]: {queries}")
                search_result = run_search(queries)
                print(f"[Search result]: {search_result[:300]}...")
                tool_response = (
                    f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                    f"<|im_end|>\n<|im_start|>assistant\n"
                )
                context += new_text + tool_response
            else:
                context += new_text

        print("\n" + "=" * 60)
        if answer:
            print(f"ANSWER: {answer}")
        else:
            print("(No answer extracted — max turns reached)")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    args = parser.parse_args()
    run_demo(args.model)
