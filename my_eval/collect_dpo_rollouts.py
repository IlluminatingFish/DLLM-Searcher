#!/usr/bin/env python3
"""
从 x0pred 模型收集 DPO rollout 数据。

对每道题用 temperature sampling 跑 N 次，保存所有轨迹。
后续用 make_dpo_pairs.py 做 LLM judge + 配对。

用法（8 GPU 并行，每卡跑一份）：
  CUDA_VISIBLE_DEVICES=0 python collect_dpo_rollouts.py --rank 0 --world 8 --n_rolls 4
"""

import argparse, json, os, re, requests, torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 超参 ─────────────────────────────────────────────────────────────────────
BLOCK_SIZE   = 64
NUM_STEPS    = 64
MAX_BLOCKS   = 16
MAX_TURNS    = 5
TEMPERATURE  = 0.8   # < 1 保证质量，> 0 产生多样性

DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)

# ── Search ────────────────────────────────────────────────────────────────────
def _load_config():
    for p in ["/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
              os.path.join(os.path.dirname(__file__), "..", "config.json")]:
        if os.path.exists(p):
            return json.load(open(p))
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"

def google_search(q):
    try:
        r = requests.post(SEARCH_URL, headers={"X-API-KEY": GOOGLE_KEY,
                          "Content-Type": "application/json"},
                          json={"q": q, "num": 5}, timeout=10)
        if r.status_code != 200: return f"Search error {r.status_code}"
        data = r.json()
        if "organic" not in data: return f"No results for '{q}'."
        return f"Results for '{q}':\n" + "\n".join(
            f"{i}. [{p['title']}] {p.get('snippet','')}"
            for i, p in enumerate(data["organic"][:5], 1))
    except Exception as e:
        return f"Search failed: {e}"

def run_search(queries):
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

# ── Parse ─────────────────────────────────────────────────────────────────────
def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m: return None
    try:
        return json.loads(m.group(1).strip()).get("arguments", {}).get("query", [])
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1))
            except: pass
    return None

def extract_answer(text):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ── Prompt ────────────────────────────────────────────────────────────────────
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

# ── Block diffusion with temperature sampling ─────────────────────────────────
@torch.no_grad()
def denoise_block_sample(model, input_ids, block_start, block_end,
                         mask_id, num_steps, temperature=1.0):
    """Block diffusion with temperature sampling for diversity."""
    block_len = block_end - block_start
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0:
            break

        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)

        # Sample for diversity (vs argmax in standard eval)
        pred_ids   = torch.multinomial(probs, 1).squeeze(-1)
        confidence = probs.max(dim=-1).values

        n_reveal = max(1, round(block_len / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, n_still)
        top_idx  = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]
    return input_ids


@torch.no_grad()
def llada_generate_sample(model, tokenizer, prompt_ids, mask_id,
                          block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                          max_blocks=MAX_BLOCKS, stop_strings=None,
                          temperature=TEMPERATURE):
    device       = next(model.parameters()).device
    eos_id       = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids     = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block   = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block_sample(
            model, input_ids, block_start, block_end,
            mask_id, num_steps, temperature
        )

        block_tokens  = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text  = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop in stop_strings:
            if stop in current_text:
                return current_text[:current_text.index(stop) + len(stop)]

        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Single rollout ─────────────────────────────────────────────────────────────
def run_one_rollout(model, tokenizer, mask_id, question, temperature=TEMPERATURE):
    stop_strings = ["</tool_call>", "<|box_end|>"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prediction = None

    for turn in range(MAX_TURNS):
        prompt_ids = tokenizer(full_context, return_tensors="pt",
                               add_special_tokens=False).input_ids
        new_text = llada_generate_sample(
            model, tokenizer, prompt_ids, mask_id,
            stop_strings=stop_strings, temperature=temperature
        )
        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction = answer
            break

        queries = parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            messages.append({"role": "user",
                             "content": f"<tool_response>\n{search_result}\n</tool_response>"})
            tool_resp = (f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                        f"<|im_end|>\n<|im_start|>assistant\n")
            full_context += new_text + tool_resp
        else:
            full_context += new_text

    return {"prediction": prediction, "messages": messages,
            "termination": "answer" if prediction else "max_turns"}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--input",   default=DEFAULT_INPUT)
    parser.add_argument("--output",  default="output/dpo_rollouts")
    parser.add_argument("--n_rolls", type=int, default=4, help="每题 rollout 次数")
    parser.add_argument("--rank",    type=int, default=0)
    parser.add_argument("--world",   type=int, default=8)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    samples = samples[args.rank::args.world]
    print(f"Rank {args.rank}/{args.world}: {len(samples)} questions × {args.n_rolls} rolls")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rank{args.rank}.jsonl"

    done = set()
    if out_file.exists():
        for l in open(out_file):
            try:
                d = json.loads(l)
                done.add((d["question"], d["roll_idx"]))
            except: pass
        print(f"已完成 {len(done)} 条，继续")

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    mask_id = model.config.mask_token_id
    print(f"mask_id={mask_id}  temperature={args.temperature}\n")

    with open(out_file, "a") as f:
        for sample in tqdm(samples, desc=f"rank{args.rank}"):
            q = sample["question"]
            a = sample["answer"]
            for roll_idx in range(args.n_rolls):
                if (q, roll_idx) in done:
                    continue
                try:
                    result = run_one_rollout(model, tokenizer, mask_id, q, args.temperature)
                except Exception as e:
                    tqdm.write(f"[ERROR] roll={roll_idx} {q[:40]}: {e}")
                    result = {"prediction": None, "messages": [], "termination": "error"}

                record = {
                    "question":    q,
                    "answer":      a,
                    "roll_idx":    roll_idx,
                    "prediction":  result["prediction"],
                    "messages":    result["messages"],
                    "termination": result["termination"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\n完成！输出: {out_file}")

if __name__ == "__main__":
    main()
