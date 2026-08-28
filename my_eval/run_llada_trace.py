#!/usr/bin/env python3
"""
LLaDA eval with full denoising trace.
Records for each generated block:
  - step_reveal[i] = which step position i was revealed (0-indexed)
  - tokens[i]      = final token string at position i
  - token_ids[i]   = final token id at position i
"""

import argparse, json, os, re, requests, signal, torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 兼容补丁 ─────────────────────────────────────────────────────
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self): return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val): object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

def _patched_get_tied_weight_keys(module):
    keys = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None) or {}
        if isinstance(tied, dict):  keys.extend([f"{name}.{k}" if name else k for k in tied])
        elif isinstance(tied, (list, tuple)): keys.extend([f"{name}.{k}" if name else k for k in tied])
    return keys
_mu._get_tied_weight_keys = _patched_get_tied_weight_keys

# ── 常量 ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE        = 128
NUM_STEPS         = 128
MAX_BLOCKS        = 16
MAX_TURNS         = 5
FORCE_ANSWER_TURN = 3

DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128/ckpt_4/optimized"
)

# ── Search ────────────────────────────────────────────────────────────────────
def _load_config():
    for path in [
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"

def google_search(query: str) -> str:
    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(SEARCH_URL, headers=headers,
                             json={"q": query, "num": 5}, timeout=10)
        if resp.status_code != 200: return f"Search error {resp.status_code}"
        data = resp.json()
        if "organic" not in data: return f"No results for '{query}'."
        lines = [f"{i}. [{p['title']}] {p.get('snippet','')}"
                 for i, p in enumerate(data["organic"][:5], 1)]
        return f"Results for '{query}':\n" + "\n".join(lines)
    except Exception as e: return f"Search failed: {e}"

def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

# ── Parse ─────────────────────────────────────────────────────────────────────
def parse_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m: return None
    try:
        obj = json.loads(m.group(1).strip())
        return obj.get("arguments", {}).get("query", [])
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1))
            except: pass
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

# ── Denoising with trace ──────────────────────────────────────────────────────
@torch.no_grad()
def denoise_block_traced(model, input_ids, block_start, block_end, mask_id, num_steps):
    """
    Like denoise_block but records for each position:
      step_revealed[pos] = step index (0-based) when position was revealed
    Returns (input_ids, step_revealed, pred_at_reveal)
    """
    block_len = block_end - block_start
    step_revealed  = [-1] * block_len   # -1 = still masked at end
    token_at_reveal = [mask_id] * block_len

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

        n_reveal = max(1, round(block_len / num_steps))
        conf_m = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, n_still)
        top_idx = conf_m.topk(n_reveal).indices

        for idx in top_idx:
            pos = idx.item()
            input_ids[0, block_start + pos] = pred_ids[pos]
            if step_revealed[pos] == -1:
                step_revealed[pos]   = step
                token_at_reveal[pos] = pred_ids[pos].item()

    return input_ids, step_revealed, token_at_reveal


@torch.no_grad()
def llada_generate_blocks_traced(model, tokenizer, prompt_ids, mask_id,
                                  block_size=64, num_steps=64, max_blocks=16,
                                  stop_strings=None):
    """Returns (text, block_traces) where block_traces is a list of dicts per block."""
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids   = prompt_ids.to(device).clone()
    generated_ids = []
    block_traces  = []

    for blk_idx in range(max_blocks):
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids, step_revealed, token_at_reveal = denoise_block_traced(
            model, input_ids, block_start, block_end, mask_id, num_steps
        )

        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)

        # decode each token individually for display
        token_strs = [tokenizer.decode([t], skip_special_tokens=False) for t in block_tokens]

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # find stop point (in characters, then map back)
        stop_hit   = None
        stop_text  = current_text
        for stop_str in stop_strings:
            if stop_str in current_text:
                idx = current_text.index(stop_str) + len(stop_str)
                stop_text = current_text[:idx]
                stop_hit  = stop_str
                break

        # how many generated chars are kept
        kept_text = stop_text if stop_hit else current_text
        # number of tokens in kept portion
        kept_token_count = len(tokenizer(kept_text, add_special_tokens=False).input_ids)
        kept_in_block    = max(0, kept_token_count - (len(generated_ids) - block_size))

        block_traces.append({
            "block_idx":    blk_idx,
            "tokens":       token_strs,
            "token_ids":    block_tokens,
            "step_revealed":step_revealed,   # list[int], position → step
            "kept_tokens":  kept_in_block,   # how many tokens in this block are in final output
            "stop_hit":     stop_hit,
        })

        if stop_hit:
            return stop_text, block_traces

        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True), block_traces

    return tokenizer.decode(generated_ids, skip_special_tokens=True), block_traces


# ── Single-question rollout with trace ───────────────────────────────────────
def run_one_traced(model, tokenizer, mask_id, question: str) -> dict:
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
    turns_trace        = []   # list of per-turn dicts with block_traces

    for turn in range(MAX_TURNS):
        num_turns   = turn + 1
        force_answer = (turn >= FORCE_ANSWER_TURN)

        if force_answer:
            gen_context = full_context + "<|box_start|>"
            cur_stops   = ["<|box_end|>"]
        else:
            gen_context = full_context
            cur_stops   = stop_strings

        prompt_ids = tokenizer(
            gen_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_text, block_traces = llada_generate_blocks_traced(
            model, tokenizer, prompt_ids,
            mask_id=mask_id,
            block_size=BLOCK_SIZE,
            num_steps=NUM_STEPS,
            max_blocks=MAX_BLOCKS,
            stop_strings=cur_stops,
        )

        turn_info = {
            "turn_id":      turn,
            "force_answer": force_answer,
            "new_text":     new_text,
            "block_traces": block_traces,
        }

        if force_answer:
            ans_text = new_text.split("<|box_end|>")[0].strip() \
                       if "<|box_end|>" in new_text else new_text.strip()
            if ans_text:
                prediction         = ans_text
                termination_reason = "forced_answer"
            messages.append({"role": "assistant", "content": f"<|box_start|>{new_text}"})
            turns_trace.append(turn_info)
            break

        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction         = answer
            termination_reason = "answer"
            turns_trace.append(turn_info)
            break

        queries = parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            messages.append({
                "role":    "user",
                "content": f"<tool_response>\n{search_result}\n</tool_response>",
            })
            tool_response = (
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"<tool_response>\n{search_result}\n</tool_response>"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            # 统计 search_result 的字符数和 token 数（用于分析模型是否读全了信息）
            sr_chars = len(search_result)
            sr_tokens = len(tokenizer(search_result, add_special_tokens=False).input_ids)
            ctx_after = full_context + new_text + tool_response
            ctx_tokens = len(tokenizer(ctx_after, add_special_tokens=False).input_ids)
            turn_info["search_queries"]       = queries
            turn_info["search_result"]        = search_result
            turn_info["search_result_chars"]  = sr_chars
            turn_info["search_result_tokens"] = sr_tokens
            turn_info["context_tokens_after"] = ctx_tokens
            full_context += new_text + tool_response
        else:
            full_context += new_text

        turns_trace.append(turn_info)

    return {
        "question":           question,
        "prediction":         prediction,
        "num_turns":          num_turns,
        "termination_reason": termination_reason,
        "messages":           messages,
        "turns_trace":        turns_trace,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global FORCE_ANSWER_TURN, BLOCK_SIZE, NUM_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--offset",      type=int, default=0)
    parser.add_argument("--force_answer_turn", type=int, default=FORCE_ANSWER_TURN)
    parser.add_argument("--block_size",        type=int, default=BLOCK_SIZE)
    parser.add_argument("--num_steps",         type=int, default=NUM_STEPS)
    args = parser.parse_args()
    FORCE_ANSWER_TURN = args.force_answer_turn
    BLOCK_SIZE        = args.block_size
    NUM_STEPS         = args.num_steps

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    if args.offset > 0:   samples = samples[args.offset:]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f"eval 数据: {len(samples)} 题")

    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        mask_id = 126336
    print(f"模型加载完成，mask_token_id={mask_id}")

    QUESTION_TIMEOUT = 300  # 每题最多 5 分钟

    def _timeout_handler(signum, frame):
        raise TimeoutError("推理超时")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        for i, sample in enumerate(samples):
            q = sample.get("question", sample.get("q", ""))
            print(f"[{i+1}/{len(samples)}] {q[:80]}", flush=True)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(QUESTION_TIMEOUT)
            try:
                r = run_one_traced(model, tokenizer, mask_id, q)
            except TimeoutError:
                print(f"  → 超时（>{QUESTION_TIMEOUT}s），跳过", flush=True)
                r = {"prediction": "", "num_turns": 0,
                     "termination_reason": "timeout", "turns": [], "block_traces": []}
            finally:
                signal.alarm(0)
            r["answer"]       = sample.get("answer", "")
            r["short_answer"] = sample.get("short_answer", "")
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  → pred={r['prediction']!r}  turns={r['num_turns']}  reason={r['termination_reason']}", flush=True)
    print(f"\n完成！结果 → {args.output}", flush=True)


if __name__ == "__main__":
    main()
