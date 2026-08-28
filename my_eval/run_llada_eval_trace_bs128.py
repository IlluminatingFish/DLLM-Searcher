#!/usr/bin/env python3
"""
run_llada_eval_trace_bs128.py
=============================
bs128 ckpt_4 版本的 token 生成顺序追踪脚本。
与 run_llada_eval_trace.py 完全相同，唯一区别：
  - BLOCK_SIZE = 128, NUM_STEPS = 128
  - 默认模型 = sft_llada_x0pred_bs128/ckpt_4/optimized
  - 支持 --rank / --world 多 GPU 并行（每个 GPU 处理 158/world 道题）

用法（8 GPU 并行，screen 或 nohup）：
  for rank in $(seq 0 7); do
    SCREEN="bs128_trace_r${rank}"
    inner=/tmp/bs128_trace_r${rank}.sh
    cat > $inner << INNER
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${rank}
export TRITON_CACHE_DIR=/tmp/triton_bs128_trace_${rank}
mkdir -p /tmp/triton_bs128_trace_${rank}
python /research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/run_llada_eval_trace_bs128.py \\
  --rank ${rank} --world 8 \\
  --out /research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/bs128/trace/rank${rank}.jsonl
INNER
    chmod +x $inner
    screen -dmS "$SCREEN" bash $inner
  done
"""

import argparse, json, math, os, re, sys, torch, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x compat patch ────────────────────────────────────────────
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self): return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val): object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

def _patched_get_tied_weight_keys(module):
    keys = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None) or {}
        if isinstance(tied, dict): keys.extend([f"{name}.{k}" if name else k for k in tied])
        elif isinstance(tied, (list, tuple)): keys.extend([f"{name}.{k}" if name else k for k in tied])
    return keys
_mu._get_tied_weight_keys = _patched_get_tied_weight_keys

if "_finalize_model_loading" in PreTrainedModel.__dict__:
    _orig_fn = PreTrainedModel.__dict__["_finalize_model_loading"].__func__
    def _patched_finalize(model, load_config, loading_info):
        _orig_tw = type(model).tie_weights
        def _safe_tw(self, **kwargs):
            try: return _orig_tw(self, **kwargs)
            except TypeError: return _orig_tw(self)
        type(model).tie_weights = _safe_tw
        try: return _orig_fn(model, load_config, loading_info)
        finally: type(model).tie_weights = _orig_tw
    PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
    _orig_adj = PreTrainedModel._adjust_tied_keys_with_tied_pointers
    def _safe_adj(self, *a, **kw): return _orig_adj(self, *a, **kw)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adj

# ── 默认路径 ──────────────────────────────────────────────────────────────────
ROOT         = "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
BS128_MODEL  = f"{ROOT}/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred_bs128/ckpt_4/optimized"
WRONG_QS     = f"{ROOT}/my_eval/results/bs128/trace/bs128_wrong_qs.jsonl"
DEFAULT_OUT  = f"{ROOT}/my_eval/results/bs128/trace/rank0.jsonl"

# ── 超参（bs128 专用）────────────────────────────────────────────────────────
BLOCK_SIZE        = 128
NUM_STEPS         = 128
MAX_BLOCKS        = 12
MAX_TURNS         = 5
FORCE_ANSWER_TURN = 3

# ── Search API ────────────────────────────────────────────────────────────────
def _load_config():
    for path in [f"{ROOT}/config.json",
                 os.path.join(os.path.dirname(__file__), "..", "config.json")]:
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
        if resp.status_code != 200:
            return f"Search error {resp.status_code}"
        data = resp.json()
        if "organic" not in data:
            return f"No results for '{query}'."
        lines = [f"{i}. [{p['title']}] {p.get('snippet','')}"
                 for i, p in enumerate(data["organic"][:5], 1)]
        return f"Results for '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"

def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

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

# ── Parse helpers ─────────────────────────────────────────────────────────────
def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m: return None
    try:
        obj = json.loads(m.group(1).strip())
        return obj.get("arguments", {}).get("query", [])
    except:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1))
            except: pass
    return None

def extract_answer(text):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ── denoise_block_trace（bs128 版）───────────────────────────────────────────
@torch.no_grad()
def denoise_block_trace(model, input_ids, block_start, block_end, mask_id, num_steps):
    block_len = block_end - block_start
    reveal_step = [-1] * block_len
    reveal_conf = [0.0] * block_len

    for step in range(num_steps):
        logits = model(input_ids=input_ids).logits
        block_logits = logits[0, block_start:block_end]
        probs      = torch.softmax(block_logits.float(), dim=-1)
        pred_ids   = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0: break

        n_reveal = max(1, round(block_len / num_steps))
        conf_m = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, n_still)
        top_idx = conf_m.topk(n_reveal).indices

        for idx in top_idx:
            i = idx.item()
            input_ids[0, block_start + i] = pred_ids[i]
            reveal_step[i] = step
            reveal_conf[i] = confidence[i].item()

    for i in range(block_len):
        if reveal_step[i] < 0:
            reveal_step[i] = num_steps - 1
            reveal_conf[i] = 0.0

    return input_ids, reveal_step, reveal_conf


@torch.no_grad()
def llada_generate_trace(model, tokenizer, prompt_ids, mask_id,
                         stop_strings=None, max_blocks=12):
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids = prompt_ids.to(device).clone()
    generated_ids = []
    blocks_trace = []

    for block_idx in range(max_blocks):
        new_block = torch.full((1, BLOCK_SIZE), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - BLOCK_SIZE
        block_end   = input_ids.shape[1]

        input_ids, rev_step, rev_conf = denoise_block_trace(
            model, input_ids, block_start, block_end, mask_id, NUM_STEPS)

        block_ids  = input_ids[0, block_start:block_end].tolist()
        block_toks = tokenizer.convert_ids_to_tokens(block_ids)
        generated_ids.extend(block_ids)

        blocks_trace.append({
            "block_idx":   block_idx,
            "tokens":      [t if t else "<?>" for t in block_toks],
            "token_ids":   block_ids,
            "reveal_step": rev_step,
            "reveal_conf": [round(c, 4) for c in rev_conf],
        })

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        for stop in stop_strings:
            if stop in current_text:
                idx = current_text.index(stop) + len(stop)
                return current_text[:idx], blocks_trace

        tail = block_ids[-8:]
        if all(t == eos_id for t in tail):
            return tokenizer.decode(generated_ids, skip_special_tokens=True), blocks_trace

    return tokenizer.decode(generated_ids, skip_special_tokens=True), blocks_trace


# ── run_one_trace ─────────────────────────────────────────────────────────────
def run_one_trace(model, tokenizer, mask_id, question):
    stop_strings = ["</tool_call>", "<|box_end|>"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    prediction = None
    termination_reason = "max_turns"
    turns_trace = []
    tool_resp_idx = 0

    for turn in range(MAX_TURNS):
        force_answer = (turn >= FORCE_ANSWER_TURN)
        if force_answer:
            gen_context = full_context + "<|box_start|>"
            cur_stops   = ["<|box_end|>"]
        else:
            gen_context = full_context
            cur_stops   = stop_strings

        prompt_ids = tokenizer(
            gen_context, return_tensors="pt", add_special_tokens=False).input_ids

        new_text, blocks = llada_generate_trace(
            model, tokenizer, prompt_ids,
            mask_id=mask_id, stop_strings=cur_stops, max_blocks=MAX_BLOCKS)

        is_toolcall = "<tool_call>" in new_text
        has_answer  = "<|box_start|>" in new_text or force_answer

        turn_entry = {
            "turn_idx":     turn + 1,
            "is_toolcall":  is_toolcall,
            "force_answer": force_answer,
            "text":         new_text.strip(),
            "blocks":       blocks,
        }
        turns_trace.append(turn_entry)

        print(f"    turn {turn+1} tc={is_toolcall} ans={has_answer} "
              f"blocks={len(blocks)}  text={repr(new_text.strip()[:60])}")

        if force_answer:
            ans = new_text.split("<|box_end|>")[0].strip() if "<|box_end|>" in new_text else new_text.strip()
            if ans: prediction = ans
            termination_reason = "forced_answer"
            break

        answer = extract_answer(new_text)
        if answer:
            prediction = answer
            termination_reason = "answer"
            break

        queries = parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            turn_entry["search_result"] = search_result
            print(f"    → search #{tool_resp_idx+1}: {queries[:2]}")
            tool_resp_idx += 1
            tool_response = (
                f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                f"<|im_end|>\n<|im_start|>assistant\n")
            full_context += new_text + tool_response
        else:
            full_context += new_text

    return {
        "prediction":         prediction,
        "termination_reason": termination_reason,
        "turns":              turns_trace,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",   type=int, default=0)
    parser.add_argument("--world",  type=int, default=1)
    parser.add_argument("--model",  type=str, default=BS128_MODEL)
    parser.add_argument("--input",  type=str, default=WRONG_QS)
    parser.add_argument("--out",    type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    # 读取题目列表，按 rank 分片
    all_qs = [json.loads(l) for l in open(args.input) if l.strip()]
    chunk_size = math.ceil(len(all_qs) / args.world)
    my_qs = all_qs[args.rank * chunk_size : (args.rank + 1) * chunk_size]
    print(f"[rank {args.rank}/{args.world}] {len(my_qs)} 道题  model={args.model}")

    print(f"[load model] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"[ready] mask_id={mask_id}  BLOCK={BLOCK_SIZE}  STEPS={NUM_STEPS}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    for qi, qdata in enumerate(my_qs):
        question = qdata["question"]
        gt       = qdata["gt"]
        cat      = qdata["cat"]

        print(f"\n{'='*60}")
        print(f"[rank{args.rank} {qi+1}/{len(my_qs)}] cat={cat}  gt={gt!r}")
        print(f"  Q: {question[:80]}")

        result = run_one_trace(model, tokenizer, mask_id, question)

        pred = result["prediction"] or ""
        from cal_acc import cover_exact_match_score_1
        regen_correct = cover_exact_match_score_1(pred, gt)

        entry = {
            "idx":        qi + 1,
            "rank":       args.rank,
            "question":   question,
            "gt":         gt,
            "cat":        cat,
            "prediction": pred,
            "regen_correct":     regen_correct,
            "termination_reason": result["termination_reason"],
            "turns":      result["turns"],
        }
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  → pred={repr(pred[:60])}  ok={regen_correct}")

    out_f.close()
    print(f"\n[rank {args.rank}] 完成! → {args.out}")


if __name__ == "__main__":
    main()
