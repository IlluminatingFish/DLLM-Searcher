#!/usr/bin/env python3
"""
Agentic rollout 收集：扩展版 collect_grpo_rollouts.py，支持 Plan A/B/C 三种 reward 模式。

Plan A (basic, for hermes server):
    reward = F1 × efficiency + format_bonus  (capped at 1.0)
    - efficiency = max(0.5, 1.0 - 0.1 × max(0, n_searches - 2))
    - format_bonus = +0.1 if ≥1 turn had a valid <tool_call> JSON

Plan B (multi-component, for achilleas server):
    total = r_correctness × r_efficiency + 0.15 × r_format + 0.10 × r_search_utility
    - r_correctness: F1 score (or binary for yes/no questions)
    - r_format: fraction of tool_call turns with parseable JSON containing 'query' field
    - r_search_utility: fraction of searches where GT keywords appear in results
    - r_efficiency: same as Plan A efficiency
    (capped at 1.0; reward_breakdown saved in output for analysis)

Plan C (SALT-lite turn-level credit, for homer server):
    - Final reward: same formula as Plan A
    - Additionally saves turn_credits list per rollout:
        [{turn_idx, queries, tool_call_text, search_result, gt_keywords_found}, ...]
      Used downstream by make_agentic_data.py for per-turn advantage assignment.

Usage (8 GPU 并行，每卡一份):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python collect_agentic_rollouts.py \\
        --plan A --rank $i --world 8 --n_rolls 8 &
    done
    # Replace --plan A with B or C for different servers.
"""

import argparse
import json
import os
import re
import requests
import string
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x 兼容补丁（LLaDA tie_weights / all_tied_weights_keys）──────
# LLaDA 缺少 all_tied_weights_keys 属性，需要在 _adjust_tied_keys 之前改成 property。
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self):
        return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val):
        object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

def _patched_get_tied_weight_keys(module):
    keys = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None) or {}
        if isinstance(tied, dict):
            keys.extend([f"{name}.{k}" if name else k for k in tied])
        elif isinstance(tied, (list, tuple)):
            keys.extend([f"{name}.{k}" if name else k for k in tied])
    return keys
_mu._get_tied_weight_keys = _patched_get_tied_weight_keys

if "_finalize_model_loading" in PreTrainedModel.__dict__:
    _orig_finalize_fn = PreTrainedModel.__dict__["_finalize_model_loading"].__func__
    def _patched_finalize(model, load_config, loading_info):
        _orig_tw = type(model).tie_weights
        def _safe_tw(self, **kwargs):
            try:
                return _orig_tw(self, **kwargs)
            except TypeError:
                return _orig_tw(self)
        type(model).tie_weights = _safe_tw
        try:
            return _orig_finalize_fn(model, load_config, loading_info)
        finally:
            type(model).tie_weights = _orig_tw
    PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
    _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
    def _safe_adjust(self, *args, **kwargs):
        return _orig_adjust(self, *args, **kwargs)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# ── 超参 ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 64  # 可被 --block_size 覆盖
NUM_STEPS   = 64  # 可被 --num_steps 覆盖
MAX_BLOCKS  = 16   # max 16 × 64 = 1024 generated tokens per assistant turn
MAX_TURNS   = 5    # max 5 think→search cycles
TEMPERATURE = 0.9  # slightly higher than DPO rollouts for diversity

# ── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/Agentic/output/rollouts"
)

# ── Search API ────────────────────────────────────────────────────────────────
def _load_config():
    """Load config.json containing the Serper API key."""
    for p in ["/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"


def google_search(q: str) -> str:
    """Single query to Serper Google Search API. Returns formatted snippet text."""
    try:
        r = requests.post(
            SEARCH_URL,
            headers={"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"},
            json={"q": q, "num": 5},
            timeout=10,
        )
        if r.status_code != 200:
            return f"Search error {r.status_code}"
        data = r.json()
        if "organic" not in data:
            return f"No results for '{q}'."
        return f"Results for '{q}':\n" + "\n".join(
            f"{i}. [{p['title']}] {p.get('snippet', '')}"
            for i, p in enumerate(data["organic"][:5], 1)
        )
    except Exception as e:
        return f"Search failed: {e}"


def run_search(queries: list) -> str:
    """Run multiple queries in parallel and join results."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)


# ── Text normalization & F1 scoring ──────────────────────────────────────────
def _normalize(s: str) -> str:
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())


def _bool_map(s: str) -> str:
    """Map Python bool strings to yes/no."""
    return {"True": "yes", "False": "no"}.get(s, s)


def _first_sentence(s: str) -> str:
    """Take the first sentence (before first period/newline) as the key answer span."""
    s = s.strip()
    for sep in [". ", ".\n", "\n\n", "\n"]:
        idx = s.find(sep)
        if idx > 10:
            return s[: idx + 1]
    return s[:200]


def _yes_no(s: str):
    """Detect yes/no polarity from text. Returns 'yes', 'no', or None."""
    s = s.strip().lower()
    if re.match(r"^(yes[,\s]|yes$)", s):
        return "yes"
    if re.match(r"^(no[,\s]|no$)", s):
        return "no"
    if re.search(r"\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b", s):
        return "no"
    if re.search(r"\bthe same\b|\bboth\b|\byes\b", s):
        return "yes"
    return None


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalized prediction and first-sentence GT."""
    if not prediction or not str(prediction).strip():
        return 0.0
    gt_key    = _normalize(_bool_map(_first_sentence(str(ground_truth))))
    pred_norm = _normalize(_bool_map(str(prediction)))
    gt_tok    = set(gt_key.split())
    pred_tok  = set(pred_norm.split())
    if not gt_tok:
        return 0.0
    common = gt_tok & pred_tok
    if not common:
        return 0.0
    precision = len(common) / max(len(pred_tok), 1)
    recall    = len(common) / len(gt_tok)
    return 2 * precision * recall / (precision + recall + 1e-9)


def count_searches(messages: list) -> int:
    """Count tool_response turns (= number of actual web searches performed)."""
    return sum(
        1 for m in messages
        if m.get("role") == "user" and "<tool_response>" in m.get("content", "")
    )


def get_gt_keywords(ground_truth: str) -> set:
    """Extract meaningful content keywords from the ground truth answer.

    Used for r_search_utility and turn_credits gt_keywords_found computation.
    Filters out common English stopwords to focus on substantive terms.
    """
    _stopwords = {
        "a", "an", "the", "in", "of", "is", "are", "was", "were", "to",
        "and", "or", "it", "its", "this", "that", "at", "by", "for", "on",
        "as", "be", "but", "not", "with", "from", "has", "have", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "been", "being", "their", "they", "them",
        "he", "she", "we", "you", "i", "me", "my", "your", "his", "her",
        "us", "our", "who", "which", "what", "when", "where", "how", "",
    }
    gt_norm = _normalize(_bool_map(_first_sentence(str(ground_truth))))
    return set(gt_norm.split()) - _stopwords


# ── Reward functions (one per plan) ──────────────────────────────────────────

def _base_correctness(prediction, ground_truth):
    """Compute correctness score.

    Returns:
        (score, is_yes_no): score in [0, 1], is_yes_no indicates binary question.
    """
    if not prediction:
        return 0.0, False
    gt_yn   = _yes_no(str(ground_truth))
    pred_yn = _yes_no(str(prediction))
    if gt_yn is not None and pred_yn is not None:
        return (1.0 if gt_yn == pred_yn else 0.0), True
    return f1_score(prediction, ground_truth), False


def compute_reward_plan_a(prediction, ground_truth, messages, turn_info) -> float:
    """Plan A reward: F1 × efficiency + format_bonus, capped at 1.0.

    - efficiency = max(0.5, 1.0 - 0.1 × max(0, n_searches - 2))
    - format_bonus = +0.1 if ≥1 turn had a valid <tool_call> JSON (with 'query' field)
    """
    n_searches = count_searches(messages)
    efficiency = max(0.5, 1.0 - 0.1 * max(0, n_searches - 2))

    r_corr, is_yn = _base_correctness(prediction, ground_truth)
    if is_yn:
        # Binary yes/no: full efficiency or 0
        base = r_corr * efficiency
    else:
        base = 0.0 if r_corr <= 0.0 else r_corr * efficiency

    # format bonus: +0.1 for any valid tool_call JSON with "query" field
    has_valid_tc = any(t.get("valid_tool_call", False) for t in turn_info)
    format_bonus = 0.1 if has_valid_tc else 0.0

    return round(min(1.0, base + format_bonus), 4)


def compute_reward_plan_b(prediction, ground_truth, messages, turn_info) -> float:
    """Plan B reward: multi-component, capped at 1.0.

    total = r_correctness × r_efficiency + 0.15 × r_format + 0.10 × r_search_utility
    """
    n_searches   = count_searches(messages)
    r_efficiency = max(0.5, 1.0 - 0.1 * max(0, n_searches - 2))

    # Correctness (F1 or binary)
    r_corr, _ = _base_correctness(prediction, ground_truth)

    # Format: fraction of search turns with parseable JSON containing "query" field
    if turn_info:
        r_format = sum(1.0 for t in turn_info if t.get("valid_tool_call", False)) / len(turn_info)
    else:
        r_format = 0.0

    # Search utility: fraction of searches where GT keywords appear in results
    gt_kw = get_gt_keywords(ground_truth)
    if turn_info and gt_kw:
        found_count = sum(
            1 for t in turn_info
            if any(kw in _normalize(t.get("search_result", "")) for kw in gt_kw)
        )
        r_search_utility = found_count / len(turn_info)
    else:
        r_search_utility = 0.0

    total = (r_corr * r_efficiency
             + 0.15 * r_format
             + 0.10 * r_search_utility)
    return round(min(1.0, total), 4)


def compute_reward_plan_c(prediction, ground_truth, messages, turn_info) -> float:
    """Plan C reward: same as Plan A. turn_credits are computed separately."""
    return compute_reward_plan_a(prediction, ground_truth, messages, turn_info)


def compute_turn_credits(ground_truth: str, turn_info: list) -> list:
    """Compute per-search-turn credits for Plan C (SALT-lite).

    For each search turn, records whether GT keywords appeared in the search
    result. This information is used by make_agentic_data.py to assign per-turn
    advantages.

    Returns:
        List of dicts: [{turn_idx, queries, tool_call_text, search_result,
                         gt_keywords_found}, ...]
    """
    gt_kw = get_gt_keywords(ground_truth)
    credits = []
    for t in turn_info:
        # Only include turns that actually did a search (have a search_result)
        if not t.get("queries"):
            continue
        search_result = t.get("search_result", "")
        gt_found = bool(gt_kw and any(
            kw in _normalize(search_result) for kw in gt_kw
        ))
        credits.append({
            "turn_idx":          t.get("turn_idx", -1),
            "queries":           t.get("queries", []),
            "tool_call_text":    t.get("tool_call_text", ""),
            "search_result":     search_result,
            "gt_keywords_found": gt_found,
        })
    return credits


def get_plan_b_breakdown(prediction, ground_truth, messages, turn_info) -> dict:
    """Compute Plan B component breakdown for logging/analysis."""
    n_s = count_searches(messages)
    r_e = max(0.5, 1.0 - 0.1 * max(0, n_s - 2))
    r_c, _ = _base_correctness(prediction, ground_truth)

    gt_kw = get_gt_keywords(ground_truth)
    if turn_info and gt_kw:
        found = sum(
            1 for t in turn_info
            if any(kw in _normalize(t.get("search_result", "")) for kw in gt_kw)
        )
        r_su = found / len(turn_info)
    else:
        r_su = 0.0

    r_f = (
        sum(1.0 for t in turn_info if t.get("valid_tool_call", False)) / len(turn_info)
        if turn_info else 0.0
    )
    return {
        "r_correctness":    round(r_c, 4),
        "r_efficiency":     round(r_e, 4),
        "r_format":         round(r_f, 4),
        "r_search_utility": round(r_su, 4),
    }


def compute_reward(plan: str, prediction, ground_truth, messages, turn_info) -> float:
    """Dispatch to the plan-specific reward function."""
    if plan == "A":
        return compute_reward_plan_a(prediction, ground_truth, messages, turn_info)
    elif plan == "B":
        return compute_reward_plan_b(prediction, ground_truth, messages, turn_info)
    elif plan == "C":
        return compute_reward_plan_c(prediction, ground_truth, messages, turn_info)
    else:
        raise ValueError(f"Unknown plan: '{plan}'. Choose A, B, or C.")


# ── Parse helpers ─────────────────────────────────────────────────────────────
def parse_tool_call(text: str):
    """Parse <tool_call>...</tool_call> from generated assistant text.

    Returns:
        (queries, valid_tool_call)
        - queries: list of query strings (may be empty []) or None if no tool call
        - valid_tool_call: True if found a <tool_call> block with parseable JSON
          containing a 'query' field that is a list. Used for r_format scoring.
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None, False

    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        queries = obj.get("arguments", {}).get("query", [])
        if isinstance(queries, list):
            return queries, True   # valid JSON with "query" list (may be empty)
        return [], False            # "query" field is not a list
    except Exception:
        # Fallback: try to extract the query array with regex
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if m2:
            try:
                queries = json.loads(m2.group(1))
                if isinstance(queries, list):
                    return queries, True
            except Exception:
                pass
    return None, False              # <tool_call> present but JSON malformed


def extract_answer(text: str):
    """Extract the final answer from <|box_start|>...<|box_end|> tags."""
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Web Information Seeking Master. Your task is to thoroughly seek "
    "the internet for information and provide accurate answers to questions. "
    "No matter how complex the query, you will not give up until you find the "
    "corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n"
    "1. **Persistent Actions for Answers**: You will engage in many interactions, "
    "delving deeply into the topic to explore all possible aspects until a satisfactory "
    "answer is found.\n\n"
    "2. **Repeated Verification**: Before presenting a Final Answer, you will "
    "**cross-check** and **validate the information** you've gathered to confirm its "
    "accuracy and reliability.\n\n"
    "3. **Attention to Detail**: You will carefully analyze each information source "
    "to ensure that all data is current, relevant, and from credible origins."
)

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


# ── Block diffusion generation (LLaDA MDLM style) ────────────────────────────
@torch.no_grad()
def denoise_block_sample(model, input_ids, block_start, block_end,
                         mask_id, num_steps, temperature=1.0):
    """Iteratively reveal masked tokens in a single block via confidence-based unmasking.

    Each step:
      1. Forward pass → logits for the block
      2. Identify still-masked positions
      3. Sample token ids with softmax(logits/temperature)
      4. Unmask the top n_reveal positions by confidence
    """
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = input_ids[0, block_start:block_end] == mask_id
        if still_masked.sum() == 0:
            break
        if temperature != 1.0:
            logits = logits / temperature
        probs      = torch.softmax(logits.float(), dim=-1)
        pred_ids   = torch.multinomial(probs, 1).squeeze(-1)
        confidence = probs.max(dim=-1).values

        n_reveal = max(1, round((block_end - block_start) / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0          # don't re-reveal already-revealed
        n_reveal = min(n_reveal, still_masked.sum().item())
        top_idx  = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]
    return input_ids


@torch.no_grad()
def llada_generate_sample(model, tokenizer, prompt_ids, mask_id,
                          block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                          max_blocks=MAX_BLOCKS, stop_strings=None,
                          temperature=TEMPERATURE):
    """Generate text block by block using LLaDA masked diffusion.

    Each block of `block_size` tokens is denoised jointly over `num_steps` steps,
    then appended to the sequence. Generation stops at any stop_string or EOS.
    """
    device        = next(model.parameters()).device
    eos_id        = tokenizer.eos_token_id or 0
    stop_strings  = stop_strings or []
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

        # Check stop strings
        for stop in stop_strings:
            if stop in current_text:
                return current_text[: current_text.index(stop) + len(stop)]

        # Check EOS: if last 8 tokens are all EOS, stop
        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Single agentic rollout ────────────────────────────────────────────────────
def run_one_rollout(model, tokenizer, mask_id, question: str,
                    temperature: float = TEMPERATURE) -> dict:
    """Run one complete agentic rollout for a question.

    The rollout follows: think → tool_call → tool_response → ... → answer
    For up to MAX_TURNS search cycles.

    Returns:
        {
          "prediction": str or None,
          "messages":   list of {role, content} dicts (full conversation),
          "termination": "answer" | "max_turns",
          "turn_info":  list of per-search-turn dicts for reward computation,
        }

    turn_info entries (only for turns that invoked a search):
        {
          "turn_idx":       int   (rollout loop index),
          "tool_call_text": str   (the full assistant text containing <tool_call>),
          "queries":        list  (list of search query strings),
          "search_result":  str   (the concatenated search API response),
          "valid_tool_call": bool (whether JSON parsed successfully),
        }
    """
    stop_strings = ["</tool_call>", "<|box_end|>"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prediction = None
    turn_info  = []  # accumulates one entry per search turn

    for turn in range(MAX_TURNS):
        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids
        new_text = llada_generate_sample(
            model, tokenizer, prompt_ids, mask_id,
            stop_strings=stop_strings, temperature=temperature,
        )
        messages.append({"role": "assistant", "content": new_text.strip()})

        # Check if the model gave a final answer
        answer = extract_answer(new_text)
        if answer:
            prediction = answer
            break

        # Check if the model made a tool call
        queries, valid_tc = parse_tool_call(new_text)

        if queries:  # non-None and non-empty list → perform search
            search_result = run_search(queries)
            tool_resp_content = f"<tool_response>\n{search_result}\n</tool_response>"
            messages.append({"role": "user", "content": tool_resp_content})

            # Extend context for next generation
            tool_resp_formatted = (
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{tool_resp_content}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            full_context += new_text + tool_resp_formatted

            # Record this search turn for reward computation
            turn_info.append({
                "turn_idx":        turn,
                "tool_call_text":  new_text.strip(),
                "queries":         queries,
                "search_result":   search_result,
                "valid_tool_call": True,  # must be True if queries is truthy
            })

        else:
            # No search performed this turn (either no tool call, or empty queries,
            # or malformed JSON). Still track for format reward if JSON was valid.
            full_context += new_text
            if valid_tc:
                # Valid JSON structure but empty query array
                turn_info.append({
                    "turn_idx":        turn,
                    "tool_call_text":  new_text.strip(),
                    "queries":         queries if queries is not None else [],
                    "search_result":   "",
                    "valid_tool_call": True,
                })

    return {
        "prediction":  prediction,
        "messages":    messages,
        "termination": "answer" if prediction else "max_turns",
        "turn_info":   turn_info,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Collect on-policy agentic rollouts with Plan A/B/C reward."
    )
    parser.add_argument(
        "--plan", choices=["A", "B", "C"], default="A",
        help=(
            "Reward plan: "
            "A = F1×efficiency+format_bonus (hermes), "
            "B = multi-component (achilleas), "
            "C = SALT-lite with turn_credits (homer)"
        ),
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Path to the SFT checkpoint to roll out from")
    parser.add_argument("--input",       default=DEFAULT_INPUT,
                        help="JSONL file with {question, answer} records")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT,
                        help="Output directory; plan subdir created automatically")
    parser.add_argument("--n_rolls",     type=int,   default=8,
                        help="Number of rollouts per question")
    parser.add_argument("--n_questions", type=int,   default=0,
                        help="How many questions to process (0 = all)")
    parser.add_argument("--rank",        type=int,   default=0,
                        help="GPU worker rank (for data sharding)")
    parser.add_argument("--world",       type=int,   default=8,
                        help="Total number of GPU workers")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature for diffusion denoising")
    parser.add_argument("--block_size",  type=int,   default=BLOCK_SIZE,
                        help="Block size for diffusion rollout（需与 SFT 训练一致）")
    parser.add_argument("--num_steps",   type=int,   default=NUM_STEPS,
                        help="每个 block 的 denoising steps")
    args = parser.parse_args()

    global BLOCK_SIZE, NUM_STEPS
    BLOCK_SIZE = args.block_size
    NUM_STEPS  = args.num_steps

    # ── Load data ──────────────────────────────────────────────────────────────
    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.n_questions > 0:
        samples = samples[: args.n_questions]
    # Shard by rank: rank 0 gets indices 0, world, 2*world, ...
    samples = samples[args.rank :: args.world]
    print(
        f"[Plan {args.plan}] Rank {args.rank}/{args.world}: "
        f"{len(samples)} questions × {args.n_rolls} rolls"
    )

    # ── Prepare output file ────────────────────────────────────────────────────
    out_dir  = Path(args.output) / f"plan_{args.plan.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rank{args.rank}.jsonl"

    # Resume support: skip already-completed (question, roll_idx) pairs
    done = set()
    if out_file.exists():
        for line in open(out_file):
            try:
                d = json.loads(line)
                done.add((d["question"], d["roll_idx"]))
            except Exception:
                pass
        print(f"Resuming: {len(done)} records already done")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"mask_id={mask_id}  temperature={args.temperature}  plan={args.plan}\n")

    # ── Rollout loop ───────────────────────────────────────────────────────────
    with open(out_file, "a") as f:
        for sample in tqdm(samples, desc=f"rank{args.rank}"):
            q = sample["question"]
            a = sample["answer"]

            for roll_idx in range(args.n_rolls):
                if (q, roll_idx) in done:
                    continue

                try:
                    result = run_one_rollout(
                        model, tokenizer, mask_id, q, args.temperature
                    )
                except Exception as e:
                    tqdm.write(f"[ERROR] roll={roll_idx} q={q[:50]}: {e}")
                    result = {
                        "prediction":  None,
                        "messages":    [],
                        "termination": "error",
                        "turn_info":   [],
                    }

                reward = compute_reward(
                    args.plan,
                    result["prediction"],
                    a,
                    result["messages"],
                    result["turn_info"],
                )

                record = {
                    "question":    q,
                    "answer":      a,
                    "roll_idx":    roll_idx,
                    "plan":        args.plan,
                    "prediction":  result["prediction"],
                    "messages":    result["messages"],
                    "termination": result["termination"],
                    "reward":      reward,
                    "n_searches":  count_searches(result["messages"]),
                }

                # Plan B: save component breakdown for offline analysis
                if args.plan == "B":
                    record["reward_breakdown"] = get_plan_b_breakdown(
                        result["prediction"], a,
                        result["messages"], result["turn_info"],
                    )

                # Plan C: save per-turn credits for SALT-lite downstream processing
                if args.plan == "C":
                    record["turn_credits"] = compute_turn_credits(a, result["turn_info"])

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\nDone. Output: {out_file}")


if __name__ == "__main__":
    main()
