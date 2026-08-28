"""
Track token reveal order during dLLM (LLaDA block diffusion) inference.
For each denoising step, records which positions get unmasked and at what step.
Runs on N questions for two models and saves JSON for visualization.
"""
import json, re, sys, time, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
EVAL_INPUT  = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/eval_600.jsonl")
OUT_DIR     = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE  = 64
NUM_STEPS   = 64
MAX_BLOCKS  = 6      # track up to 6 blocks per question (enough for 1-2 turns)
N_QUESTIONS = 30

MODELS = {
    "sft_llada":    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized",
    "x0pred":       "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized",
    "on_policy_rl": "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/model",
    "offline_rl":   "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/offline_loop/offline_train/model/checkpoint-125",
}

# RL 模型的 tokenizer 用了不支持的 TokenizersBackend，改用 SFT tokenizer
_SFT_TOKENIZER = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
TOKENIZER_PATHS = {
    "on_policy_rl": _SFT_TOKENIZER,
    "offline_rl":   _SFT_TOKENIZER,
}

SYSTEM_PROMPT = (
    "You are a Web Information Seeking Master. Your task is to thoroughly seek the "
    "internet for information and provide accurate answers to questions. No matter how "
    "complex the query, you will not give up until you find the corresponding information.\n\n"
    "As you proceed, adhere to the following principles:\n\n"
    "1. **Persistent Actions for Answers**: You will engage in many interactions, delving "
    "deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n"
    "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** "
    "and **validate the information** from multiple angles or sources.\n\n"
    "3. **Accurate, Grounded Answers**: Do not rely on mere guesses. Always back up your "
    "answers with evidence drawn from reputable sources.\n\n"
    "4. **Structured Thinking**: Organize your thoughts and actions systematically to ensure "
    "a comprehensive and methodical approach to information gathering.\n\n"
    "Final answers should be placed between <|box_start|> and <|box_end|> tags."
)

TOOL_DESC = """{
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
}"""

STOP_STRINGS = ["<|box_end|>", "<|im_end|>"]

# ── denoise with tracking ─────────────────────────────────────────────────────
@torch.no_grad()
def denoise_block_tracked(model, input_ids, block_start, block_end, mask_id, num_steps):
    """
    Like denoise_block but also returns reveal_step: list[int] (length=block_len)
    where reveal_step[i] = step at which position i was unmasked (1-indexed).
    0 means already unmasked from prompt (shouldn't happen for fresh blocks).
    """
    block_len = block_end - block_start
    reveal_step = [0] * block_len  # 0 = not yet revealed (will be filled)

    for step in range(1, num_steps + 1):
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
            if reveal_step[pos] == 0:
                reveal_step[pos] = step

    return input_ids, reveal_step


@torch.no_grad()
def generate_tracked(model, tokenizer, mask_id, prompt_text, max_blocks=MAX_BLOCKS):
    """
    Generate with full reveal-order tracking.
    Returns list of block dicts, each containing tokens + reveal steps.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated_ids = []
    blocks_data = []

    for block_idx in range(max_blocks):
        new_block = torch.full((1, BLOCK_SIZE), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - BLOCK_SIZE
        block_end   = input_ids.shape[1]

        input_ids, reveal_step = denoise_block_tracked(
            model, input_ids, block_start, block_end, mask_id, NUM_STEPS)

        block_token_ids = input_ids[0, block_start:block_end].tolist()
        block_tokens    = [tokenizer.decode([t]) for t in block_token_ids]
        generated_ids.extend(block_token_ids)

        blocks_data.append({
            "block_idx":   block_idx,
            "token_ids":   block_token_ids,
            "tokens":      block_tokens,
            "reveal_step": reveal_step,
            "num_steps":   NUM_STEPS,
        })

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        stopped = False
        for stop in STOP_STRINGS:
            if stop in current_text:
                stopped = True
                break
        if stopped:
            break

        tail = block_token_ids[-8:]
        if all(t == eos_id for t in tail):
            break

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return full_text, blocks_data


def build_prompt(question):
    user_content = (
        f"A conversation between User and Assistant. The user asks a question, and the "
        f"assistant solves it by calling one or more of the following tools.\n<tools>\n"
        f"{TOOL_DESC}\n</tools>\n\n"
        "The assistant starts with one or more cycles of (thinking about which tool to use "
        "-> performing tool call -> waiting for tool response), and ends with answer of the question.\n\n"
        "Example response:\n<think>\nthinking process here\n</think>\n"
        "<tool_call>\n{{\"name\": \"search\", \"arguments\": {{\"query\": [\"query string 1\", \"query string 2\"]}}}}\n</tool_call>\n"
        "<tool_response>\ntool_response here\n</tool_response>\n"
        "<think>\nthinking process here\n</think>\n"
        "<|box_start|>\nanswer here\n<|box_end|>\n"
        "The assistant must strictly abide by the above format.\n"
        f"User: {question}"
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--n",          type=int, default=N_QUESTIONS)
    parser.add_argument("--offset",     type=int, default=0)
    parser.add_argument("--out_path",   type=str, default=None,
                        help="Override output file path (default: OUT_DIR/{model_name}_order.json)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model_path = MODELS[args.model_name]

    print(f"[{args.model_name}] Loading from {model_path} → {device}")
    tok_path = TOKENIZER_PATHS.get(args.model_name, model_path)
    if tok_path != model_path:
        print(f"  tokenizer override → {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    mask_id = model.config.mask_token_id
    print(f"  mask_id = {mask_id}")

    questions = []
    with open(EVAL_INPUT) as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    questions = questions[args.offset: args.offset + args.n]
    print(f"  Running {len(questions)} questions")

    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        prompt = build_prompt(q["question"])
        try:
            full_text, blocks = generate_tracked(model, tokenizer, mask_id, prompt)
        except Exception as ex:
            print(f"  [Q{i}] ERROR: {ex}")
            continue
        elapsed = time.time() - t0
        print(f"  [Q{i+1}/{len(questions)}] {elapsed:.1f}s  blocks={len(blocks)}  "
              f"q={q['question'][:60]}")
        results.append({
            "question":  q["question"],
            "answer":    q.get("answer", ""),
            "source":    q.get("source", ""),
            "full_text": full_text,
            "blocks":    blocks,
        })

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = OUT_DIR / f"{args.model_name}_order.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}  ({len(results)} questions)")


if __name__ == "__main__":
    main()
