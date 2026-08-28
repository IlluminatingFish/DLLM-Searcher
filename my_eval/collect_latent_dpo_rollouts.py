#!/usr/bin/env python3
"""
Plan C 专用：用 latent inference 收集 DPO rollout 数据。

与 collect_dpo_rollouts.py 的区别：
  - Think 阶段用 DDPM 去噪（x0_head），在连续 embedding 空间完成
  - Action 阶段用标准 block diffusion
  - 产生的 think 轨迹质量更高（有目的性），给 DPO 提供更好的 chosen/rejected

用法:
  CUDA_VISIBLE_DEVICES=0 python collect_latent_dpo_rollouts.py --rank 0 --world 8 --n_rolls 4
"""

import argparse, json, os, re, requests, sys, torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 复用 latent eval 里的核心函数
sys.path.insert(0, str(Path(__file__).parent))
from run_llada_latent_eval import (
    latent_think_then_generate,
    parse_tool_call, extract_answer,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    N_THINK, N_DENOISE, SIGMA_MAX, SIGMA_MIN,
    BLOCK_SIZE, NUM_STEPS, MAX_BLOCKS, MAX_TURNS
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


# ── Single latent rollout ─────────────────────────────────────────────────────
def run_one_latent_rollout(model, x0_head, sigma_proj, tokenizer, mask_id, d_model, question):
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
                               add_special_tokens=False).input_ids.to(next(model.parameters()).device)
        try:
            action_text = latent_think_then_generate(
                model, x0_head, sigma_proj, tokenizer, prompt_ids,
                mask_id=mask_id, d_model=d_model,
                n_think=N_THINK, n_denoise=N_DENOISE,
                block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                max_blocks=MAX_BLOCKS, stop_strings=stop_strings
            )
        except Exception as e:
            print(f"[WARN] latent gen error: {e}")
            action_text = ""

        messages.append({"role": "assistant", "content": action_text.strip()})

        answer = extract_answer(action_text)
        if answer:
            prediction = answer
            break

        queries = parse_tool_call(action_text)
        if queries:
            search_result = run_search(queries)
            messages.append({"role": "user",
                             "content": f"<tool_response>\n{search_result}\n</tool_response>"})
            tool_resp = (f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                        f"<|im_end|>\n<|im_start|>assistant\n")
            full_context += action_text + tool_resp
        else:
            full_context += action_text

    return {"prediction": prediction, "messages": messages,
            "termination": "answer" if prediction else "max_turns"}


# ── Main ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--input",   default=DEFAULT_INPUT)
    parser.add_argument("--output",  default="output/dpo_rollouts_latent")
    parser.add_argument("--n_rolls", type=int, default=4)
    parser.add_argument("--rank",    type=int, default=0)
    parser.add_argument("--world",   type=int, default=8)
    args = parser.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    samples = samples[args.rank::args.world]
    print(f"Rank {args.rank}/{args.world}: {len(samples)} questions × {args.n_rolls} rolls (latent)")

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

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    mask_id = model.config.mask_token_id
    device  = next(model.parameters()).device
    d_model = getattr(model.config, "hidden_size", 4096)

    # 加载 x0pred 组件
    x0_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16).to(device)
    x0_head.load_state_dict(
        torch.load(Path(args.model) / "x0_head.pt", map_location=device)
    )
    x0_head.eval()

    sigma_proj = nn.Sequential(
        nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
    ).to(dtype=torch.bfloat16, device=device)
    sigma_proj.load_state_dict(
        torch.load(Path(args.model) / "sigma_proj.pt", map_location=device)
    )
    sigma_proj.eval()

    print(f"mask_id={mask_id}  x0_head=✓  sigma_proj=✓\n")

    with open(out_file, "a") as f:
        for sample in tqdm(samples, desc=f"latent rank{args.rank}"):
            q = sample["question"]
            a = sample["answer"]
            for roll_idx in range(args.n_rolls):
                if (q, roll_idx) in done:
                    continue
                try:
                    result = run_one_latent_rollout(
                        model, x0_head, sigma_proj, tokenizer, mask_id, d_model, q
                    )
                except Exception as e:
                    tqdm.write(f"[ERROR] {q[:40]}: {e}")
                    result = {"prediction": None, "messages": [], "termination": "error"}

                record = {
                    "question":    q, "answer": a,
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
