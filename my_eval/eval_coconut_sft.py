#!/usr/bin/env python3
"""
eval_coconut_sft.py — 评估 COCONUT-SFT 微调后的 LLaDA 模型

推理流程（每轮）：
  Phase 1: 自蒸馏 latent 生成
    input_ids = [context | MASK×N]
    H_lat = model.hidden_states[-1][L_ctx : L_ctx+N]   (N, d_model)

  Phase 2: Block diffusion（注入 H_lat）
    embeddings[L_ctx : L_ctx+N] = H_lat
    逐 block 迭代去噪 → action text

用法:
  python my_eval/eval_coconut_sft.py \
    --model /common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_stage1/checkpoint-180 \
    --input Dataroller/data/sft_train_200.jsonl \
    --output my_eval/results/coconut_sft_ckpt180.jsonl \
    --max_samples 200
"""

import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 超参 ────────────────────────────────────────────────────────────────────────
N_LATENT   = 64    # 与训练一致
BLOCK_SIZE = 64
NUM_STEPS  = 64    # 每个 block 的去噪步数
MAX_BLOCKS = 20    # 最多生成 20 个 block per turn (~1280 tokens)
MAX_TURNS  = 6     # 最多 6 轮工具调用

DEFAULT_MODEL  = (
    "/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_stage1/checkpoint-180"
)
DEFAULT_INPUT  = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data/sft_train_200.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/coconut_sft_ckpt180.jsonl"
)

# ── Search API ──────────────────────────────────────────────────────────────────
def _load_config():
    for p in [
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

_CFG = _load_config()
GOOGLE_KEY  = os.getenv("GOOGLE_SEARCH_KEY") or _CFG.get("google_search_key", "")
SEARCH_URL  = "https://google.serper.dev/search"


def google_search(query: str) -> str:
    if not GOOGLE_KEY:
        return f"[Search unavailable — no API key] query: {query}"
    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(SEARCH_URL, headers=headers,
                             json={"q": query, "num": 5}, timeout=15)
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
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)


# ── Parse helpers ───────────────────────────────────────────────────────────────
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


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Web Information Seeking Master. Your task is to thoroughly seek the "
    "internet for information and provide accurate answers to questions. No matter how "
    "complex the query, you will not give up until you find the corresponding information.\n\n"
    "As you proceed, adhere to the following principles:\n\n"
    "1. **Persistent Actions for Answers**: You will engage in many interactions, delving "
    "deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n"
    "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** "
    "and **validate the information** you've gathered to confirm its accuracy and reliability.\n\n"
    "3. **Attention to Detail**: You will carefully analyze each information source to ensure "
    "that all data is current, relevant, and from credible origins."
)

USER_TEMPLATE = (
    'A conversation between User and Assistant. The user asks a question, and the assistant '
    'solves it by calling one or more of the following tools.\n'
    '<tools>\n'
    '{\n'
    '  "name": "search",\n'
    '  "description": "Performs batched web searches: supply an array \'query\'; the tool '
    'retrieves the top 10 results for each query in one call.",\n'
    '  "parameters": {\n'
    '    "type": "object",\n'
    '    "properties": {\n'
    '      "query": {\n'
    '        "type": "array",\n'
    '        "items": {"type": "string"},\n'
    '        "description": "Array of query strings. Include 3 or less search queries in a single call."\n'
    '      }\n'
    '    },\n'
    '    "required": ["query"]\n'
    '    }\n'
    '}\n'
    '</tools>\n\n'
    'The assistant starts with one or more cycles of (thinking about which tool to use -> '
    'performing tool call -> waiting for tool response), and ends with answer of the question.\n\n'
    'Example response:\n'
    '<think>\nthinking process here\n</think>\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": ["query string 1"]}}\n</tool_call>\n'
    '<tool_response>\ntool_response here\n</tool_response>\n'
    '<think>\nthinking process here\n</think>\n'
    '<|box_start|>\nanswer here\n<|box_end|>\n'
    'The assistant must strictly abide by the above format.\nUser: '
)


# ── COCONUT 推理核心 ────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_latent(model, embed_layer, context_ids, mask_id, N):
    """Phase 1: [context | MASK×N] → hidden_states at MASK positions."""
    device = context_ids.device
    L_ctx = context_ids.shape[1]
    lat_ph = torch.full((1, N), mask_id, dtype=torch.long, device=device)
    input_ids = torch.cat([context_ids, lat_ph], dim=1)
    out = model(input_ids=input_ids, output_hidden_states=True)
    H_lat = out.hidden_states[-1][0, L_ctx : L_ctx + N].detach()  # (N, d_model)
    return H_lat


@torch.no_grad()
def denoise_block(model, embed_layer, action_ctx_embeds, block_start, block_end,
                  mask_id, num_steps, H_lat=None, lat_slice=None):
    """
    Block diffusion。H_lat 不为 None 时通过 embed hook 注入 latent（Turn 1）；
    H_lat 为 None 时直接做普通 block diffusion（Turn 2+）。
    action_ctx_embeds: (1, L, d_model)，最后 block_size 个位置为 MASK。
    """
    block_len = block_end - block_start
    input_embeds = action_ctx_embeds.clone()

    device = input_embeds.device
    block_ids = torch.full((block_len,), mask_id, dtype=torch.long, device=device)

    for _ in range(num_steps):
        if H_lat is not None:
            def _hook(module, args, out):
                o = out.clone()
                o[0, lat_slice] = H_lat.to(o.dtype)
                return o
            h = embed_layer.register_forward_hook(_hook)

        out = model(inputs_embeds=input_embeds)

        if H_lat is not None:
            h.remove()

        block_logits = out.logits[0, block_start:block_end]
        probs        = torch.softmax(block_logits.float(), dim=-1)
        pred_ids     = probs.argmax(dim=-1)
        confidence   = probs.max(dim=-1).values

        still_masked = (block_ids == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0:
            break

        n_reveal = max(1, round(block_len / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, n_still)
        top_idx  = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            block_ids[idx] = pred_ids[idx]

        # 用新 token 更新 input_embeds 中的 block 部分
        new_block_embeds = embed_layer(block_ids.unsqueeze(0)).detach()
        input_embeds[0, block_start:block_end] = new_block_embeds[0]

    return block_ids  # (block_len,) token ids


def coconut_generate(model, tokenizer, embed_layer, context_ids, mask_id, d_model,
                     n_latent=N_LATENT, block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                     max_blocks=MAX_BLOCKS, stop_strings=None):
    """
    COCONUT 两阶段生成：
      1. 自蒸馏 latent（Phase 1）
      2. Block diffusion + latent 注入（Phase 2）
    返回生成的 action 文本。
    """
    device = context_ids.device
    L_ctx  = context_ids.shape[1]
    stop_strings = stop_strings or []
    eos_id = tokenizer.eos_token_id or 0

    # Phase 1: latent 生成
    H_lat = generate_latent(model, embed_layer, context_ids, mask_id, n_latent)
    # H_lat: (N, d_model)

    # Phase 2: block diffusion
    # 构建初始 action_ctx_embeds = embed([context | MASK×N])
    lat_ph = torch.full((1, n_latent), mask_id, dtype=torch.long, device=device)
    base_ids   = torch.cat([context_ids, lat_ph], dim=1)        # (1, L_ctx+N)
    base_embeds = embed_layer(base_ids).detach().clone()          # (1, L_ctx+N, d)
    base_embeds[0, L_ctx : L_ctx + n_latent] = H_lat.to(base_embeds.dtype)

    lat_slice   = slice(L_ctx, L_ctx + n_latent)
    action_embeds = base_embeds                                    # grows each block
    generated_ids = []

    for blk in range(max_blocks):
        # 追加新的 MASK block
        mask_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        new_embeds  = embed_layer(mask_block).detach().clone()
        action_embeds = torch.cat([action_embeds, new_embeds], dim=1)

        cur_len     = action_embeds.shape[1]
        block_start = cur_len - block_size
        block_end   = cur_len

        block_ids = denoise_block(
            model, embed_layer, action_embeds,
            block_start, block_end,
            mask_id, num_steps, H_lat, lat_slice,
        )

        # 用解码后的 token 更新 action_embeds
        decoded_embeds = embed_layer(block_ids.unsqueeze(0)).detach()
        action_embeds[0, block_start:block_end] = decoded_embeds[0]

        generated_ids.extend(block_ids.tolist())
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for s in stop_strings:
            if s in current_text:
                current_text = current_text[:current_text.index(s) + len(s)]
                return current_text

        if all(t == eos_id for t in block_ids[-8:].tolist()):
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def standard_generate(model, tokenizer, embed_layer, context_ids, mask_id,
                      block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                      max_blocks=MAX_BLOCKS, stop_strings=None):
    """
    普通 block diffusion，不走 latent 阶段。
    用于 Turn 2+ 的生成（训练时这些轮次也没有独立的 latent）。
    """
    device = context_ids.device
    stop_strings = stop_strings or []
    eos_id = tokenizer.eos_token_id or 0

    action_embeds = embed_layer(context_ids).detach().clone()  # (1, L_ctx, d)
    generated_ids = []

    for blk in range(max_blocks):
        mask_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        new_embeds  = embed_layer(mask_block).detach().clone()
        action_embeds = torch.cat([action_embeds, new_embeds], dim=1)

        cur_len     = action_embeds.shape[1]
        block_start = cur_len - block_size
        block_end   = cur_len

        block_ids = denoise_block(
            model, embed_layer, action_embeds,
            block_start, block_end,
            mask_id, num_steps,        # H_lat=None → 不注入
        )

        decoded_embeds = embed_layer(block_ids.unsqueeze(0)).detach()
        action_embeds[0, block_start:block_end] = decoded_embeds[0]

        generated_ids.extend(block_ids.tolist())
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for s in stop_strings:
            if s in current_text:
                current_text = current_text[:current_text.index(s) + len(s)]
                return current_text

        if all(t == eos_id for t in block_ids[-8:].tolist()):
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 单题 agentic rollout ─────────────────────────────────────────────────────────
def run_one(model, tokenizer, embed_layer, mask_id, d_model, question: str) -> dict:
    stop_strings = ["</tool_call>", "<|box_end|>"]
    device = next(model.parameters()).device

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE + question},
    ]

    prediction         = None
    termination_reason = "max_turns"
    full_context       = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for turn in range(MAX_TURNS):
        ctx_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        try:
            if turn == 0:
                # Turn 1：走 latent think（与训练结构一致）
                action_text = coconut_generate(
                    model, tokenizer, embed_layer, ctx_ids, mask_id, d_model,
                    stop_strings=stop_strings,
                )
            else:
                # Turn 2+：直接 block diffusion，不走 latent（训练时这些轮次无独立 latent）
                action_text = standard_generate(
                    model, tokenizer, embed_layer, ctx_ids, mask_id,
                    stop_strings=stop_strings,
                )
        except Exception as e:
            tqdm.write(f"  [ERROR] turn={turn}: {e}")
            import traceback; traceback.print_exc()
            break

        # 去掉 action_text 开头的乱码 assistant header（block diffusion 训练格式
        # 导致模型输出 <|im_start|>assistant\n<think>... 而 full_context 末尾已有该 header）
        m = re.search(r'<think>|<tool_call>|<\|box_start\|>', action_text)
        clean_action = action_text[m.start():] if (m and m.start() < 200) else action_text
        clean_action = clean_action.strip()

        messages.append({"role": "assistant", "content": clean_action})

        answer = extract_answer(action_text)
        if answer:
            prediction         = answer
            termination_reason = "answer"
            break

        queries = parse_tool_call(action_text)
        if queries:
            search_result = run_search(queries)
            tool_resp = f"<tool_response>\n{search_result}\n</tool_response>"
            messages.append({"role": "user", "content": tool_resp})

        # 每轮结束后用 apply_chat_template 重建 full_context，避免手动拼接引入重复 header
        full_context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return {
        "prediction":         prediction,
        "num_turns":          turn + 1,
        "termination_reason": termination_reason,
        "messages":           messages,
    }


# ── 主函数 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--input",       default=DEFAULT_INPUT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--offset",      type=int, default=0)
    parser.add_argument("--n_latent",    type=int, default=N_LATENT)
    parser.add_argument("--num_steps",   type=int, default=NUM_STEPS)
    parser.add_argument("--max_blocks",  type=int, default=MAX_BLOCKS)
    args = parser.parse_args()

    # 加载数据
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.offset > 0:
        samples = samples[args.offset:]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"数据: {len(samples)} 题  offset={args.offset}")

    # 续写：跳过已完成的题
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    pass
        print(f"已完成 {len(done)} 题，继续剩余")
    todo = [s for s in samples if s["question"] not in done]
    if not todo:
        print("全部完成！")
        return

    # 加载模型（单卡 bf16）
    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # embed 层路径（与训练器一致：model.model.transformer.wte）
    embed_layer = model.model.transformer.wte
    mask_id = getattr(model.config, "mask_token_id", 126336)
    d_model = getattr(model.config, "hidden_size",
               getattr(model.config, "d_model", 4096))

    print(f"mask_id={mask_id}  d_model={d_model}  n_latent={args.n_latent}")
    print(f"num_steps={args.num_steps}  max_blocks={args.max_blocks}")
    print(f"待评测: {len(todo)} 题\n")

    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="COCONUT-SFT eval"):
            question  = sample["question"]
            gt_answer = sample.get("answer", "")

            try:
                result = run_one(model, tokenizer, embed_layer, mask_id, d_model, question)
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

    print(f"\n完成！结果: {out_path}")
    print(f"打分: python my_eval/cal_acc.py --data {out_path} --no_llm_judge")


if __name__ == "__main__":
    main()
