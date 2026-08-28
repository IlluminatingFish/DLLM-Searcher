#!/usr/bin/env python3
"""
LLaDA 潜在推理（Latent Reasoning）评测脚本。

推理流程（每轮）：
  1. Think 去噪阶段：给定当前上下文，在 embedding 空间对 N_THINK 个位置
     从随机噪声迭代去噪（用 x0_head），得到连续的 think latents。
  2. Action 生成阶段：把 think latents 通过 embedding hook 注入模型，
     用标准 LLaDA block diffusion 生成 <tool_call> 或 <|box_start|>answer<|box_end|>。

与 run_llada_eval.py 的区别：
  - think span 不是离散 token，是连续 embedding（模型永远不需要"解码" think）
  - 需要 x0_head.pt 和 sigma_proj.pt（来自 x0pred checkpoint）
  - full_context 改为用 tensor 追踪（含 think 位置的 placeholder token）

用法:
  python my_eval/run_llada_latent_eval.py \
    --model  dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized \
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
    --output dLLM_trainer/VRPO/output/llada_eval/latent_ckpt6_100.jsonl \
    --max_samples 100

  python my_eval/cal_acc.py --data dLLM_trainer/VRPO/output/llada_eval/latent_ckpt6_100.jsonl
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
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import transformers.modeling_utils as _mu

# transformers 5.x 兼容补丁：caching_allocator_warmup 在加载时访问 all_tied_weights_keys
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

# ── 超参 ──────────────────────────────────────────────────────────────────────
N_THINK        = 64    # think span 的 latent token 数
N_DENOISE      = 20    # 去噪步数（sigma_max → sigma_min）
SIGMA_MAX      = 1.0
SIGMA_MIN      = 0.1
BLOCK_SIZE     = 64
NUM_STEPS      = 64
MAX_BLOCKS     = 16
MAX_TURNS      = 5

DEFAULT_MODEL  = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
)
DEFAULT_INPUT  = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/output/llada_eval/latent_ckpt6_100.jsonl"
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
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
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


# ── 核心：连续 think 去噪 → action 生成 ─────────────────────────────────────
@torch.no_grad()
def denoise_think(
    model, x0_head, sigma_proj,
    context_ids,        # (1, L) — prompt 上下文（已含 <think> token）
    L_think_start,      # think 占位符开始的绝对位置
    n_think,            # think latent 的 token 数
    d_model,
    n_steps=N_DENOISE,
    sigma_max=SIGMA_MAX,
    sigma_min=SIGMA_MIN,
):
    """
    从纯高斯噪声出发，迭代去噪，返回 (1, n_think, d_model) 的干净 think latents。
    """
    device = context_ids.device
    embed_layer = model.model.transformer.wte
    ln_f        = model.model.transformer.ln_f

    # 初始化为 sigma_max 级别的高斯噪声
    think_latents = sigma_max * torch.randn(
        1, n_think, d_model, dtype=torch.bfloat16, device=device
    )

    sigmas = torch.linspace(sigma_max, sigma_min, n_steps + 1,
                            device=device, dtype=torch.bfloat16)

    for t in range(n_steps):
        sigma_t    = sigmas[t]
        sigma_next = sigmas[t + 1]

        _fh = {}
        _lt = think_latents   # 当前 noisy latents
        _st = sigma_t

        def _inject(module, args, out, lt=_lt, st=_st):
            sig_emb = sigma_proj(st.view(1, 1).to(out.dtype)).view(1, 1, d_model)
            o = out.clone()
            o[0, L_think_start:L_think_start+n_think] = lt[0] + sig_emb.squeeze(0)
            return o

        def _cap(module, args, out):
            _fh["h"] = out

        h1 = embed_layer.register_forward_hook(_inject)
        h2 = ln_f.register_forward_hook(_cap)
        model(input_ids=context_ids)
        h1.remove()
        h2.remove()

        # x0_head 预测干净 embedding
        x0_pred = x0_head(_fh["h"][0, L_think_start:L_think_start+n_think])

        # DDPM 去噪步：x_{t-1} = x0 + sigma_next * new_noise
        if sigma_next > 1e-4:
            think_latents = x0_pred.unsqueeze(0) + sigma_next * torch.randn_like(think_latents)
        else:
            think_latents = x0_pred.unsqueeze(0)

    return think_latents  # (1, n_think, d_model)


@torch.no_grad()
def denoise_block_with_latents(
    model, input_ids,
    block_start, block_end,
    mask_id,
    embed_layer,
    think_latents,    # (1, n_think, d_model)
    think_pos_slice,  # slice 对象，指向 input_ids 中 think 占位符的绝对位置
    num_steps=NUM_STEPS,
):
    """标准 block diffusion，但加了 embedding hook 把 think latents 注入 think 位置。"""
    block_len = block_end - block_start

    def _make_hook(lt, pos):
        def _h(module, args, out):
            o = out.clone()
            o[0, pos] = lt[0].to(o.dtype)
            return o
        return _h

    for _ in range(num_steps):
        h = embed_layer.register_forward_hook(
            _make_hook(think_latents, think_pos_slice)
        )
        logits = model(input_ids=input_ids).logits
        h.remove()

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
            input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


def latent_think_then_generate(
    model, x0_head, sigma_proj, tokenizer,
    context_ids,    # (1, L) — 当前轮次的 prompt tensor，每轮从文本重建，不含上轮 think 占位符
    mask_id,
    d_model,
    n_think=N_THINK,
    n_denoise=N_DENOISE,
    block_size=BLOCK_SIZE,
    num_steps=NUM_STEPS,
    max_blocks=MAX_BLOCKS,
    stop_strings=None,
):
    """
    两阶段生成：
      1. think 去噪：得到 (1, n_think, d_model) 的 latent think embeddings
      2. action 生成：注入 think latents，用标准 block diffusion 生成 tool_call/answer

    返回: action_text（只有 tool_call 或 answer 文本，不含 think 文本）
    多轮上下文通过文本 full_context 追踪，think latents 只在当轮内部使用。
    """
    device = context_ids.device
    embed_layer = model.model.transformer.wte

    # ── 在上下文末尾追加 <think> token ──────────────────────────────────────
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids   = tokenizer.encode("</think>",  add_special_tokens=False)

    think_start_tensor = torch.tensor([think_start_ids], dtype=torch.long, device=device)
    think_end_tensor   = torch.tensor([think_end_ids],   dtype=torch.long, device=device)

    # think 占位符（用 mask_id 占位，embedding hook 会替换为 latents）
    think_placeholder  = torch.full((1, n_think), mask_id, dtype=torch.long, device=device)

    # 含 <think> 的去噪上下文：[context | <think> | n_think × mask]
    denoise_ctx = torch.cat([context_ids, think_start_tensor, think_placeholder], dim=1)
    L_think_start = denoise_ctx.shape[1] - n_think  # think 占位符的起始位置

    # ── Phase 1: think 去噪 ──────────────────────────────────────────────────
    think_latents = denoise_think(
        model, x0_head, sigma_proj,
        denoise_ctx, L_think_start, n_think, d_model,
        n_steps=n_denoise,
    )

    # ── Phase 2: action 生成（注入 think latents） ───────────────────────────
    # 构建 action 生成的基础 context：
    # [context | <think> | n_think占位 | </think>]
    action_ctx = torch.cat([
        context_ids,
        think_start_tensor,
        think_placeholder,
        think_end_tensor,
    ], dim=1)

    # think 位置在 action_ctx 中的切片（随着 context 增长保持固定）
    think_pos = slice(L_think_start, L_think_start + n_think)

    generated_ids = []
    stop_strings = stop_strings or []
    eos_id = tokenizer.eos_token_id or 0

    for _ in range(max_blocks):
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        action_ctx  = torch.cat([action_ctx, new_block], dim=1)
        block_start = action_ctx.shape[1] - block_size
        block_end   = action_ctx.shape[1]

        action_ctx = denoise_block_with_latents(
            model, action_ctx,
            block_start, block_end,
            mask_id, embed_layer,
            think_latents, think_pos,
            num_steps=num_steps,
        )

        block_tokens = action_ctx[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for s in stop_strings:
            if s in current_text:
                current_text = current_text[:current_text.index(s) + len(s)]
                return current_text

        if all(t == eos_id for t in block_tokens[-8:]):
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 单题 agentic rollout（潜在推理版） ───────────────────────────────────────
def run_one(model, x0_head, sigma_proj, tokenizer, mask_id, d_model, question: str) -> dict:
    """
    多轮交互方式与原版 run_llada_eval.py 相同：
    - full_context 是纯文本字符串，每轮追加 action_text + tool_response
    - 每轮从 full_context 重新 tokenize 得到 prompt_ids（不含 think 占位符）
    - think latents 只在 latent_think_then_generate 内部临时存在，不写入上下文
    """
    stop_strings = ["</tool_call>", "<|box_end|>"]
    device = next(model.parameters()).device

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

        # 每轮从文本重建 prompt（干净，不含上轮 think 占位符）
        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        try:
            action_text = latent_think_then_generate(
                model, x0_head, sigma_proj, tokenizer,
                prompt_ids, mask_id, d_model,
                stop_strings=stop_strings,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] turn={turn}: {e}")
            break

        messages.append({"role": "assistant", "content": action_text.strip()})

        answer = extract_answer(action_text)
        if answer:
            prediction         = answer
            termination_reason = "answer"
            break

        queries = parse_tool_call(action_text)
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
            full_context += action_text + tool_response
        else:
            full_context += action_text

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
    parser.add_argument("--offset",      type=int, default=0,         help="从第几题开始（用于多进程并行）")
    parser.add_argument("--n_think",     type=int, default=N_THINK,   help="think latent 长度")
    parser.add_argument("--n_denoise",   type=int, default=N_DENOISE, help="去噪步数")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    # 先取 offset 之后的题目，再限制 max_samples
    if args.offset > 0:
        samples = samples[args.offset:]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"eval 数据: {len(samples)} 题  来自 {args.input}  offset={args.offset}")

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

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    mask_id = model.config.mask_token_id
    device  = next(model.parameters()).device

    d_model = getattr(model.config, "d_model",
              getattr(model.config, "hidden_size", 4096))

    # 加载 x0_head 和 sigma_proj
    x0_head = torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16).to(device)
    x0_head.load_state_dict(
        torch.load(Path(args.model) / "x0_head.pt", map_location=device)
    )
    x0_head.eval()

    sigma_proj = torch.nn.Sequential(
        torch.nn.Linear(1, d_model),
        torch.nn.SiLU(),
        torch.nn.Linear(d_model, d_model),
    ).to(dtype=torch.bfloat16, device=device)
    sigma_proj.load_state_dict(
        torch.load(Path(args.model) / "sigma_proj.pt", map_location=device)
    )
    sigma_proj.eval()

    print(f"模型加载完成  mask_id={mask_id}  d_model={d_model}")
    print(f"x0_head + sigma_proj 加载完成")
    print(f"n_think={args.n_think}  n_denoise={args.n_denoise}")
    print(f"待评测: {len(todo)} 题\n")

    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="LLaDA latent eval"):
            question  = sample["question"]
            gt_answer = sample["answer"]

            try:
                result = run_one(
                    model, x0_head, sigma_proj,
                    tokenizer, mask_id, d_model, question,
                )
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
    print("打分命令:")
    print(f"  python my_eval/cal_acc.py --data {out_path}")


if __name__ == "__main__":
    main()
