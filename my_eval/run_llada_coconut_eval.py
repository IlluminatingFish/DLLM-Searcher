#!/usr/bin/env python3
"""
LLaDA COCONUT 风格潜在推理评测脚本。

推理流程（每轮）：
  1. Continuous Thought 链式生成（N_THOUGHTS 步）：
       step 0: forward([context | <think> | MASK])
               → ln_f 输出在 MASK 位置的 hidden state = c₀
       step 1: forward([context | <think> | inject(c₀) | MASK])
               → hidden[MASK] = c₁
       ...
       step k-1: forward([... | inject(c₀,...,c_{k-2}) | MASK])
               → c_{k-1}
     最终得到 k 个 continuous thought 向量 [c₀, c₁, ..., c_{k-1}]

  2. Action 生成（标准 LLaDA block diffusion）：
       input = [context | <think> | MASK×k | </think> | MASK×64]
       embed hook 把 MASK×k 替换成 [c₀,...,c_{k-1}]
       → block diffusion → tool_call 或 box answer

与原 run_llada_latent_eval.py 的区别：
  - 不用 x0_head / sigma_proj / DDPM 去噪
  - Thoughts 来自模型自身 ln_f 输出的 hidden state（不需要额外 head）
  - 链式生成：c_t 由 [context + c_0,...,c_{t-1}] 决定，每个 thought 建立在前一个基础上
  - 不需要外部 checkpoint 文件（x0_head.pt / sigma_proj.pt）

用法:
  python my_eval/run_llada_coconut_eval.py \\
    --model  dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized \\
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \\
    --output dLLM_trainer/VRPO/output/llada_eval/coconut_ckpt6_100.jsonl \\
    --max_samples 100
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

# ── transformers 5.x 兼容补丁 ────────────────────────────────────────────────
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
N_THOUGHTS  = 16    # continuous thought 步数（每步一次完整 forward pass）
BLOCK_SIZE  = 64
NUM_STEPS   = 64
MAX_BLOCKS  = 16
MAX_TURNS   = 5

DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/output/llada_eval/coconut_ckpt6_100.jsonl"
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


# ── Prompt templates（与 run_llada_eval.py 完全相同） ─────────────────────────
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

USER_PROMPT_TEMPLATE = (
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
    '        "items": {\n'
    '          "type": "string"\n'
    '        },\n'
    '        "description": "Array of query strings. Include 3 or less search queries in a single call."\n'
    '      }\n'
    '    },\n'
    '    "required": [\n'
    '      "query"\n'
    '    ]\n'
    '    }\n'
    '}\n'
    '</tools>\n\n'
    'The assistant starts with one or more cycles of (thinking about which tool to use -> '
    'performing tool call -> waiting for tool response), and ends with answer of the question.\n\n'
    'Example response:\n'
    '<think>\n'
    'thinking process here\n'
    '</think>\n'
    '<tool_call>\n'
    '{"name": "search", "arguments": {"query": ["query string 1", "query string 2"]}}\n'
    '</tool_call>\n'
    '<tool_response>\n'
    'tool_response here\n'
    '</tool_response>\n'
    '<think>\n'
    'thinking process here\n'
    '</think>\n'
    '<|box_start|>\n'
    'answer here\n'
    '<|box_end|>\n'
    'The assistant must strictly abide by the above format.\n'
    'User: '
)


# ── 核心：COCONUT 链式 continuous thought 生成 ────────────────────────────────
@torch.no_grad()
def generate_continuous_thoughts(
    model,
    context_ids,      # (1, L) — [prompt | <think>]，thought 从 L 位置开始追加
    L_thought_start,  # thought 占位符在序列中的绝对起始位置（= context_ids.shape[1]）
    n_thoughts,
    d_model,
    mask_id,
):
    """
    COCONUT 链式生成：每步 forward 一次，取 thought 位置的 ln_f hidden state。

    step t：
      input_ids  = [context | MASK × (t+1)]
      embed hook : 把位置 [L, L+1, ..., L+t-1] 替换为已有的 thoughts [c₀,...,c_{t-1}]
                   位置 L+t 保持 MASK embedding（让模型在此位置自由聚合信息）
      forward    → ln_f 输出 → 取位置 L+t 的 hidden state = cₜ

    返回: thoughts，shape = (n_thoughts, d_model)
    """
    device = context_ids.device
    embed_layer = model.model.transformer.wte
    ln_f        = model.model.transformer.ln_f

    thoughts = []  # 每个元素 shape = (d_model,)

    for t in range(n_thoughts):
        # input_ids：context + (t+1) 个 MASK 占位符
        placeholders = torch.full((1, t + 1), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([context_ids, placeholders], dim=1)

        _fh = {}
        # 快照前 t 个 thoughts（避免 Python 闭包捕获引用问题）
        _prev = [th.clone() for th in thoughts]

        def _inject(module, args, out, prev=_prev, ts=L_thought_start):
            o = out.clone()
            # 把前 t 个位置的 MASK embedding 替换成已知的 thought 向量
            for i, th in enumerate(prev):
                o[0, ts + i] = th.to(o.dtype)
            # 位置 ts+t 保持 MASK embedding，不注入
            return o

        def _cap(module, args, out):
            _fh["h"] = out  # (1, L+t+1, d_model)

        h1 = embed_layer.register_forward_hook(_inject)
        h2 = ln_f.register_forward_hook(_cap)
        model(input_ids=input_ids)
        h1.remove()
        h2.remove()

        # 取最新 thought 位置（L+t）的 hidden state 作为 cₜ
        new_thought = _fh["h"][0, L_thought_start + t].detach()  # (d_model,)
        thoughts.append(new_thought)

    return torch.stack(thoughts, dim=0)  # (n_thoughts, d_model)


@torch.no_grad()
def denoise_block_with_thoughts(
    model, input_ids,
    block_start, block_end,
    mask_id,
    embed_layer,
    thoughts,           # (n_thoughts, d_model)
    thought_pos_slice,  # slice，指向 thought 占位符在 input_ids 中的位置
    num_steps=NUM_STEPS,
):
    """标准 LLaDA block diffusion，embed hook 把 thought 占位符替换为 continuous thoughts。"""
    block_len = block_end - block_start

    def _make_hook(th, pos):
        def _h(module, args, out):
            o = out.clone()
            o[0, pos] = th.to(o.dtype)  # th: (n_thoughts, d_model), pos: slice
            return o
        return _h

    for _ in range(num_steps):
        h = embed_layer.register_forward_hook(
            _make_hook(thoughts, thought_pos_slice)
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


def coconut_think_then_generate(
    model, tokenizer,
    context_ids,   # (1, L) — 当前轮次的 prompt tensor
    mask_id, d_model,
    n_thoughts=N_THOUGHTS,
    block_size=BLOCK_SIZE,
    num_steps=NUM_STEPS,
    max_blocks=MAX_BLOCKS,
    stop_strings=None,
):
    """
    两阶段生成（COCONUT 风格）：
      Phase 1: 链式 thought 生成 → thoughts (n_thoughts, d_model)
      Phase 2: block diffusion 注入 thoughts → tool_call 或 answer 文本

    返回: action_text（只含 tool_call 或 box answer，不含 think 文本）
    """
    device = context_ids.device
    embed_layer = model.model.transformer.wte

    # 在 context 末尾追加 <think>，thought 链从这里开始
    think_start_ids    = tokenizer.encode("<think>",   add_special_tokens=False)
    think_end_ids      = tokenizer.encode("</think>",  add_special_tokens=False)
    think_start_tensor = torch.tensor([think_start_ids], dtype=torch.long, device=device)
    think_end_tensor   = torch.tensor([think_end_ids],   dtype=torch.long, device=device)

    # denoise_ctx = [context | <think>]
    # thought 占位符将从 L_thought_start 位置追加
    denoise_ctx     = torch.cat([context_ids, think_start_tensor], dim=1)
    L_thought_start = denoise_ctx.shape[1]  # thought 占位符的绝对起始位置

    # ── Phase 1: COCONUT 链式 thought 生成 ──────────────────────────────────
    # 每步 forward 一次，共 n_thoughts 步，thoughts[t] = cₜ
    thoughts = generate_continuous_thoughts(
        model, denoise_ctx, L_thought_start, n_thoughts, d_model, mask_id
    )
    # thoughts: (n_thoughts, d_model)

    # ── Phase 2: action 生成（注入 continuous thoughts） ─────────────────────
    # action_ctx = [context | <think> | MASK×n_thoughts | </think>]
    thought_placeholders = torch.full(
        (1, n_thoughts), mask_id, dtype=torch.long, device=device
    )
    action_ctx = torch.cat([
        denoise_ctx,
        thought_placeholders,
        think_end_tensor,
    ], dim=1)
    thought_pos = slice(L_thought_start, L_thought_start + n_thoughts)

    generated_ids = []
    stop_strings  = stop_strings or []
    eos_id        = tokenizer.eos_token_id or 0

    for _ in range(max_blocks):
        new_block  = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        action_ctx = torch.cat([action_ctx, new_block], dim=1)
        block_start = action_ctx.shape[1] - block_size
        block_end   = action_ctx.shape[1]

        action_ctx = denoise_block_with_thoughts(
            model, action_ctx,
            block_start, block_end,
            mask_id, embed_layer,
            thoughts, thought_pos,
            num_steps=num_steps,
        )

        block_tokens  = action_ctx[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text  = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for s in stop_strings:
            if s in current_text:
                current_text = current_text[:current_text.index(s) + len(s)]
                return current_text

        if all(tok == eos_id for tok in block_tokens[-8:]):
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 单题 agentic rollout ──────────────────────────────────────────────────────
def run_one(model, tokenizer, mask_id, d_model, question: str) -> dict:
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

        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        try:
            action_text = coconut_think_then_generate(
                model, tokenizer,
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
    parser.add_argument("--offset",      type=int, default=0)
    parser.add_argument("--n_thoughts",  type=int, default=N_THOUGHTS,
                        help="continuous thought 步数（每步一次完整 forward）")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
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

    print(f"模型加载完成  mask_id={mask_id}  d_model={d_model}")
    print(f"COCONUT n_thoughts={args.n_thoughts}  block_size={BLOCK_SIZE}")
    print(f"待评测: {len(todo)} 题\n")

    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="LLaDA COCONUT eval"):
            question  = sample["question"]
            gt_answer = sample["answer"]

            try:
                result = run_one(model, tokenizer, mask_id, d_model, question)
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
    print(f"  python my_eval/score_paper_metrics.py --data {out_path} --no_llm_judge")


if __name__ == "__main__":
    main()
