#!/usr/bin/env python3
"""
LLaDA SFT eval with tool-call logit bias.

在 block diffusion 的每次去噪步骤中，对 <tool_call> 相关 token 加 logit bias，
以提升模型主动发起搜索的概率。

用法:
  python my_eval/run_llada_toolbias_eval.py \
    --model  dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized \
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
    --output dLLM_trainer/VRPO/output/llada_eval/sft_toolbias_ckpt6_100.jsonl \
    --logit_bias 0.5

  # 打分
  python my_eval/cal_acc.py --data ...sft_toolbias_ckpt6_100.jsonl
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# transformers 5.x 兼容补丁
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

_orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
def _safe_adjust(self, *args, **kwargs):
    return _orig_adjust(self, *args, **kwargs)
PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# ── 推理超参 ──────────────────────────────────────────────────────────────────
BLOCK_SIZE = 64
NUM_STEPS  = 64
MAX_BLOCKS = 16
MAX_TURNS  = 5

DEFAULT_MODEL = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_6/optimized"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/output/llada_eval/sft_toolbias_ckpt6_100.jsonl"
)

# ── 确定需要 bias 的 token IDs ─────────────────────────────────────────────────
# 推迟到 tokenizer 加载后初始化
_BIAS_TOKEN_IDS: torch.Tensor | None = None


def build_bias_token_ids(tokenizer, logit_bias: float) -> tuple[torch.Tensor, float]:
    """
    返回 (bias_ids, logit_bias)。
    只选择特异性高的 token（即只在 tool call 上下文中频繁出现的），
    避免对 <, >, ", 换行符等极高频通用 token 加 bias 引入噪声。

    目标 token（以 LLaDA tokenizer 为准）：
      - 'tool'  (28347)  ← <tool_call> 的一部分
      - '_call' (32074)  ← <tool_call> 的一部分
      - '{"'    (12657)  ← JSON 开头，tool call body 的第一个 token
    """
    # 精选：特异性高、在 tool call 中必出现、在普通文本中极少出现
    target_strings = ["tool_call", '{"']
    ids = set()
    for s in target_strings:
        toks = tokenizer(s, add_special_tokens=False).input_ids
        ids.update(toks)

    # 如果 <tool_call> 是单个特殊 token，也加进去
    for special in ["<tool_call>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)

    ids_tensor = torch.tensor(sorted(ids), dtype=torch.long)
    print(f"[toolbias] logit_bias={logit_bias:+.2f}  bias tokens ({len(ids)}): "
          f"{[tokenizer.convert_ids_to_tokens(i) for i in sorted(ids)]}")
    return ids_tensor, logit_bias


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


# ── Block-by-block LLaDA 生成（带 logit bias）────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps,
                  bias_ids=None, logit_bias=0.0):
    block_len = block_end - block_start
    device = input_ids.device

    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits
        block_logits = logits[0, block_start:block_end].float()  # (block_len, vocab)

        # 只对仍是 [MASK] 的位置加 bias
        still_masked = (input_ids[0, block_start:block_end] == mask_id)  # (block_len,)
        if bias_ids is not None and logit_bias != 0.0 and still_masked.any():
            bias_ids_dev = bias_ids.to(device)
            masked_rows = still_masked.nonzero(as_tuple=True)[0]  # (n_masked,)
            # advanced indexing: (n_masked, 1) x (1, n_bias) → (n_masked, n_bias) positions
            block_logits[masked_rows.unsqueeze(1), bias_ids_dev.unsqueeze(0)] += logit_bias

        probs      = torch.softmax(block_logits, dim=-1)
        pred_ids   = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

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


@torch.no_grad()
def llada_generate_blocks(model, tokenizer, prompt_ids, mask_id,
                           block_size=64, num_steps=64, max_blocks=16,
                           stop_strings=None, bias_ids=None, logit_bias=0.0):
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block(model, input_ids, block_start, block_end,
                                  mask_id, num_steps,
                                  bias_ids=bias_ids, logit_bias=logit_bias)

        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop_str in stop_strings:
            if stop_str in current_text:
                idx = current_text.index(stop_str) + len(stop_str)
                return current_text[:idx]

        tail = block_tokens[-8:]
        if all(t == eos_id for t in tail):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 单题 agentic rollout ──────────────────────────────────────────────────────
def run_one(model, tokenizer, mask_id, question: str,
            bias_ids=None, logit_bias=0.0) -> dict:
    stop_strings = ["</tool_call>", "<|box_end|>"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prediction        = None
    termination_reason = "max_turns"
    num_turns          = 0

    for turn in range(MAX_TURNS):
        num_turns = turn + 1

        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_text = llada_generate_blocks(
            model, tokenizer, prompt_ids,
            mask_id=mask_id,
            block_size=BLOCK_SIZE,
            num_steps=NUM_STEPS,
            max_blocks=MAX_BLOCKS,
            stop_strings=stop_strings,
            bias_ids=bias_ids,
            logit_bias=logit_bias,
        )

        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction         = answer
            termination_reason = "answer"
            break

        queries = parse_tool_call(new_text)
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
            full_context += new_text + tool_response
        else:
            full_context += new_text

    return {
        "prediction":         prediction,
        "num_turns":          num_turns,
        "termination_reason": termination_reason,
        "messages":           messages,
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--input",        default=DEFAULT_INPUT)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples",  type=int, default=0)
    parser.add_argument("--offset",       type=int, default=0)
    parser.add_argument("--logit_bias",   type=float, default=0.5,
                        help="加到 tool_call token logits 上的偏置（default 0.5）")
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
    print(f"eval 数据: {len(samples)} 题  来自 {args.input}")

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

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.eval()
    mask_id = model.config.mask_token_id
    print(f"模型加载完成，mask_token_id={mask_id}")

    bias_ids, logit_bias = build_bias_token_ids(tokenizer, args.logit_bias)

    print(f"待评测: {len(todo)} 题\n")

    answered = len(done_questions)
    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="LLaDA toolbias eval"):
            question  = sample["question"]
            gt_answer = sample["answer"]

            try:
                result = run_one(model, tokenizer, mask_id, question,
                                 bias_ids=bias_ids, logit_bias=logit_bias)
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
                "logit_bias":         logit_bias,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            if result["prediction"]:
                answered += 1

    total = len(done_questions) + len(todo)
    print(f"\n完成！共 {total} 题，有答案 {answered} 题 ({answered/total*100:.1f}%)")
    print(f"结果 → {out_path}")
    print(f"\n打分命令:")
    print(f"  python my_eval/cal_acc.py --data {out_path}")


if __name__ == "__main__":
    main()
