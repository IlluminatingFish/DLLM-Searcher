#!/usr/bin/env python3
"""
LLaDA SFT 模型的批量 agentic 评测脚本。

对 eval JSONL 里每个问题做完整的 agentic rollout（think → search → answer），
输出 {question, answer, prediction} JSONL，可直接送给 cal_acc.py 打分。

用法:
  # 评测 ckpt_6（默认）
  python my_eval/run_llada_eval.py \
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
    --output dLLM_trainer/VRPO/output/llada_eval/ckpt6_sft200.jsonl

  # 评测指定 checkpoint
  python my_eval/run_llada_eval.py \
    --model  dLLM_trainer/SFT/dLLM-RL/sft_llada/ckpt_3/optimized \
    --input  dLLM_trainer/VRPO/data/sft_train_as_eval_200.jsonl \
    --output dLLM_trainer/VRPO/output/llada_eval/ckpt3_sft200.jsonl

  # 打分
  python my_eval/cal_acc.py --data dLLM_trainer/VRPO/output/llada_eval/ckpt6_sft200.jsonl
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

# transformers 5.x 兼容补丁：LLaDA 缺少 all_tied_weights_keys 属性
# caching_allocator_warmup 在 _adjust_tied_keys 之前就访问该属性，需改成 property
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

if hasattr(PreTrainedModel, '_adjust_tied_keys_with_tied_pointers'):
    _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
    def _safe_adjust(self, *args, **kwargs):
        return _orig_adjust(self, *args, **kwargs)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# ── 推理超参 ──────────────────────────────────────────────────────────────────
BLOCK_SIZE        = 64  # 与训练一致（可被 --block_size 覆盖）
NUM_STEPS         = 64  # =block_size，确保每个 token 都有机会被揭露（可被 --num_steps 覆盖）
MAX_BLOCKS        = 16  # 最多生成 16×64=1024 token
MAX_TURNS         = 5   # 最多 5 轮 think→search
FORCE_ANSWER_TURN = 3   # 第 3 轮起（0-indexed）强制注入 <|box_start|> 要求模型给答案

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
    "/dLLM_trainer/VRPO/output/llada_eval/ckpt6_sft200.jsonl"
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


# ── Block-by-block LLaDA 生成 ─────────────────────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps):
    block_len = block_end - block_start
    for _ in range(num_steps):
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
            input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


@torch.no_grad()
def llada_generate_blocks(model, tokenizer, prompt_ids, mask_id,
                           block_size=64, num_steps=64, max_blocks=16,
                           stop_strings=None):
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
                                  mask_id, num_steps)

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
def run_one(model, tokenizer, mask_id, question: str) -> dict:
    """
    对单个问题做 agentic rollout。
    返回 {prediction, num_turns, termination_reason, messages}，
    其中 messages 包含完整对话轨迹（含 tool_call / tool_response）。
    """
    stop_strings = ["</tool_call>", "<|box_end|>"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    # apply_chat_template 产生 Llama3 格式（<|start_header_id|> 等真实 special token）。
    # 注：<|im_start|>/<|im_end|> 不在该 tokenizer 词表中，会被拆成普通字符，
    # 而 Llama3 special token 是模型预训练阶段已学过的强结构信号，实测性能更好。
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prediction        = None
    termination_reason = "max_turns"
    num_turns          = 0

    for turn in range(MAX_TURNS):
        num_turns = turn + 1
        force_answer = (turn >= FORCE_ANSWER_TURN)

        if force_answer:
            # 最后两轮：注入 <|box_start|>，强制模型生成答案
            gen_context   = full_context + "<|box_start|>"
            cur_stops     = ["<|box_end|>"]
        else:
            gen_context   = full_context
            cur_stops     = stop_strings

        prompt_ids = tokenizer(
            gen_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_text = llada_generate_blocks(
            model, tokenizer, prompt_ids,
            mask_id=mask_id,
            block_size=BLOCK_SIZE,
            num_steps=NUM_STEPS,
            max_blocks=MAX_BLOCKS,
            stop_strings=cur_stops,
        )

        if force_answer:
            # new_text 是 <|box_start|> 之后的内容，截到 <|box_end|> 前
            ans_text = new_text.split("<|box_end|>")[0].strip() if "<|box_end|>" in new_text else new_text.strip()
            if ans_text:
                prediction         = ans_text
                termination_reason = "forced_answer"
            messages.append({"role": "assistant", "content": f"<|box_start|>{new_text}"})
            break

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
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"<tool_response>\n{search_result}\n</tool_response>"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
    global FORCE_ANSWER_TURN, BLOCK_SIZE, NUM_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_MODEL,  help="模型路径")
    parser.add_argument("--input",       default=DEFAULT_INPUT,  help="eval JSONL 路径")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT, help="结果 JSONL 输出路径")
    parser.add_argument("--max_samples",       type=int, default=0,                help="只跑前 N 题（0=全部）")
    parser.add_argument("--offset",            type=int, default=0,                help="从第几题开始（用于多进程并行）")
    parser.add_argument("--force_answer_turn", type=int, default=FORCE_ANSWER_TURN, help="从第几轮起强制输出答案（999=禁用）")
    parser.add_argument("--block_size",        type=int, default=BLOCK_SIZE,        help="推理 block size，需与训练一致")
    parser.add_argument("--num_steps",         type=int, default=NUM_STEPS,         help="每个 block 的 denoising steps")
    args = parser.parse_args()
    FORCE_ANSWER_TURN = args.force_answer_turn
    BLOCK_SIZE        = args.block_size
    NUM_STEPS         = args.num_steps

    # 加载 eval 数据
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

    # 断点续跑：跳过已完成的问题
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
        print("全部题目已完成！运行打分:")
        print(f"  python my_eval/cal_acc.py --data {out_path}")
        return

    # 加载模型
    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.eval()
    mask_id = model.config.mask_token_id
    print(f"模型加载完成，mask_token_id={mask_id}")
    print(f"待评测: {len(todo)} 题\n")

    # 逐题推理，实时写入（方便中断后续跑）
    answered = len(done_questions)  # 之前已有答案的
    with open(out_path, "a") as f_out:
        for sample in tqdm(todo, desc="LLaDA agentic eval"):
            question = sample["question"]
            gt_answer = sample["answer"]
            short_answer = sample.get("short_answer", "")  # 保留短答案字段（CEM-1 专用）

            try:
                result = run_one(model, tokenizer, mask_id, question)
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
                "short_answer":       short_answer,   # cal_acc.py 的 CEM-1/Token-F1 优先用此字段
                "prediction":         result["prediction"],
                "num_turns":          result["num_turns"],
                "termination_reason": result["termination_reason"],
                "messages":           result["messages"],
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
