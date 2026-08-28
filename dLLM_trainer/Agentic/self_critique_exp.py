"""
自我纠错实验（Self-Critique Experiment）
========================================
把失败案例的完整对话（含错误答案）喂给 π₃ 模型，
在末尾加一条"请验证这个答案"的用户消息，
观察模型是否能：
  1. 意识到答案可能有问题
  2. 主动再次搜索
  3. 得到正确答案

用法：
  python self_critique_exp.py --n 5 --gpu 0
"""
import argparse
import json
import os
import re
import sys
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x 兼容补丁（LLaDA tie_weights / all_tied_weights_keys）──────
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

# ─────────────────────────── 路径 ───────────────────────────────────────────
MODEL_PATH   = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/model"
MERGED_PATH  = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/eval/merged.jsonl"
HALL_PATH    = "/tmp/claude-1905237802/-research-cbim-vast-mz751-Projects-DLLM-Searcher-dLLM-trainer-VRPO/bc4244b6-9acd-4dae-a2de-99cb6ecf18de/scratchpad/hall_full_v2.json"
CONFIG_PATH  = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json"
OUT_PATH     = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/self_critique_results.jsonl"

# ─────────────────────────── 生成参数 ───────────────────────────────────────
BLOCK_SIZE   = 64
NUM_STEPS    = 64
MAX_BLOCKS   = 20     # 每次最多生成 20 块 = 1280 token
TEMPERATURE  = 0.0    # greedy

# ─────────────────────────── 搜索 API ───────────────────────────────────────
_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        _cfg = json.load(f)
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"

def google_search(q: str) -> str:
    try:
        r = requests.post(SEARCH_URL,
            headers={"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"},
            json={"q": q, "num": 5}, timeout=10)
        if r.status_code != 200:
            return f"Search error {r.status_code}"
        data = r.json()
        if "organic" not in data:
            return f"No results for '{q}'."
        return f"Results for '{q}':\n" + "\n".join(
            f"{i}. [{p['title']}] {p.get('snippet','')}"
            for i, p in enumerate(data["organic"][:5], 1))
    except Exception as e:
        return f"Search failed: {e}"

def run_search(queries: list) -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

# ─────────────────────────── 解析 tool_call ──────────────────────────────────
def parse_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None, False
    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        queries = obj.get("arguments", {}).get("query", [])
        if isinstance(queries, list):
            return queries, True
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if m2:
            try:
                queries = json.loads(m2.group(1))
                if isinstance(queries, list):
                    return queries, True
            except Exception:
                pass
    return None, False

def extract_answer(text: str):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ─────────────────────────── Block 扩散生成 ──────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps, temperature=0.0):
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = input_ids[0, block_start:block_end] == mask_id
        if still_masked.sum() == 0:
            break
        if temperature > 0:
            probs    = torch.softmax(logits.float() / temperature, dim=-1)
            pred_ids = torch.multinomial(probs, 1).squeeze(-1)
            confidence = probs.max(dim=-1).values
        else:
            # greedy
            pred_ids   = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).max(dim=-1).values
        n_reveal = max(1, round((block_end - block_start) / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, still_masked.sum().item())
        top_idx  = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]
    return input_ids

@torch.no_grad()
def llada_generate(model, tokenizer, prompt_ids, mask_id,
                   block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                   max_blocks=MAX_BLOCKS, stop_strings=None, temperature=TEMPERATURE):
    device       = next(model.parameters()).device
    eos_id       = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []
    input_ids    = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block   = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block(model, input_ids, block_start, block_end,
                                  mask_id, num_steps, temperature)
        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop in stop_strings:
            if stop in current_text:
                return current_text[:current_text.index(stop) + len(stop)]
        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─────────────────────────── 核心实验函数 ────────────────────────────────────
CRITIQUE_MSG = (
    "请仔细验证你上面给出的答案。你确信这个答案是正确的吗？"
    "如果你不确定，请立刻再次进行网络搜索来确认。"
)

def run_critique_turn(model, tokenizer, mask_id, device,
                      original_messages: list, gt: str,
                      max_extra_turns: int = 3) -> dict:
    """
    以原始失败对话为基础，加一条验证消息，
    再运行最多 max_extra_turns 轮，观察模型行为。

    Returns dict with keys:
      - messages: updated message list
      - prediction: new final answer (or None)
      - turns_taken: number of extra turns
      - did_search: bool
      - new_queries: list of query strings used
      - new_search_results: list of search result strings
      - termination: "answer" | "max_turns"
      - correct: bool
    """
    stop_strings = ["</tool_call>", "<|box_end|>"]
    messages     = original_messages.copy()
    messages.append({"role": "user", "content": CRITIQUE_MSG})

    prediction   = None
    did_search   = False
    new_queries  = []
    new_results  = []
    turns_taken  = 0

    # Rebuild full_context from messages
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for turn in range(max_extra_turns):
        turns_taken += 1
        prompt_ids = tokenizer(full_context, return_tensors="pt",
                               add_special_tokens=False).input_ids
        new_text = llada_generate(model, tokenizer, prompt_ids, mask_id,
                                  stop_strings=stop_strings)
        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction = answer
            break

        queries, _ = parse_tool_call(new_text)
        if queries:
            did_search = True
            new_queries.extend(queries)
            search_result = run_search(queries)
            new_results.append(search_result)
            tool_resp_content = f"<tool_response>\n{search_result}\n</tool_response>"
            messages.append({"role": "user", "content": tool_resp_content})
            # 用 apply_chat_template 重建 full_context（最安全，格式一定正确）
            full_context = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # no tool call, no box — 重建上下文继续
            full_context = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    correct = False
    if prediction and gt:
        correct = gt.lower().strip() in prediction.lower()

    return {
        "messages":          messages,
        "prediction":        prediction,
        "turns_taken":       turns_taken,
        "did_search":        did_search,
        "new_queries":       new_queries,
        "new_search_results": new_results,
        "termination":       "answer" if prediction else "max_turns",
        "correct":           correct,
    }


# ─────────────────────────── 主程序 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=5,  help="题目数量")
    parser.add_argument("--gpu", type=int, default=0,  help="GPU index")
    parser.add_argument("--subcat", type=str, default="",
                        help="只跑某个子类 (result_misread/bad_query/fake_search)")
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── 加载失败案例 ──────────────────────────────────────────────────────────
    hall_data  = json.load(open(HALL_PATH))
    merged_map = {json.loads(l)["question"]: json.loads(l)
                  for l in open(MERGED_PATH)}

    # 按子类均匀采样
    subcats = ["result_misread", "bad_query", "fake_search"]
    cases = []
    if args.subcat:
        pool = [r for r in hall_data if r["subcat"] == args.subcat]
        cases = pool[:args.n]
    else:
        # 各子类至少取 1 个，凑满 n 个
        import random; random.seed(42)
        for sc in subcats:
            pool = [r for r in hall_data if r["subcat"] == sc]
            cases.extend(pool[:max(1, args.n // len(subcats))])
        cases = cases[:args.n]

    print(f"[self-critique] 共 {len(cases)} 道题")
    for c in cases:
        print(f"  subcat={c['subcat']}  idx={c['idx']}  gt={c['gt']!r}")
        print(f"    Q: {c['question'][:70]}")

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print(f"\n[loading] {MODEL_PATH}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # LLaDA 必要配置
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"[mask_id] = {mask_id}  (from model.config.mask_token_id)")

    # ── 逐题实验 ─────────────────────────────────────────────────────────────
    results = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    for i, case in enumerate(cases):
        q = case["question"]
        gt = case["gt"]
        orig_pred = case["prediction"]
        subcat = case["subcat"]

        # 从 merged.jsonl 取原始消息（含实际的 tool_response 内容）
        merged = merged_map.get(q, {})
        orig_msgs = merged.get("messages", [])

        if not orig_msgs:
            print(f"[{i+1}] SKIP: no messages for q={q[:50]!r}")
            continue

        print(f"\n{'='*65}")
        print(f"[{i+1}/{len(cases)}] subcat={subcat}")
        print(f"  Q:   {q}")
        print(f"  GT:  {gt!r}")
        print(f"  原始答案: {orig_pred[:80]!r}")
        print(f"  原始消息数: {len(orig_msgs)}")
        print(f"  → 加入验证消息，运行额外生成...")

        result = run_critique_turn(
            model, tokenizer, mask_id, device,
            original_messages=orig_msgs,
            gt=gt,
            max_extra_turns=3,
        )

        status = "✓ 纠正!" if result["correct"] else ("✗ 仍错" if result["prediction"] else "○ 无答案")
        print(f"  结果: {status}")
        print(f"  新预测: {(result['prediction'] or '(无)')[:80]!r}")
        print(f"  搜索了吗: {result['did_search']}  新查询: {result['new_queries'][:2]}")
        print(f"  额外轮数: {result['turns_taken']}  终止: {result['termination']}")

        entry = {
            "idx":         case["idx"],
            "subcat":      subcat,
            "question":    q,
            "gt":          gt,
            "orig_pred":   orig_pred,
            "new_pred":    result["prediction"],
            "correct":     result["correct"],
            "did_search":  result["did_search"],
            "new_queries": result["new_queries"],
            "turns_taken": result["turns_taken"],
            "termination": result["termination"],
            # 最后 2 条助手消息（验证轮 + 新答案轮）
            "new_asst_msgs": [
                m["content"] for m in result["messages"]
                if m["role"] == "assistant"
            ][-3:],
        }
        results.append(entry)
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"实验完成，共 {len(results)} 题")
    n_correct   = sum(r["correct"]    for r in results)
    n_searched  = sum(r["did_search"] for r in results)
    n_answered  = sum(r["new_pred"] is not None for r in results)
    print(f"  重新搜索:  {n_searched}/{len(results)} ({n_searched/len(results)*100:.0f}%)")
    print(f"  给出新答案: {n_answered}/{len(results)}")
    print(f"  纠错成功:  {n_correct}/{len(results)} ({n_correct/len(results)*100:.0f}%)")
    print(f"\n[saved] {args.out}")
    return results


if __name__ == "__main__":
    main()
