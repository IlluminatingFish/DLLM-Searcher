"""
Base Model 错误分析实验
=======================
用原始 LLaDA base model（未经 SFT/RL）对比 π₃ 的自我纠错实验：
- 输入：完全相同的失败对话上下文
- 末尾加一条消息：请分析这个答案为什么可能有问题
- 不要求搜索，只要求分析
- 对比：base model 是否有任何元认知能力？

用法：
  python base_analysis_exp.py --n 3 --gpu 0
"""
import argparse, json, os, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x 兼容补丁 ─────────────────────────────────────────────────
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
            try: return _orig_tw(self, **kwargs)
            except TypeError: return _orig_tw(self)
        type(model).tie_weights = _safe_tw
        try: return _orig_finalize_fn(model, load_config, loading_info)
        finally: type(model).tie_weights = _orig_tw
    PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
    _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
    def _safe_adjust(self, *args, **kwargs):
        return _orig_adjust(self, *args, **kwargs)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/common/users/mz751/Projects/dLLM_trainer/checkpoints/base/llada_model"
MERGED_PATH     = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/eval/merged.jsonl"
HALL_PATH       = "/tmp/claude-1905237802/-research-cbim-vast-mz751-Projects-DLLM-Searcher-dLLM-trainer-VRPO/bc4244b6-9acd-4dae-a2de-99cb6ecf18de/scratchpad/hall_full_v2.json"
OUT_PATH        = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/base_analysis_results.jsonl"

# ── 生成参数 ──────────────────────────────────────────────────────────────────
BLOCK_SIZE = 64
NUM_STEPS  = 64
MAX_BLOCKS = 16    # 最多 1024 token 的分析
TEMPERATURE = 0.0

# ── 新的分析 prompt（不要求搜索）─────────────────────────────────────────────
ANALYSIS_MSG = (
    "请仔细阅读上面的对话。助手给出了一个答案，"
    "但这个答案可能存在错误。"
    "请分析：(1) 这个答案哪里可能有问题？"
    "(2) 正确答案应该是什么？"
    "请给出你的判断和理由。"
)

# ── Block 扩散生成 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps):
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = input_ids[0, block_start:block_end] == mask_id
        if still_masked.sum() == 0:
            break
        pred_ids   = logits.argmax(dim=-1)
        confidence = logits.softmax(dim=-1).max(dim=-1).values
        n_reveal   = max(1, round((block_end - block_start) / num_steps))
        conf_m     = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal   = min(n_reveal, still_masked.sum().item())
        top_idx    = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]
    return input_ids

@torch.no_grad()
def llada_generate(model, tokenizer, prompt_ids, mask_id,
                   block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                   max_blocks=MAX_BLOCKS, stop_strings=None):
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
        input_ids   = denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps)

        block_tokens  = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text  = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop in stop_strings:
            if stop in current_text:
                return current_text[:current_text.index(stop) + len(stop)]
        # EOS 停止
        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)
        # 包含 eot_id 也停止
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot and eot in block_tokens[-4:]:
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"   # 设完 VISIBLE_DEVICES 后进程内只有 cuda:0

    # ── 加载失败案例（同 π₃ 实验的 3 道题）────────────────────────────────────
    hall_data  = json.load(open(HALL_PATH))
    merged_map = {json.loads(l)["question"]: json.loads(l)
                  for l in open(MERGED_PATH)}

    import random; random.seed(42)
    # 按真实比例采样：result_misread=91, bad_query=19, fake_search=5
    # n=20 → 12 + 5 + 3 = 20（fake_search 最多 5 个）
    sc_quota = {
        "result_misread": max(1, round(args.n * 0.60)),   # ~60%
        "bad_query":      max(1, round(args.n * 0.25)),   # ~25%
        "fake_search":    min(5, max(1, round(args.n * 0.15))),  # ~15%, 上限5
    }
    cases = []
    for sc, quota in sc_quota.items():
        pool = [r for r in hall_data if r["subcat"] == sc]
        random.shuffle(pool)
        cases.extend(pool[:quota])
    # 如果总数不够 n，从 result_misread 补齐
    if len(cases) < args.n:
        rm_pool = [r for r in hall_data if r["subcat"] == "result_misread"
                   and r not in cases]
        cases.extend(rm_pool[:args.n - len(cases)])
    cases = cases[:args.n]

    print(f"[base-analysis] 共 {len(cases)} 道题（与 π₃ 实验完全相同）")
    for c in cases:
        print(f"  subcat={c['subcat']:16s}  gt={c['gt']!r}")
        print(f"    Q: {c['question'][:70]}")

    # ── 加载 base 模型 ────────────────────────────────────────────────────────
    print(f"\n[loading] {BASE_MODEL_PATH}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device).eval()
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"[mask_id] = {mask_id}")
    print(f"[model loaded]  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # ── 逐题实验 ──────────────────────────────────────────────────────────────
    results = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f   = open(args.out, "w")

    for i, case in enumerate(cases):
        q         = case["question"]
        gt        = case["gt"]
        subcat    = case["subcat"]
        merged    = merged_map.get(q, {})
        orig_msgs = merged.get("messages", [])
        orig_pred = (merged.get("prediction") or "")

        if not orig_msgs:
            print(f"[{i+1}] SKIP")
            continue

        # 构建输入：原始失败对话 + 分析请求
        msgs_with_analysis = orig_msgs + [{"role": "user", "content": ANALYSIS_MSG}]
        full_ctx = tokenizer.apply_chat_template(
            msgs_with_analysis, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(full_ctx, return_tensors="pt",
                               add_special_tokens=False).input_ids

        print(f"\n{'='*65}")
        print(f"[{i+1}/{len(cases)}] subcat={subcat}")
        print(f"  Q:   {q}")
        print(f"  GT:  {gt!r}")
        print(f"  原始错误答案: {orig_pred[:80]!r}")
        print(f"  Context tokens: {prompt_ids.shape[1]}")
        print(f"  → base model 分析中...")

        # 生成分析
        stop_strings = ["<|eot_id|>", "<|endoftext|>"]
        output = llada_generate(
            model, tokenizer, prompt_ids, mask_id,
            stop_strings=stop_strings,
        )

        # 清理输出
        output_clean = output.strip()
        # 去掉可能残留的 special token 前缀
        for tok_str in ["<|eot_id|>", "<|endoftext|>", "<|start_header_id|>", "<|end_header_id|>"]:
            output_clean = output_clean.replace(tok_str, "").strip()

        print(f"\n  [base model 输出]:")
        print(f"  {output_clean}")

        # 简单判断：base model 的分析里有没有提到 GT
        gt_mentioned = gt.lower() in output_clean.lower()
        print(f"\n  GT '{gt}' 出现在输出中: {gt_mentioned}")

        entry = {
            "idx":        case.get("idx", i+1),
            "subcat":     subcat,
            "question":   q,
            "gt":         gt,
            "orig_pred":  orig_pred,
            "base_output": output_clean,
            "gt_mentioned": gt_mentioned,
            "ctx_tokens": prompt_ids.shape[1],
        }
        results.append(entry)
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"完成，共 {len(results)} 题")
    n_gt = sum(r["gt_mentioned"] for r in results)
    print(f"  Base model 输出中提到 GT: {n_gt}/{len(results)}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
