"""
Token Generation Order Tracer
==============================
对选定的 20 道题重跑 π₃（greedy, temperature=0），
在每个 block 的 denoising 过程中记录每个 token 在第几步被揭示，
以及揭示时的置信度。输出供 report 可视化使用。

输出格式（每道题一条 JSON）：
{
  "question": ...,
  "gt": ...,
  "correct": bool,
  "cat": ...,
  "turns": [           # 每个生成 turn（tool_call 或 answer）
    {
      "turn_idx": int,
      "role": "assistant",
      "text": str,
      "blocks": [      # 每个 64-token block
        {
          "block_idx": int,
          "tokens": [str],          # token 字符串
          "token_ids": [int],
          "reveal_step": [int],     # 每个 token 在第几步被揭示（0-indexed）
          "reveal_conf": [float],   # 揭示时的置信度
        }
      ]
    }
  ]
}

用法：
  python token_order_trace.py --gpu 3
"""
import argparse, json, os, re, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x 兼容补丁 ─────────────────────────────────────────────────
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self): return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val): object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

def _patched_get_tied_weight_keys(module):
    keys = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None) or {}
        if isinstance(tied, dict): keys.extend([f"{name}.{k}" if name else k for k in tied])
        elif isinstance(tied, (list, tuple)): keys.extend([f"{name}.{k}" if name else k for k in tied])
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
    def _safe_adjust(self, *a, **kw): return _orig_adjust(self, *a, **kw)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# ── 路径 ──────────────────────────────────────────────────────────────────────
PI3_PATH   = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/model"
MERGED_PATH= "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi3/eval/merged.jsonl"
DATA20_PATH= "/tmp/claude-1905237802/-research-cbim-vast-mz751-Projects-DLLM-Searcher-dLLM-trainer-VRPO/bc4244b6-9acd-4dae-a2de-99cb6ecf18de/scratchpad/report20_steps.json"
OUT_PATH   = "/tmp/claude-1905237802/-research-cbim-vast-mz751-Projects-DLLM-Searcher-dLLM-trainer-VRPO/bc4244b6-9acd-4dae-a2de-99cb6ecf18de/scratchpad/token_order_20.jsonl"

BLOCK_SIZE = 64
NUM_STEPS  = 64

# ── 带 step 记录的 denoising ──────────────────────────────────────────────────
@torch.no_grad()
def denoise_block_trace(model, input_ids, block_start, block_end, mask_id, num_steps):
    """
    Greedy (temperature=0) denoising with per-token reveal step & confidence logging.
    Returns:
      input_ids (updated),
      reveal_step[i]  = which step token i was first revealed (0-indexed)
      reveal_conf[i]  = confidence at reveal time
    """
    blen = block_end - block_start
    reveal_step = [-1] * blen
    reveal_conf = [0.0] * blen

    for step in range(num_steps):
        logits       = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        if still_masked.sum() == 0:
            break

        pred_ids   = logits.argmax(dim=-1)
        confidence = logits.softmax(dim=-1).max(dim=-1).values

        n_reveal = max(1, round(blen / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, still_masked.sum().item())
        top_idx  = conf_m.topk(n_reveal).indices

        for idx in top_idx:
            i = idx.item()
            input_ids[0, block_start + i] = pred_ids[i]
            reveal_step[i] = step
            reveal_conf[i] = confidence[i].item()

    # 填充未揭示的（理论上不应该有）
    for i in range(blen):
        if reveal_step[i] < 0:
            reveal_step[i] = num_steps - 1
            reveal_conf[i] = 0.0

    return input_ids, reveal_step, reveal_conf


@torch.no_grad()
def llada_generate_trace(model, tokenizer, prompt_ids, mask_id,
                         stop_strings=None, max_blocks=20):
    """
    Block-by-block generation with full trace.
    Returns:
      text: str
      blocks_trace: list of {tokens, token_ids, reveal_step, reveal_conf}
    """
    device       = next(model.parameters()).device
    stop_strings = stop_strings or []
    input_ids    = prompt_ids.to(device).clone()
    generated    = []
    blocks_trace = []

    for block_idx in range(max_blocks):
        new_block   = torch.full((1, BLOCK_SIZE), mask_id, dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - BLOCK_SIZE
        block_end   = input_ids.shape[1]

        input_ids, rev_step, rev_conf = denoise_block_trace(
            model, input_ids, block_start, block_end, mask_id, NUM_STEPS)

        block_ids   = input_ids[0, block_start:block_end].tolist()
        block_toks  = tokenizer.convert_ids_to_tokens(block_ids)
        generated.extend(block_ids)

        blocks_trace.append({
            "block_idx":   block_idx,
            "tokens":      [t if t else "<?>" for t in block_toks],
            "token_ids":   block_ids,
            "reveal_step": rev_step,
            "reveal_conf": [round(c, 4) for c in rev_conf],
        })

        current_text = tokenizer.decode(generated, skip_special_tokens=False)
        for stop in stop_strings:
            if stop in current_text:
                cut = current_text[:current_text.index(stop) + len(stop)]
                return cut, blocks_trace

        # EOS check
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id and eot_id in block_ids[-4:]:
            return tokenizer.decode(generated, skip_special_tokens=True), blocks_trace

    return tokenizer.decode(generated, skip_special_tokens=True), blocks_trace


def build_conversation(tokenizer, messages):
    """Apply chat template and tokenize."""
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids  = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--out",  type=str, default=OUT_PATH)
    parser.add_argument("--n",    type=int, default=20, help="how many questions to trace")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # 加载选好的 20 道题及其原始对话
    data20    = json.load(open(DATA20_PATH))[:args.n]
    merged_map= {json.loads(l)["question"]: json.loads(l) for l in open(MERGED_PATH)}

    print(f"[load] {PI3_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(PI3_PATH, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        PI3_PATH, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"[model loaded]  mask_id={mask_id}  params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    for qi, qdata in enumerate(data20):
        q      = qdata["question"]
        gt     = qdata["gt"]
        cat    = qdata["cat"]
        merged = merged_map.get(q, {})
        msgs   = merged.get("messages", [])

        if not msgs:
            print(f"[{qi+1}] SKIP (no messages)")
            continue

        print(f"\n{'='*60}")
        print(f"[{qi+1}/{len(data20)}] cat={cat}  GT={gt!r}")
        print(f"  Q: {q[:70]}")

        # 动态 agentic 循环：
        # - 只取原始 messages 里的 user 问题（第1条）作为起点
        # - 每轮生成一个 assistant turn；检测输出类型（tool_call / answer）
        # - tool_call → 注入 tool_response（来自 merged.jsonl，或 placeholder）→ 继续
        # - answer 出现 → 停止
        # - 最多 MAX_TURNS 轮

        MAX_TURNS = 8
        STOP_STRINGS = ["</tool_call>", "<|box_end|>"]

        # 从 merged.jsonl 取所有 tool_response user messages，按顺序备用
        tool_resps = [m["content"] for m in msgs
                      if m["role"] == "user" and "<tool_response>" in m.get("content", "")]

        # 初始 ctx：只保留第一条 user 消息（题目）
        first_user = next((m for m in msgs if m["role"] == "user"), None)
        ctx_msgs = [first_user] if first_user else []

        turns_trace = []
        tool_resp_idx = 0
        turn_num = 0

        for _ in range(MAX_TURNS):
            turn_num += 1
            prompt_ids = build_conversation(tokenizer, ctx_msgs)
            print(f"  turn {turn_num}  ctx={prompt_ids.shape[1]} tokens")

            text, blocks = llada_generate_trace(
                model, tokenizer, prompt_ids, mask_id,
                stop_strings=STOP_STRINGS, max_blocks=12)

            # 清理特殊 token
            for tok in ["<|eot_id|>", "<|box_end|>"]:
                if tok in text:
                    text = text[:text.index(tok) + len(tok)]
            text_clean = re.sub(r'[^\x00-\x7F一-鿿　-〿＀-￯\n .,;:!?\-\(\)\[\]\'\"<>_|]+', '', text).strip()

            is_toolcall = "<tool_call>" in text_clean
            has_answer  = "<|box_start|>" in text_clean

            print(f"    → {len(blocks)} blocks  tc={is_toolcall} ans={has_answer}  text={repr(text_clean[:80])}")

            turns_trace.append({
                "turn_idx":    turn_num,
                "is_toolcall": is_toolcall,
                "text":        text_clean,
                "blocks":      blocks,
            })

            if has_answer:
                # 生成了答案，结束循环
                break

            if is_toolcall:
                # 把本轮生成的 assistant message 加入 ctx
                ctx_msgs.append({"role": "assistant", "content": text_clean})
                # 注入 tool_response
                if tool_resp_idx < len(tool_resps):
                    resp_content = tool_resps[tool_resp_idx]
                    print(f"    → tool_resp #{tool_resp_idx+1} from merged.jsonl ({len(resp_content)} chars)")
                else:
                    # 没有对应的原始搜索结果，注入空 placeholder
                    resp_content = "<tool_response>\n<response>\nNo results found.\n</response>\n</tool_response>"
                    print(f"    → tool_resp #{tool_resp_idx+1} PLACEHOLDER (no real result)")
                tool_resp_idx += 1
                ctx_msgs.append({"role": "user", "content": resp_content})
            else:
                # 既没有 tool_call 也没有 answer，停止
                print(f"    → 既非 tool_call 也非 answer，停止")
                break

        entry = {
            "idx":     qi + 1,
            "question":q,
            "gt":      gt,
            "correct": qdata["correct"],
            "cat":     cat,
            "turns":   turns_trace,
        }
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  [saved]")

    out_f.close()
    print(f"\n完成! 已保存 {args.out}")


if __name__ == "__main__":
    main()
