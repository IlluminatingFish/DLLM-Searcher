#!/usr/bin/env python3
"""
验证训练数据中第一个 completion block 的 position 63 / 126 与推理假设是否对齐。
推理假定：block 内 pos 63 = <tool_call> 或 <|box_start|>，pos 126 = </tool_call> 或 <|box_end|>
"""
import json
import os
import sys

# 需要能 import tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "my_train"))

def main():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Need transformers. Install or run from VRPO env.")
        return

    # 使用与训练一致的 tokenizer 路径（按你实际 config 调整）
    model_path = os.environ.get("MODEL_PATH", "../SFT/dLLM-RL/sft_sdar/ckpt_1/optimized")
    if not os.path.exists(model_path):
        model_path = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B")
    print(f"Using tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 特殊 token id（按实际 vocab 核对）
    TOOL_CALL_START = 151657   # <tool_call>
    TOOL_CALL_END = 151658     # </tool_call>
    BOX_START = 151648         # <|box_start|>
    BOX_END = 151649           # <|box_end|>

    # 检查 vocab 中是否存在
    try:
        _ = tokenizer.decode([TOOL_CALL_START, TOOL_CALL_END, BOX_START, BOX_END])
    except Exception:
        pass

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl")
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    stats = {
        "total": 0,
        "pos63_is_toolcall_start": 0,
        "pos63_is_box_start": 0,
        "pos126_is_toolcall_end": 0,
        "pos126_is_box_end": 0,
        "pos63_in_thinking": 0,
        "pos63_in_tool_response": 0,
        "pos63_other": 0,
    }

    sample_results = []
    block_length = 128

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 100:  # 采样前 100 条
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]

            prompt_len = len(prompt_ids)
            # 第一个 completion block 在全局序列中的范围
            prefill_blocks = prompt_len // block_length
            prefill_length = prefill_blocks * block_length

            # 第一个 completion block: [prefill_length, prefill_length + block_length)
            if prefill_length + block_length > len(prompt_ids) + len(chosen_ids):
                # completion 不足一个 block，跳过
                continue

            full_ids = prompt_ids + chosen_ids
            block_start = prefill_length
            block_end = block_start + block_length

            if block_end > len(full_ids):
                continue

            block = full_ids[block_start:block_end]
            pos63_token = block[63] if len(block) > 63 else None
            pos126_token = block[126] if len(block) > 126 else None

            stats["total"] += 1

            r63 = "other"
            if pos63_token == TOOL_CALL_START:
                stats["pos63_is_toolcall_start"] += 1
                r63 = "toolcall_start"
            elif pos63_token == BOX_START:
                stats["pos63_is_box_start"] += 1
                r63 = "box_start"
            else:
                # 粗略判断：thinking 通常在前面，tool_response 在特定区间
                # 这里简化：非特殊 token 都算 other
                stats["pos63_other"] += 1
                r63 = f"token_{pos63_token}"

            r126 = "other"
            if pos126_token == TOOL_CALL_END:
                stats["pos126_is_toolcall_end"] += 1
                r126 = "toolcall_end"
            elif pos126_token == BOX_END:
                stats["pos126_is_box_end"] += 1
                r126 = "box_end"
            else:
                r126 = f"token_{pos126_token}"

            if len(sample_results) < 5:
                sample_results.append({
                    "idx": i,
                    "pos63": r63,
                    "pos126": r126,
                    "pos63_decode": tokenizer.decode([pos63_token]) if pos63_token else "",
                    "pos126_decode": tokenizer.decode([pos126_token]) if pos126_token else "",
                })

    print("\n=== Block Alignment Check (first completion block, pos 63 & 126) ===\n")
    print(f"Sampled {stats['total']} examples from train.jsonl\n")
    print("Position 63:")
    print(f"  - <tool_call> (151657): {stats['pos63_is_toolcall_start']} ({100*stats['pos63_is_toolcall_start']/max(1,stats['total']):.1f}%)")
    print(f"  - <|box_start|> (151648): {stats['pos63_is_box_start']} ({100*stats['pos63_is_box_start']/max(1,stats['total']):.1f}%)")
    print(f"  - Other: {stats['pos63_other']} ({100*stats['pos63_other']/max(1,stats['total']):.1f}%)")
    print("\nPosition 126:")
    print(f"  - </tool_call> (151658): {stats['pos126_is_toolcall_end']} ({100*stats['pos126_is_toolcall_end']/max(1,stats['total']):.1f}%)")
    print(f"  - <|box_end|> (151649): {stats['pos126_is_box_end']} ({100*stats['pos126_is_box_end']/max(1,stats['total']):.1f}%)")
    print("\nSample decode (first 5):")
    for s in sample_results:
        print(f"  [{s['idx']}] pos63={s['pos63']} ({repr(s['pos63_decode'][:30])}) | pos126={s['pos126']} ({repr(s['pos126_decode'][:30])})")

    if stats["pos63_other"] > stats["total"] * 0.5:
        print("\n*** 结论: 多数 sample 的 pos 63/126 不是推理假设的 boundary token，存在 alignment 风险。***")


if __name__ == "__main__":
    main()
