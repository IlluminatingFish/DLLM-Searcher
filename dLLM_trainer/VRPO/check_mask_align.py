#!/usr/bin/env python3
"""
检查 get_trainable_mask 与 tokenization 是否一致
"""
import json
import sys
from transformers import AutoTokenizer

TOOL_RESP_LEFT = "<|im_start|>user\n<tool_response>"
TOOL_RESP_RIGHT = "</tool_response><|im_end|>\n<|im_start|>assistant\n"
MODEL_PATH = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_1/optimized"
DATA_PATH = "data/train.jsonl"
MAX_PROMPT = 648
MAX_COMPLETION = 2048


def get_trainable_mask(completion_text, tokenizer, tool_left, tool_right):
    """Token-space: same tokenizer(completion) as tokenize_row, find marker spans in token ids."""
    token_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    left_ids = tokenizer(tool_left, add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(tool_right, add_special_tokens=False)["input_ids"]

    if not left_ids or not right_ids:
        return [True] * len(token_ids)

    spans_not_trainable = []
    i = 0
    while i <= len(token_ids) - len(left_ids):
        if token_ids[i:i + len(left_ids)] != left_ids:
            i += 1
            continue
        start = i
        i += len(left_ids)
        while i <= len(token_ids) - len(right_ids):
            if token_ids[i:i + len(right_ids)] == right_ids:
                end = i + len(right_ids)
                spans_not_trainable.append((start, end))
                i = end
                break
            i += 1
        else:
            spans_not_trainable.append((start, len(token_ids)))
            break

    trainable = []
    for j in range(len(token_ids)):
        in_span = any(s <= j < e for s, e in spans_not_trainable)
        trainable.append(not in_span)
    return trainable


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Loading samples...")
    samples = []
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= 50:  # 检查前 50 个样本
                break
            samples.append(json.loads(line))

    mismatch_chosen = 0
    mismatch_rejected = 0
    all_false_chosen = 0
    all_false_rejected = 0
    details = []

    for idx, s in enumerate(samples):
        prompt = s["prompt"]
        chosen = s["chosen"]
        rejected = s["rejected"]

        # 与 tokenize_row 完全一致
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        if MAX_PROMPT:
            prompt_ids = prompt_ids[-MAX_PROMPT:]
        if MAX_COMPLETION:
            chosen_ids = chosen_ids[:MAX_COMPLETION]
            rejected_ids = rejected_ids[:MAX_COMPLETION]

        chosen_len = len(chosen_ids)
        rejected_len = len(rejected_ids)

        # New API: mask built from completion only, aligned with tokenizer(completion)
        chosen_trainable = get_trainable_mask(chosen, tokenizer, TOOL_RESP_LEFT, TOOL_RESP_RIGHT)
        chosen_trainable = chosen_trainable + [True]
        chosen_trainable = chosen_trainable[:chosen_len]

        rejected_trainable = get_trainable_mask(rejected, tokenizer, TOOL_RESP_LEFT, TOOL_RESP_RIGHT)
        rejected_trainable = rejected_trainable + [True]
        rejected_trainable = rejected_trainable[:rejected_len]

        chosen_ok = len(chosen_trainable) == chosen_len
        rejected_ok = len(rejected_trainable) == rejected_len
        if not chosen_ok:
            mismatch_chosen += 1
            details.append((idx, "chosen", len(chosen_trainable), chosen_len, chosen_len))
        if not rejected_ok:
            mismatch_rejected += 1
            details.append((idx, "rejected", len(rejected_trainable), rejected_len, rejected_len))

        if chosen_trainable and not any(chosen_trainable):
            all_false_chosen += 1
        if rejected_trainable and not any(rejected_trainable):
            all_false_rejected += 1

    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    print(f"检查样本数: {len(samples)}")
    print(f"chosen mask 长度不匹配: {mismatch_chosen}")
    print(f"rejected mask 长度不匹配: {mismatch_rejected}")
    print(f"chosen 全 False (无 trainable): {all_false_chosen}")
    print(f"rejected 全 False (无 trainable): {all_false_rejected}")
    print()

    if details:
        print("不匹配详情 (前 10 个):")
        for d in details[:10]:
            print(f"  样本 {d[0]} {d[1]}: mask_len={d[2]}, expected={d[3]}")
    else:
        print("所有样本 mask 长度均满足要求。")

    # 额外：直接对比 tokenize 一致性（用第一个样本）
    print("\n" + "-" * 60)
    print("Tokenize 一致性检查 (prompt+chosen 整体 vs 分段)")
    s0 = samples[0]
    p, c = s0["prompt"], s0["chosen"]
    full_len = len(tokenizer.encode(p + c, add_special_tokens=False))
    seg_len = len(tokenizer.encode(p, add_special_tokens=False)) + len(tokenizer.encode(c, add_special_tokens=False))
    print(f"  整体 tokenize(prompt+chosen) 长度: {full_len}")
    print(f"  tokenize(prompt)+tokenize(chosen) 长度之和: {seg_len}")
    print(f"  是否一致: {full_len == seg_len}")


if __name__ == "__main__":
    main()
