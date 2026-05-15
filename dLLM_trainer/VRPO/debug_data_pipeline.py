#!/usr/bin/env python3
"""
诊断脚本：检查数据读入 + tokenize_row 后的状态是否正常。
模拟训练时的 tokenize 流程，检查 mask 长度、全 False、block 级 trainable 等。
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def get_trainable_mask(completion_text, tokenizer, tool_left, tool_right):
    """与 my_dpo_trainer 完全一致"""
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

TOOL_LEFT = "<|im_start|>user\n<tool_response>"
TOOL_RIGHT = "</tool_response><|im_end|>\n<|im_start|>assistant\n"
MODEL_PATH = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_1/optimized"
DATA_PATH = "data/train.jsonl"
MAX_PROMPT = 648
MAX_COMPLETION = 2048
BLOCK_LENGTH = 128


def tokenize_row_like_trainer(features, tokenizer):
    """与 tokenize_row 完全一致的逻辑"""
    prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    if MAX_PROMPT:
        prompt_input_ids = prompt_input_ids[-MAX_PROMPT:]
    if MAX_COMPLETION:
        chosen_input_ids = chosen_input_ids[:MAX_COMPLETION]
        rejected_input_ids = rejected_input_ids[:MAX_COMPLETION]

    chosen_trainable_mask = get_trainable_mask(
        features["chosen"], tokenizer, TOOL_LEFT, TOOL_RIGHT
    )
    chosen_trainable_mask = chosen_trainable_mask + [True]
    chosen_trainable_mask = chosen_trainable_mask[:len(chosen_input_ids)]

    rejected_trainable_mask = get_trainable_mask(
        features["rejected"], tokenizer, TOOL_LEFT, TOOL_RIGHT
    )
    rejected_trainable_mask = rejected_trainable_mask + [True]
    rejected_trainable_mask = rejected_trainable_mask[:len(rejected_input_ids)]

    return {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        "chosen_trainable_mask": chosen_trainable_mask,
        "rejected_trainable_mask": rejected_trainable_mask,
    }


def check_block_trainable(prompt_len, completion_len, trainable_mask):
    """
    模拟 _get_elbo_blk_with_trainable 的 block 划分，检查每个 block 是否有 trainable。
    若某 block 全 False，blk_forward_process_with_trainable 会 num_masks=0，该 block 被 skip。
    """
    num_blocks = (prompt_len + completion_len + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    prefill_blocks = prompt_len // BLOCK_LENGTH
    prefill_length = prefill_blocks * BLOCK_LENGTH
    num_completion_blocks = num_blocks - prefill_blocks

    # 与 trainer 一致：block 在完整序列中的边界
    blocks_with_zero_trainable = []
    for j in range(num_completion_blocks):
        if j == 0:
            block_start_full = prompt_len
            block_end_full = (prefill_blocks + 1) * BLOCK_LENGTH
        else:
            block_start_full = prefill_length + j * BLOCK_LENGTH
            block_end_full = block_start_full + BLOCK_LENGTH

        # completion 在完整序列中的区间是 [prompt_len, prompt_len+completion_len)
        # block 对应的 completion 区间
        c_start = block_start_full - prompt_len
        c_end = block_end_full - prompt_len
        c_start = max(0, c_start)
        c_end = min(len(trainable_mask), c_end)
        if c_start >= c_end:
            continue
        block_mask = trainable_mask[c_start:c_end]
        if sum(block_mask) == 0:
            blocks_with_zero_trainable.append(j)
    return blocks_with_zero_trainable


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading data...")
    samples = []
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}\n")
    print("=" * 70)
    print("检查 1: mask 长度是否与 input_ids 一致")
    print("=" * 70)

    len_mismatch_chosen = []
    len_mismatch_rejected = []
    all_false_chosen = []
    all_false_rejected = []
    block_zero_trainable_chosen = []
    block_zero_trainable_rejected = []

    for idx, s in enumerate(samples):
        row = tokenize_row_like_trainer(s, tokenizer)
        c_ids = row["chosen_input_ids"]
        r_ids = row["rejected_input_ids"]
        c_mask = row["chosen_trainable_mask"]
        r_mask = row["rejected_trainable_mask"]

        if len(c_mask) != len(c_ids):
            len_mismatch_chosen.append((idx, len(c_mask), len(c_ids)))
        if len(r_mask) != len(r_ids):
            len_mismatch_rejected.append((idx, len(r_mask), len(r_ids)))

        if c_mask and not any(c_mask):
            all_false_chosen.append(idx)
        if r_mask and not any(r_mask):
            all_false_rejected.append(idx)

        prompt_len = len(row["prompt_input_ids"])
        chosen_len = len(c_ids)
        rejected_len = len(r_ids)

        bad_c = check_block_trainable(prompt_len, chosen_len, c_mask)
        bad_r = check_block_trainable(prompt_len, rejected_len, r_mask)
        if bad_c:
            block_zero_trainable_chosen.append((idx, bad_c))
        if bad_r:
            block_zero_trainable_rejected.append((idx, bad_r))

    print(f"chosen mask 长度不匹配: {len(len_mismatch_chosen)}")
    if len_mismatch_chosen:
        for t in len_mismatch_chosen[:5]:
            print(f"  样本 {t[0]}: mask_len={t[1]}, ids_len={t[2]}")

    print(f"rejected mask 长度不匹配: {len(len_mismatch_rejected)}")
    if len_mismatch_rejected:
        for t in len_mismatch_rejected[:5]:
            print(f"  样本 {t[0]}: mask_len={t[1]}, ids_len={t[2]}")

    print(f"\nchosen 全 False: {len(all_false_chosen)}")
    if all_false_chosen:
        print(f"  样本索引: {all_false_chosen[:10]}...")

    print(f"rejected 全 False: {len(all_false_rejected)}")
    if all_false_rejected:
        print(f"  样本索引: {all_false_rejected[:10]}...")

    print("\n" + "=" * 70)
    print("检查 2: 是否存在 completion block 内 trainable=0（会导致 blk_num_masks=0）")
    print("=" * 70)

    print(f"chosen 有 block 全 False 的样本数: {len(block_zero_trainable_chosen)}")
    if block_zero_trainable_chosen:
        for t in block_zero_trainable_chosen[:5]:
            print(f"  样本 {t[0]}: block indices {t[1]}")

    print(f"rejected 有 block 全 False 的样本数: {len(block_zero_trainable_rejected)}")
    if block_zero_trainable_rejected:
        for t in block_zero_trainable_rejected[:5]:
            print(f"  样本 {t[0]}: block indices {t[1]}")

    print("\n" + "=" * 70)
    print("检查 3: Step 0 vs Step 1 的样本对比（8 GPU × 8 acc = 64 samples/step）")
    print("=" * 70)

    # 训练时通常有 shuffle，这里假设按顺序取（无 shuffle 时）
    step0_indices = list(range(64))
    step1_indices = list(range(64, 128))

    def summarize(indices, name):
        issues = []
        for i in indices:
            if i >= len(samples):
                break
            row = tokenize_row_like_trainer(samples[i], tokenizer)
            c_mask, r_mask = row["chosen_trainable_mask"], row["rejected_trainable_mask"]
            if len(c_mask) != len(row["chosen_input_ids"]):
                issues.append(f"c_len_mismatch")
            if len(r_mask) != len(row["rejected_input_ids"]):
                issues.append(f"r_len_mismatch")
            if c_mask and not any(c_mask):
                issues.append("c_all_false")
            if r_mask and not any(r_mask):
                issues.append("r_all_false")
            bad_c = check_block_trainable(len(row["prompt_input_ids"]), len(row["chosen_input_ids"]), c_mask)
            bad_r = check_block_trainable(len(row["prompt_input_ids"]), len(row["rejected_input_ids"]), r_mask)
            if bad_c:
                issues.append("c_block_zero")
            if bad_r:
                issues.append("r_block_zero")
        return issues

    s0_issues = [summarize([i], "x") for i in step0_indices if i < len(samples)]
    s1_issues = [summarize([i], "x") for i in step1_indices if i < len(samples)]

    s0_has_issue = sum(1 for x in s0_issues if x)
    s1_has_issue = sum(1 for x in s1_issues if x)

    print(f"Step 0 样本 (0-63): 有问题的样本数 = {s0_has_issue}")
    print(f"Step 1 样本 (64-127): 有问题的样本数 = {s1_has_issue}")

    if s1_has_issue > 0 and s0_has_issue == 0:
        print("\n>>> Step 1 有异常样本而 Step 0 无，与 NaN 现象一致！")
        print("问题样本索引 (step1):")
        for i in step1_indices:
            if i >= len(samples):
                break
            issues = summarize([i], "x")[0]
            if issues:
                print(f"  样本 {i}: {issues}")

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
