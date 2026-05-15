#!/usr/bin/env python3
"""
NaN 排查脚本：按数据流顺序依次检查各 NaN 检测点。
不依赖 trl/accelerate，仅用 transformers + torch 做单次 forward 诊断。
运行: cd VRPO && python debug_nan_forward.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 添加路径以便可选导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 从 my_dpo_trainer 复制的核心逻辑（避免 trl 依赖）==========
TOOL_LEFT = "<|im_start|>user\n<tool_response>"
TOOL_RIGHT = "</tool_response><|im_end|>\n<|im_start|>assistant\n"
MODEL_PATH = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_1/optimized"
DATA_PATH = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/train.jsonl"
BLOCK_LENGTH = 128
NUM_MC = 2
MASK_TOKEN_ID = 128002


def get_trainable_mask(completion_text, tokenizer, tool_left=TOOL_LEFT, tool_right=TOOL_RIGHT):
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


def make_basic_block_attention(N: int, start_pos: int, block_size: int) -> torch.Tensor:
    L0 = start_pos
    L1 = (N - L0) // 2
    assert L0 + 2 * L1 == N
    bias = torch.full((1, 1, N, N), 0)
    rows = torch.arange(L0 + L1, L0 + 2 * L1)
    rows_token = torch.arange(L0, L0 + L1)
    for bi in range((L1 + block_size - 1) // block_size):
        left_end = L0 + min(bi * block_size, L1)
        right_start = L0 + L1 + (left_end - L0)
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, L1)
        block_rows = rows[i_start:i_end]
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
        bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1
        block_rows = rows_token[i_start:i_end]
        left_end = L0 + min((bi + 1) * block_size, L1)
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
    if L0 > 0:
        num_blocks_pre = (L0 + block_size - 1) // block_size
        for bi in range(num_blocks_pre):
            row_end = max(L0 - bi * block_size, 0)
            row_start = max(L0 - (bi + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end)
                bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
    return bias


def blk_forward_process_with_trainable(batch, trainable_mask, mask_id, eos_id):
    b, l = batch.shape
    device = batch.device
    noisy_batch = batch.clone()
    num_masks = torch.zeros(b, dtype=torch.long, device=device)
    valid_block = torch.zeros(b, dtype=torch.bool, device=device)
    for i in range(b):
        block_trainable = trainable_mask[i]
        num_trainable = block_trainable.sum().item()
        if num_trainable == 0:
            valid_block[i] = False
            num_masks[i] = 0
            continue
        valid_block[i] = True
        trainable_indices = torch.where(block_trainable)[0]
        last_trainable = trainable_indices[-1].item()
        if last_trainable < l - 1:
            noisy_batch[i, last_trainable + 1:] = eos_id
        k = torch.randint(1, num_trainable + 1, (), device=device).item()
        perm = torch.randperm(num_trainable, device=device)
        mask_indices = trainable_indices[perm[:k]]
        noisy_batch[i, mask_indices] = mask_id
        num_masks[i] = k
    return noisy_batch, num_masks, valid_block


def mdm_ce_loss_with_trainable(logits, clean_batch, noisy_batch, trainable_mask, block_lengths, num_trainable_masks):
    b, l, v = logits.shape
    device = logits.device
    dtype = logits.dtype
    loss = torch.zeros(b, device=device, dtype=dtype)
    mask_index = clean_batch != noisy_batch
    valid_loss_mask = mask_index & trainable_mask
    for i in range(b):
        if num_trainable_masks[i] == 0:
            continue
        valid_positions = valid_loss_mask[i]
        if valid_positions.sum() == 0:
            continue
        ce_loss = F.cross_entropy(
            logits[i][valid_positions],
            clean_batch[i][valid_positions],
            reduction="sum"
        )
        loss[i] = ce_loss / num_trainable_masks[i] * block_lengths[i]
        # CHECK 2: mdm_ce_loss
        if torch.isnan(ce_loss) or torch.isnan(loss[i]):
            targets = clean_batch[i][valid_positions]
            vocab_size = logits.shape[-1]
            print(f"[NaN] CHECK 2 mdm_ce_loss: i={i} ce_loss={ce_loss.item()} loss_i={loss[i].item()} "
                  f"valid_cnt={valid_positions.sum().item()} num_masks={num_trainable_masks[i].item()} "
                  f"logits_nan={torch.isnan(logits[i]).any().item()} logits_inf={torch.isinf(logits[i]).any().item()} "
                  f"targets_min={targets.min().item()} targets_max={targets.max().item()} vocab_size={vocab_size}")
    return loss


def get_elbo_blk(model, input_ids, prompt_length, completion_length, trainable_mask, mask_seeds, eos_id):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    batch = F.pad(input_ids, (0, BLOCK_LENGTH - 1), value=eos_id)
    trainable_mask_padded = F.pad(trainable_mask, (0, BLOCK_LENGTH - 1), value=False)

    num_blocks = (prompt_length + completion_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    prompt_completion_len_with_pad = num_blocks * BLOCK_LENGTH
    prefill_blocks = prompt_length // BLOCK_LENGTH
    prefill_length = prefill_blocks * BLOCK_LENGTH
    completion_len_with_pad = prompt_completion_len_with_pad - prefill_length
    num_completion_blocks = num_blocks - prefill_blocks

    blk_num_masks = torch.zeros((num_completion_blocks, batch_size), dtype=torch.long, device=device)
    blk_trainable_lengths = torch.zeros((num_completion_blocks, batch_size), dtype=torch.long, device=device)
    noisy_batch = batch.clone()

    for i in range(batch_size):
        torch.manual_seed(mask_seeds[i].item())
        for j in range(num_completion_blocks):
            if j == 0:
                block_start = prompt_length
                block_end = (prefill_blocks + 1) * BLOCK_LENGTH
            else:
                block_start = prefill_length + j * BLOCK_LENGTH
                block_end = block_start + BLOCK_LENGTH
            block_content = batch[i:i+1, block_start:block_end]
            block_trainable = trainable_mask_padded[block_start:block_end].unsqueeze(0)
            noisy_block, num_masks, _ = blk_forward_process_with_trainable(
                block_content, block_trainable, MASK_TOKEN_ID, eos_id
            )
            noisy_batch[i:i+1, block_start:block_end] = noisy_block
            blk_num_masks[j, i] = num_masks[0]
            blk_trainable_lengths[j, i] = block_trainable.sum()

    batch_concat = torch.cat(
        (batch[:, :prompt_completion_len_with_pad],
         noisy_batch[:, prefill_length:prompt_completion_len_with_pad]),
        dim=1
    )
    attn_mask = make_basic_block_attention(
        N=prompt_completion_len_with_pad + completion_len_with_pad,
        start_pos=prefill_length,
        block_size=BLOCK_LENGTH,
    ).to(dtype=torch.bool, device=device)

    # Forward
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=batch_concat, attention_mask=attn_mask).logits

    # CHECK 1: model logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"[NaN] CHECK 1 model logits: logits_nan={torch.isnan(logits).any().item()} "
              f"logits_inf={torch.isinf(logits).any().item()} shape={logits.shape}")

    loss_expand = torch.zeros((batch_size, num_completion_blocks), device=device, dtype=torch.float32)
    block_boundaries = []
    for j in range(num_completion_blocks):
        if j == 0:
            block_boundaries.append((prompt_length, (prefill_blocks + 1) * BLOCK_LENGTH))
        else:
            start = prefill_length + j * BLOCK_LENGTH
            block_boundaries.append((start, start + BLOCK_LENGTH))

    for idx in range(num_completion_blocks):
        block_start, block_end = block_boundaries[idx]
        actual_block_len = block_end - block_start
        if blk_num_masks[idx].sum() == 0:
            continue
        logits_offset = block_start - prefill_length
        logits_start = prompt_completion_len_with_pad + logits_offset
        logits_end = logits_start + actual_block_len
        block_logits = logits[:, logits_start:logits_end, :]
        if block_logits.shape[1] < BLOCK_LENGTH:
            block_logits = F.pad(block_logits, (0, 0, 0, BLOCK_LENGTH - block_logits.shape[1]), value=0)
        clean_block = batch_concat[:, block_start:block_end]
        if clean_block.shape[1] < BLOCK_LENGTH:
            clean_block = F.pad(clean_block, (0, BLOCK_LENGTH - clean_block.shape[1]), value=eos_id)
        noisy_block = batch_concat[:, logits_start:logits_end]
        if noisy_block.shape[1] < BLOCK_LENGTH:
            noisy_block = F.pad(noisy_block, (0, BLOCK_LENGTH - noisy_block.shape[1]), value=eos_id)
        block_trainable = trainable_mask_padded[block_start:block_end]
        if len(block_trainable) < BLOCK_LENGTH:
            block_trainable = F.pad(block_trainable, (0, BLOCK_LENGTH - len(block_trainable)), value=False)
        block_trainable = block_trainable.unsqueeze(0).expand(batch_size, -1)
        block_lengths = blk_trainable_lengths[idx].float().clamp(min=1)
        local_loss = mdm_ce_loss_with_trainable(
            block_logits, clean_block, noisy_block, block_trainable,
            block_lengths, blk_num_masks[idx],
        )
        # CHECK 3: local_loss
        if torch.isnan(local_loss).any() or torch.isinf(local_loss).any():
            print(f"[NaN] CHECK 3 local_loss: block_idx={idx} local_loss={local_loss.tolist()} "
                  f"blk_num_masks={blk_num_masks[idx].tolist()}")
        loss_expand[:, idx] = -local_loss

    loss = loss_expand.sum(dim=-1)
    # CHECK 4: final loss
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"[NaN] CHECK 4 final loss: loss={loss.tolist()} loss_expand_nan={torch.isnan(loss_expand).any().item()}")
    return loss


def main():
    print("=" * 70)
    print("NaN 排查：按顺序检查 1→4")
    print("=" * 70)
    print("CHECK 1: model logits (forward 输出)")
    print("CHECK 2: mdm_ce_loss (cross_entropy 结果)")
    print("CHECK 3: local_loss (每 block 的 loss)")
    print("CHECK 4: final loss (ELBO 汇总)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("需要 CUDA，跳过 model forward")
        return

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    print("Loading data...")
    with open(DATA_PATH) as f:
        samples = [json.loads(line) for line in f]
    s = samples[0]

    prompt_ids = tokenizer(s["prompt"], add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(s["chosen"], add_special_tokens=False)["input_ids"]
    chosen_ids = chosen_ids[:2048] + [eos_id]
    prompt_ids = prompt_ids[-648:]
    chosen_mask = get_trainable_mask(s["chosen"], tokenizer)
    chosen_mask = (chosen_mask + [True])[:len(chosen_ids)]

    prompt_len = len(prompt_ids)
    chosen_len = len(chosen_ids)
    full_ids = prompt_ids + chosen_ids
    full_trainable = [False] * prompt_len + chosen_mask

    print(f"prompt_len={prompt_len} chosen_len={chosen_len} total={len(full_ids)}")

    print("\nLoading model (可能较慢，需 flash_attn)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    except ImportError as e:
        if "flash_attn" in str(e):
            print("\n[跳过] SDAR 模型需 flash_attn，当前环境未安装。")
            print("请在训练环境（有 flash_attn）下运行完整训练，用 grep '[NaN]' 日志定位。")
            print("数据层检查已通过（见 debug_data_pipeline.py 输出）。")
            return
        raise
    model.eval()

    input_ids = torch.tensor([full_ids] * NUM_MC, device=device, dtype=torch.long)
    trainable_mask = torch.tensor(full_trainable, device=device, dtype=torch.bool)
    mask_seeds = torch.randint(0, 2**20, (NUM_MC,), device=device)

    print("\nRunning forward (检查各 CHECK 点)...")
    with torch.no_grad():
        elbo = get_elbo_blk(model, input_ids, prompt_len, chosen_len, trainable_mask, mask_seeds, eos_id)

    chosen_lp = elbo.mean() / max(sum(chosen_mask), 1)
    print(f"\n最终 chosen_logp={chosen_lp.item():.6f} (nan={torch.isnan(chosen_lp).item()})")

    if not (torch.isnan(elbo).any() or torch.isinf(elbo).any()):
        print("\n所有 CHECK 均未触发 NaN → 若训练仍出现 NaN，可能是：")
        print("  - 多 GPU/分布式 差异")
        print("  - gradient checkpointing")
        print("  - 不同 batch 样本")
        print("  - 多 step 累积")

    print("\n" + "=" * 70)
    print("排查完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
