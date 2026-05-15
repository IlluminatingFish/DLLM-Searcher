#!/usr/bin/env python3
"""
NaN 根因排查：检查可能造成 model logits NaN 的若干原因。
运行: cd VRPO && python debug_nan_causes.py
"""
import torch

BLOCK_LENGTH = 128


def make_basic_block_attention(N: int, start_pos: int, block_size: int) -> torch.Tensor:
    """与 my_dpo_trainer 完全一致"""
    L0 = start_pos
    L1 = (N - L0) // 2
    assert L0 + 2 * L1 == N, f"N={N}, L0={L0}, L1={L1}"
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


def check_mask_all_false_rows(mask: torch.Tensor, name: str) -> None:
    """检查是否存在整行全 False 的 query 行（会导致 softmax NaN）"""
    # mask: [1, 1, N, N]，取 [0,0,:,:]
    m = mask[0, 0]  # [N, N]
    if m.dtype == torch.bool:
        row_sums = m.sum(dim=1)  # [N]
    else:
        row_sums = (m > 0).sum(dim=1)
    all_false_rows = (row_sums == 0).nonzero(as_tuple=True)[0].tolist()
    if all_false_rows:
        print(f"[!] {name}: 存在 {len(all_false_rows)} 行全 False → 这些 query 行 softmax 会 NaN")
        print(f"    行索引（前20个）: {all_false_rows[:20]}")
    else:
        print(f"[OK] {name}: 无全 False 行")


def main():
    print("=" * 70)
    print("NaN 根因排查")
    print("=" * 70)

    # 模拟训练时的典型参数
    max_prompt = 648
    max_completion = 2048
    block_length = BLOCK_LENGTH

    # 与 _get_elbo_blk_with_trainable 一致
    prompt_length = 560  # 示例
    completion_length = 2049  # 含 EOS
    batch = torch.zeros(1, prompt_length + completion_length + block_length - 1, dtype=torch.long)
    num_blocks = (prompt_length + completion_length + block_length - 1) // block_length
    prompt_completion_len_with_pad = num_blocks * block_length
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    completion_len_with_pad = prompt_completion_len_with_pad - prefill_length
    N = prompt_completion_len_with_pad + completion_len_with_pad

    print(f"\n参数: prompt_len={prompt_length} completion_len={completion_length}")
    print(f"      prefill_length={prefill_length} N={N}")

    # 原因 1: attention mask 是否存在全 False 行
    print("\n--- 原因 1: attention mask 全 False 行（→ softmax NaN）---")
    mask = make_basic_block_attention(N, prefill_length, block_length)
    mask_bool = mask.to(dtype=torch.bool)
    check_mask_all_false_rows(mask_bool, "make_basic_block_attention")

    # 多组参数扫描
    print("\n--- 扫描多组 (prompt_len, completion_len) ---")
    test_cases = [
        (560, 2049),
        (648, 2048),
        (0, 256),
        (128, 256),
        (256, 512),
    ]
    for plen, clen in test_cases:
        n_blocks = (plen + clen + block_length - 1) // block_length
        pc_pad = n_blocks * block_length
        pf = (plen // block_length) * block_length
        comp_pad = pc_pad - pf
        n = pc_pad + comp_pad
        try:
            m = make_basic_block_attention(n, pf, block_length).to(torch.bool)
            row_sums = m[0, 0].sum(dim=1)
            bad = (row_sums == 0).sum().item()
            status = "BAD" if bad > 0 else "OK"
            print(f"  prompt={plen} comp={clen} N={n}: {status} (全False行数={bad})")
        except Exception as e:
            print(f"  prompt={plen} comp={clen}: ERROR {e}")

    # 原因 2: bf16 数值范围
    print("\n--- 原因 2: bf16 精度 ---")
    print("  bf16 max ≈ 3.4e38, min normal ≈ 1.2e-38")
    print("  attention score 过大时 softmax 可能溢出")
    print("  建议: 试 mixed_precision='fp16' 或 'no'")

    # 原因 3: gradient checkpointing
    print("\n--- 原因 3: gradient_checkpointing ---")
    print("  某些实现下 checkpoint 重算时数值可能略有差异")
    print("  建议: 试 --gradient_checkpointing false")

    # 原因 4: 序列长度
    print("\n--- 原因 4: 序列长度 ---")
    total = prompt_length + completion_length
    print(f"  当前 total≈{total}，超长序列可能累积误差")
    print("  建议: 试 --max_completion_length 1024")

    # 原因 5: 防御性修复
    print("\n--- 原因 5: 防御性修复（已加）---")
    print("  已在 modeling_sdar.py SDARAttention 中加入全 -inf 行修复")
    print("  （SDPA 对整行 -inf 会出 NaN，将其替换为 0）")

    print("\n" + "=" * 70)
    print("建议排查顺序: 1) 试 fp32 2) 关 gradient_checkpointing 3) 缩短 max_completion_length")
    print("=" * 70)


if __name__ == "__main__":
    main()
