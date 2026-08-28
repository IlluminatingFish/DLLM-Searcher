#!/usr/bin/env python3
"""
预计算参考模型的 log probs，保存到 DPO 数据文件中。
单卡运行，输出 {input}.ref.jsonl。

用法:
  CUDA_VISIBLE_DEVICES=0 python precompute_ref_logps.py \
    --model_path <model> --data_path <dpo_pairs.jsonl> --output <out.jsonl>
"""

import argparse, json, sys, os
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

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
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    return _orig_adjust(self, *args, **kwargs)
PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

from transformers import AutoTokenizer, AutoModelForCausalLM
from llada_dpo_trainer import (
    get_trainable_mask, llada_blk_forward_process, llada_mdm_ce_loss
)


def compute_elbo_single(
    model, tokenizer, prompt_text, completion_text,
    mask_token_id, eos_token_id,
    block_length=64, num_mc=2, device="cuda"
):
    """单个 (prompt, completion) 的 block-ELBO，用于参考模型预计算。"""
    prompt_ids     = tokenizer(prompt_text,     add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [eos_token_id]

    trainable_mask_list = get_trainable_mask(completion_text, tokenizer)
    trainable_mask_list = (trainable_mask_list + [True])[:len(completion_ids)]

    prompt_len  = len(prompt_ids)
    comp_len    = len(completion_ids)
    full_ids    = torch.tensor(prompt_ids + completion_ids, device=device).unsqueeze(0)

    prompt_zeros = [False] * prompt_len
    full_tm = torch.tensor(prompt_zeros + trainable_mask_list, dtype=torch.bool, device=device)

    elbos = []
    from accelerate.utils import set_seed
    for mc in range(num_mc):
        set_seed(mc * 137 + 42)
        pad_len = (block_length - comp_len % block_length) % block_length
        fids = full_ids
        ftm  = full_tm
        if pad_len > 0:
            fids = F.pad(fids, (0, pad_len), value=eos_token_id)
            ftm  = torch.cat([ftm, torch.zeros(pad_len, dtype=torch.bool, device=device)])

        num_blocks = (comp_len + block_length - 1) // block_length
        noisy_ids = fids.clone()
        block_ks, block_Ns = [], []

        for bi in range(num_blocks):
            bs = prompt_len + bi * block_length
            be = min(bs + block_length, fids.shape[1])
            block    = fids[:, bs:be]
            block_tm = ftm[bs:be].unsqueeze(0)
            orig_len = be - bs
            if orig_len < block_length:
                block    = F.pad(block,    (0, block_length - orig_len), value=eos_token_id)
                block_tm = F.pad(block_tm, (0, block_length - orig_len), value=False)

            noisy_block, k, N = llada_blk_forward_process(
                block, block_tm, mask_token_id, eos_token_id
            )
            noisy_ids[:, bs:bs + orig_len] = noisy_block[:, :orig_len]
            block_ks.append(k)
            block_Ns.append(N)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=noisy_ids).logits

        elbo = torch.tensor(0.0, device=device, dtype=torch.float32)
        for bi in range(num_blocks):
            if block_Ns[bi] == 0 or block_ks[bi] == 0:
                continue
            bs = prompt_len + bi * block_length
            be = min(bs + block_length, fids.shape[1])
            loss_blk = llada_mdm_ce_loss(
                logits[0, bs:be], fids[0, bs:be],
                noisy_ids[0, bs:be], ftm[bs:be],
                N=block_Ns[bi], k=block_ks[bi]
            )
            elbo = elbo - loss_blk

        N_total = max(sum(trainable_mask_list), 1)
        elbos.append((elbo / N_total).item())

    return sum(elbos) / len(elbos) if elbos else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--num_mc",       type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"预计算 ref log probs: {args.data_path}")
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    model = model.to(device)
    model.config.use_cache = False  # LLaDAConfig 没有此属性，手动补上
    model.eval()

    mask_token_id = (getattr(tokenizer, "mask_token_id", None)
                     or getattr(model.config, "mask_token_id", None))
    eos_token_id  = tokenizer.eos_token_id
    print(f"mask_token_id={mask_token_id}  eos_token_id={eos_token_id}")

    data = [json.loads(l) for l in open(args.data_path) if l.strip()]
    print(f"共 {len(data)} 对 DPO 数据")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for idx, ex in enumerate(data):
            chosen_lp = compute_elbo_single(
                model, tokenizer,
                ex["prompt"], ex["chosen"],
                mask_token_id, eos_token_id,
                args.block_length, args.num_mc, device
            )
            rejected_lp = compute_elbo_single(
                model, tokenizer,
                ex["prompt"], ex["rejected"],
                mask_token_id, eos_token_id,
                args.block_length, args.num_mc, device
            )
            ex["ref_chosen_logps"]   = chosen_lp
            ex["ref_rejected_logps"] = rejected_lp
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(data)}] chosen={chosen_lp:.4f}  rejected={rejected_lp:.4f}")
            torch.cuda.empty_cache()

    print(f"\n预计算完成，保存至: {out_path}")


if __name__ == "__main__":
    main()
