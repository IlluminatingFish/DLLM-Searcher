#!/usr/bin/env python3
"""
Plan C: 在 latent think 空间做 RL。

核心创新：DPO 的 ELBO 对 think 段和 action 段使用不同的空间：
  - Action 段（tool_call + answer）：标准离散 token ELBO（同 Plan B）
  - Think 段：连续空间 ELBO，通过 x0_head 在 embedding 空间计算：
      ELBO_think = -E[||x0_head(h(z_noisy)) - embed(think_clean)||²]
    其中 z_noisy = embed(think) + σ * ε，σ ~ Uniform(0.1, 1.0)

  完整 DPO loss：
      L = -log σ(β * [ELBO_think_c + ELBO_act_c - ELBO_think_r - ELBO_act_r])

  Rollout 收集：使用 latent inference（DDPM denoising think → block diffusion action），
  产生比标准 sampling 更有目标性的 think 轨迹。

用法:
  accelerate launch --config_file recipes/accelerate_configs/zero3_cpu_offload.yaml \
    --num_processes 8 my_train/llada_dpo_train_c.py \
    --model_path /common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized \
    --data_path  data/dpo_pairs_plan_c_ref.jsonl \
    --output_dir /common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_c/model
"""

import argparse, json, os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from accelerate import Accelerator
from accelerate import logging

sys.path.insert(0, str(Path(__file__).parent))
from llada_dpo_trainer import (
    LLaDADPOTrainer, get_trainable_mask,
    DataCollatorForLLaDAPreference,
    llada_blk_forward_process, llada_mdm_ce_loss,
)

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

# staticmethod 补丁：解决 transformers 5.x tie_weights(missing_keys=...) 不兼容
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

logger = logging.get_logger(__name__)


# ── Think 段定位 ──────────────────────────────────────────────────────────────
def find_think_spans(token_ids, tokenizer):
    """返回 think 内容的 token 索引列表（不含 <think></think> 标签本身）。"""
    text    = tokenizer.decode(token_ids, skip_special_tokens=False)
    enc     = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    think_indices = []
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        s_char, e_char = m.start(1), m.end(1)
        for t_idx, (ts, te) in enumerate(offsets):
            if ts >= s_char and te <= e_char and t_idx < len(token_ids):
                think_indices.append(t_idx)
    return think_indices


# ── Latent ELBO for think segment ─────────────────────────────────────────────
def compute_latent_elbo_think(
    model, x0_head, full_ids, prompt_length,
    think_positions_in_full,   # absolute indices in full_ids
    num_mc=2, device="cuda"
):
    """
    对 think 段在连续 embedding 空间计算 ELBO：
      ELBO_think = -E_σ,ε [ ||x0_head(h(z_noisy)) - embed(think)||² ]

    σ 对每个 MC 样本从 [0.1, 1.0] 均匀采样。
    """
    if len(think_positions_in_full) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    embed_layer = model.get_input_embeddings()
    think_pos   = torch.tensor(think_positions_in_full, device=device)

    with torch.no_grad():
        x0_clean = embed_layer(full_ids[0, think_pos])  # [n, d]

    elbos = []
    for _ in range(num_mc):
        sigma = torch.empty(1, device=device).uniform_(0.1, 1.0).item()
        noise  = torch.randn_like(x0_clean)
        x0_noisy = (x0_clean + sigma * noise).detach()

        # Hook: 替换 think 位置的 embedding
        def inject_hook(module, inp, out):
            out_mod = out.clone()
            out_mod[0, think_pos] = x0_noisy.to(out.dtype)
            return out_mod

        # Hook: 捕获 final hidden state
        hidden = {}
        def capture_hook(module, inp, out):
            hidden["h"] = out

        # 找 ln_f
        ln_f = None
        for name, mod in model.named_modules():
            if hasattr(mod, "weight") and "norm" in name.lower() and mod.weight.shape[0] == model.config.hidden_size:
                # 最后一个 norm 层
                ln_f = mod

        if ln_f is None:
            # fallback: 不用 latent ELBO
            return torch.tensor(0.0, device=device, requires_grad=True)

        h1 = embed_layer.register_forward_hook(inject_hook)
        h2 = ln_f.register_forward_hook(capture_hook)

        try:
            # no_grad for model backbone: 不保存激活值，节省约 50% 内存
            # x0_head 仍有梯度（从 h_think.detach() 流入 x0_head.weight）
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(input_ids=full_ids)

            if "h" not in hidden:
                continue

            h_think  = hidden["h"][0, think_pos].detach()  # [n, d], detach from no_grad
            x0_pred  = x0_head(h_think.float())             # [n, d], x0_head 有梯度
            mse      = F.mse_loss(x0_pred, x0_clean.detach(), reduction="mean")
            # ELBO = -MSE × N/σ² (ELBO 上界，类比 VLB)
            elbo_i   = -mse * (len(think_positions_in_full) / (sigma ** 2 + 1e-8))
            elbos.append(elbo_i)
        except Exception as e:
            logger.warning(f"latent ELBO error: {e}")
        finally:
            h1.remove()
            h2.remove()

    if not elbos:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(elbos).mean()


# ── Plan C Trainer ────────────────────────────────────────────────────────────
class LLaDALatentDPOTrainer(LLaDADPOTrainer):
    """
    Plan C: Think 段用连续 ELBO（x0pred），Action 段用离散 ELBO。
    """

    def set_x0_head(self, x0_head):
        self.x0_head = x0_head

    def _get_elbo(self, model, full_ids, prompt_length, completion_length,
                  trainable_mask, seed):
        """
        覆写 _get_elbo：
          - think 段 → latent ELBO（连续 embedding 空间）
          - action 段 → 标准离散 ELBO
        """
        from accelerate.utils import set_seed
        set_seed(seed)

        device       = full_ids.device
        block_length = self.block_length
        completion   = full_ids[0, prompt_length:prompt_length + completion_length]

        # 找 think 段在 completion 里的位置
        think_local = find_think_spans(completion.tolist(), self.processing_class)
        # 转换为 full_ids 绝对位置
        think_in_full = [prompt_length + i for i in think_local]

        # Action 段 = completion 里非 think 的可训练部分
        action_trainable = trainable_mask.clone()
        for idx in think_in_full:
            if idx < len(action_trainable):
                action_trainable[idx] = False

        # ── 1. Latent ELBO for think ────────────────────────────────────────
        latent_elbo = torch.tensor(0.0, device=device)
        if think_in_full and hasattr(self, "x0_head"):
            try:
                latent_elbo = compute_latent_elbo_think(
                    model, self.x0_head.to(device),
                    full_ids, prompt_length, think_in_full,
                    num_mc=1, device=device
                )
            except Exception as e:
                logger.warning(f"latent ELBO failed: {e}")

        # ── 2. Discrete ELBO for action ─────────────────────────────────────
        import torch.nn.functional as F
        pad_len = (block_length - completion_length % block_length) % block_length
        full_ids_pad = full_ids
        if pad_len > 0:
            full_ids_pad = F.pad(full_ids, (0, pad_len), value=self.eos_token_id)
            action_trainable = torch.cat([
                action_trainable,
                torch.zeros(pad_len, dtype=torch.bool, device=device)
            ])

        num_comp_blocks = (completion_length + block_length - 1) // block_length
        noisy_ids = full_ids_pad.clone()

        block_ks, block_Ns = [], []
        for bi in range(num_comp_blocks):
            bs = prompt_length + bi * block_length
            be = min(bs + block_length, full_ids_pad.shape[1])
            block    = full_ids_pad[:, bs:be]
            block_tm = action_trainable[bs:be].unsqueeze(0)

            orig_len = be - bs
            if orig_len < block_length:
                block    = F.pad(block,    (0, block_length - orig_len), value=self.eos_token_id)
                block_tm = F.pad(block_tm, (0, block_length - orig_len), value=False)

            noisy_block, k, N = llada_blk_forward_process(
                block, block_tm, self.mask_token_id, self.eos_token_id
            )
            noisy_ids[:, bs:bs + orig_len] = noisy_block[:, :orig_len]
            block_ks.append(k)
            block_Ns.append(N)

        use_bf16 = True
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=noisy_ids).logits

        discrete_elbo = torch.tensor(0.0, device=device)
        for bi in range(num_comp_blocks):
            if block_Ns[bi] == 0 or block_ks[bi] == 0:
                continue
            bs = prompt_length + bi * block_length
            be = min(bs + block_length, full_ids_pad.shape[1])
            loss_blk = llada_mdm_ce_loss(
                logits[0, bs:be], full_ids_pad[0, bs:be],
                noisy_ids[0, bs:be], action_trainable[bs:be],
                N=block_Ns[bi], k=block_ks[bi]
            )
            discrete_elbo = discrete_elbo - loss_blk

        # 合并：latent think + discrete action
        total_elbo = latent_elbo + discrete_elbo
        return total_elbo


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--output_dir",  default="output/dpo_plan_c")
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=5e-7)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--num_mc",      type=int,   default=1)
    parser.add_argument("--block_length",type=int,   default=64)
    parser.add_argument("--max_length",  type=int,   default=2048)
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index

    data = [json.loads(l) for l in open(args.data_path) if l.strip()]
    dataset = Dataset.from_list(data)
    if rank == 0:
        print(f"DPO 数据 (latent rollouts): {len(dataset)} 对")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    _cfg.flash_attention = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=_cfg, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    if rank == 0:
        fa_count = sum(1 for m in model.modules() if getattr(m, "flash_attn_func", None) is not None)
        print(f"Flash Attention 已激活：{fa_count} 个注意力层")
    # ZeRO-3 内存不够同时存两个 8B 模型，使用预计算的 ref log probs（见 precompute_ref_logps.py）
    ref_model = None

    ckpt_dir = Path(args.model_path)
    x0_head  = None
    d_model  = model.config.hidden_size
    x0_path  = ckpt_dir / "x0_head.pt"
    if x0_path.exists():
        x0_head = nn.Linear(d_model, d_model, bias=False)
        x0_head.load_state_dict(torch.load(x0_path, map_location="cpu"))
        if rank == 0:
            print(f"加载 x0_head: {x0_path}")
    else:
        if rank == 0:
            print("[WARNING] x0_head.pt 未找到，Plan C 退化为标准 DPO")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        max_completion_length=args.max_length // 2,  # TRL 0.28 不自动计算
    )

    trainer = LLaDALatentDPOTrainer(
        model=model, ref_model=ref_model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
        block_length=args.block_length, num_mc=args.num_mc,
    )
    if x0_head is not None:
        trainer.set_x0_head(x0_head)

    if rank == 0:
        print(f"\nPlan C: Latent think ELBO + Discrete action ELBO DPO")
        print(f"lr={args.lr}  beta={args.beta}  epochs={args.epochs}\n")

    trainer.train()
    trainer.save_model(args.output_dir)
    if rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        if x0_head is not None:
            torch.save(x0_head.state_dict(), Path(args.output_dir) / "x0_head.pt")
        print(f"\n训练完成，模型保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
