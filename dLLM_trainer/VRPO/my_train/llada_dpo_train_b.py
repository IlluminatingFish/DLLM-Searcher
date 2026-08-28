#!/usr/bin/env python3
"""
Plan B: DPO + x0pred 正则化。

原理：
  L_total = L_DPO + λ * L_x0pred

  L_DPO: 标准 block-ELBO DPO（preferred > rejected）
  L_x0pred: 对 think 段，用 x0_head 预测干净 embedding，防止 RL 遗忘连续 think 空间

用法:
  accelerate launch --config_file recipes/accelerate_configs/zero3_cpu_offload.yaml \
    --num_processes 8 my_train/llada_dpo_train_b.py \
    --model_path /common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized \
    --data_path  data/dpo_pairs_plan_ab_ref.jsonl \
    --output_dir /common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_b/model \
    --lambda_x0 0.1
"""

import argparse, json, os, sys, re
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
from llada_dpo_trainer import LLaDADPOTrainer, get_trainable_mask

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


# ── x0pred 正则化 loss ────────────────────────────────────────────────────────
THINK_START_PATTERN = re.compile(r"<think>")
THINK_END_PATTERN   = re.compile(r"</think>")

def find_think_mask(token_ids, tokenizer):
    """找到 think 段的位置 mask（True = think 内容位置）。"""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    think_mask = torch.zeros(len(token_ids), dtype=torch.bool)

    # 粗略：用字符偏移找 think 段，然后映射回 token 位置
    pos = 0
    in_think = False
    think_start = -1
    char_to_token = {}

    # 用 offset_mapping 精确对齐
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    for t_idx, (start, end) in enumerate(offsets):
        char_to_token[start] = t_idx

    # 扫描文本里的 <think>...</think> 区间
    for m in THINK_START_PATTERN.finditer(text):
        start_char = m.end()  # think 内容从 <think> 结束后开始
        m_end = THINK_END_PATTERN.search(text, start_char)
        if m_end is None:
            break
        end_char = m_end.start()  # 到 </think> 开始前结束

        # 把字符区间映射到 token 区间
        for t_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= start_char and tok_end <= end_char:
                if t_idx < len(think_mask):
                    think_mask[t_idx] = True

    return think_mask


def compute_x0pred_reg_loss(model, x0_head, sigma_proj, batch, tokenizer, mask_id, device):
    """
    对 chosen 序列中的 think 段计算 x0pred 正则化 loss：
      1. 取 think tokens 的 embedding（干净目标 x0_clean）
      2. 加高斯噪声模拟 diffusion（sigma = 0.5）
      3. 过 transformer 得到 hidden state
      4. x0_head 预测 → MSE with x0_clean
    """
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    embed_layer = model.get_input_embeddings()
    sigma = torch.tensor(0.5, device=device)

    for i in range(len(batch["prompt_input_ids"])):
        prompt_ids  = batch["prompt_input_ids"][i].to(device)
        chosen_ids  = batch["chosen_input_ids"][i].to(device)
        full_ids    = torch.cat([prompt_ids, chosen_ids]).unsqueeze(0)  # [1, L]

        # 找 think 段
        chosen_text = tokenizer.decode(chosen_ids.tolist(), skip_special_tokens=False)
        think_mask  = find_think_mask(chosen_ids.tolist(), tokenizer).to(device)  # [chosen_len]

        if think_mask.sum() == 0:
            continue

        # 在 full sequence 里的位置
        full_think_mask = torch.cat([
            torch.zeros(len(prompt_ids), dtype=torch.bool, device=device),
            think_mask
        ])  # [L]

        think_positions = full_think_mask.nonzero(as_tuple=True)[0]
        if len(think_positions) == 0:
            continue

        # 干净 embedding（目标）
        with torch.no_grad():
            x0_clean = embed_layer(full_ids[0, think_positions])  # [n_think, d]

        # 加噪声
        noise   = torch.randn_like(x0_clean)
        x0_noisy = x0_clean + sigma * noise

        # 构造 noisy input
        noisy_ids = full_ids.clone().float()  # 先转 float 用于 embed 替换
        # 替换 think 位置的 embedding → 用 noisy embedding 过 transformer
        # 通过 forward hook 注入
        injected = {}
        def inject_hook(module, inp, out):
            out_new = out.clone()
            out_new[0, think_positions] = x0_noisy.to(out.dtype)
            injected["done"] = True
            return out_new

        handle = embed_layer.register_forward_hook(inject_hook)

        # 捕获 think 位置的 final hidden state
        hidden_states = {}
        ln_f = None
        for name, module in model.named_modules():
            if "ln_f" in name or name.endswith("norm") and "ln" in name.lower():
                ln_f = module
                break

        if ln_f is None:
            handle.remove()
            continue

        def capture_hook(module, inp, out):
            hidden_states["h"] = out

        handle2 = ln_f.register_forward_hook(capture_hook)

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(input_ids=full_ids)

            if "h" not in hidden_states:
                continue

            h_think = hidden_states["h"][0, think_positions]  # [n_think, d]
            x0_pred = x0_head(h_think.float())                # [n_think, d]

            loss = F.mse_loss(x0_pred, x0_clean.detach())
            total_loss = total_loss + loss
            count += 1

        except Exception as e:
            logger.warning(f"x0pred reg loss error: {e}")
        finally:
            handle.remove()
            handle2.remove()

    if count > 0:
        total_loss = total_loss / count
    return total_loss


# ── Plan B Trainer ────────────────────────────────────────────────────────────
class LLaDAX0RegDPOTrainer(LLaDADPOTrainer):
    """LLaDA DPO + x0pred 正则化。"""

    def set_x0_components(self, x0_head, sigma_proj, lambda_x0=0.1):
        self.x0_head   = x0_head
        self.sigma_proj = sigma_proj
        self.lambda_x0 = lambda_x0

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        dpo_loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        # x0pred 正则化
        if hasattr(self, "x0_head") and self.lambda_x0 > 0:
            try:
                device = next(model.parameters()).device
                x0_loss = compute_x0pred_reg_loss(
                    model, self.x0_head.to(device),
                    self.sigma_proj.to(device) if self.sigma_proj is not None else None,
                    batch, self.processing_class,
                    self.mask_token_id, device
                )
                total_loss = dpo_loss + self.lambda_x0 * x0_loss
                prefix = "eval_" if train_eval == "eval" else ""
                metrics[f"{prefix}x0pred_loss"] = x0_loss.item()
            except Exception as e:
                logger.warning(f"x0pred reg failed: {e}")
                total_loss = dpo_loss
        else:
            total_loss = dpo_loss

        return total_loss, metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--output_dir",  default="output/dpo_plan_b")
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=5e-7)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--num_mc",      type=int,   default=1)
    parser.add_argument("--block_length",type=int,   default=64)
    parser.add_argument("--lambda_x0",  type=float, default=0.1, help="x0pred 正则化权重")
    parser.add_argument("--max_length",  type=int,   default=2048)
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index

    data = [json.loads(l) for l in open(args.data_path) if l.strip()]
    dataset = Dataset.from_list(data)
    if rank == 0:
        print(f"DPO 数据: {len(dataset)} 对")

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

    # 加载 x0pred head
    ckpt_dir    = Path(args.model_path)
    x0_head     = None
    sigma_proj  = None
    x0_path     = ckpt_dir / "x0_head.pt"
    sigma_path  = ckpt_dir / "sigma_proj.pt"

    d_model = model.config.hidden_size
    if x0_path.exists():
        x0_head = nn.Linear(d_model, d_model, bias=False)
        x0_head.load_state_dict(torch.load(x0_path, map_location="cpu"))
        if rank == 0:
            print(f"加载 x0_head: {x0_path}")
    if sigma_path.exists():
        sigma_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        sigma_proj.load_state_dict(torch.load(sigma_path, map_location="cpu"))
        if rank == 0:
            print(f"加载 sigma_proj: {sigma_path}")

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

    trainer = LLaDAX0RegDPOTrainer(
        model=model, ref_model=ref_model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
        block_length=args.block_length, num_mc=args.num_mc,
    )
    trainer.set_x0_components(x0_head, sigma_proj, lambda_x0=args.lambda_x0)

    if rank == 0:
        print(f"\nPlan B: DPO + x0pred regularization (λ={args.lambda_x0})")
        print(f"lr={args.lr}  beta={args.beta}  epochs={args.epochs}\n")

    trainer.train()
    trainer.save_model(args.output_dir)
    if rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        if x0_head is not None:
            torch.save(x0_head.state_dict(), Path(args.output_dir) / "x0_head.pt")
        if sigma_proj is not None:
            torch.save(sigma_proj.state_dict(), Path(args.output_dir) / "sigma_proj.pt")
        print(f"\n训练完成，模型保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
