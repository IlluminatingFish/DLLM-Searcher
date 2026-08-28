"""
LLaDA DPO Trainer — adapts SDAR's block-ELBO DPO for LLaDA's single-sequence architecture.

Key difference from SDAR (my_dpo_trainer.py):
  SDAR: forward input = [clean_prompt+completion | noisy_completion]  (double-length)
        uses custom make_basic_block_attention
  LLaDA: forward input = [prompt | noisy_completion]                  (single-length)
         does NOT pass attention_mask (LLaDA uses its own block-causal attention internally)

Everything else — trainable mask, block-level ELBO, Monte Carlo averaging, DPO loss —
follows the same design as SDAR.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import pad
from typing import Dict, List, Optional, Any, Union, Literal
from accelerate.utils import set_seed
from accelerate import logging
from contextlib import nullcontext
from torch import autocast
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

logger = logging.get_logger(__name__)


# =============================================================================
# Trainable Mask  (identical to SDAR version)
# =============================================================================

def get_trainable_mask(
    completion_text: str,
    tokenizer: PreTrainedTokenizerBase,
    tool_resp_left: str = "<|im_start|>user\n<tool_response>",
    tool_resp_right: str = "</tool_response><|im_end|>\n<|im_start|>assistant\n",
) -> List[bool]:
    """
    Build a token-aligned trainable mask for the completion.
    Tokens inside <tool_response>...</tool_response> spans → False (not trained).
    """
    token_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    left_ids  = tokenizer(tool_resp_left,  add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(tool_resp_right, add_special_tokens=False)["input_ids"]

    if not left_ids or not right_ids:
        return [True] * len(token_ids)

    spans_not_trainable: List[tuple] = []
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


# =============================================================================
# Block noising + loss  (single-sequence, LLaDA style)
# =============================================================================

def llada_blk_forward_process(
    block: torch.Tensor,           # [1, block_length]
    trainable_mask: torch.Tensor,  # [1, block_length] bool
    mask_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Randomly mask k out of N trainable tokens in one block.
    Returns (noisy_block, num_masks_k, num_trainable_N).
    """
    device = block.device
    trainable_indices = torch.where(trainable_mask[0])[0]
    N = trainable_indices.shape[0]
    if N == 0:
        return block.clone(), 0, 0

    noisy_block = block.clone()

    # Pad trailing non-trainable positions with eos
    last_trainable = trainable_indices[-1].item()
    if last_trainable < block.shape[1] - 1:
        noisy_block[0, last_trainable + 1:] = eos_id

    k = torch.randint(1, N + 1, (), device=device).item()
    perm = torch.randperm(N, device=device)
    mask_indices = trainable_indices[perm[:k]]
    noisy_block[0, mask_indices] = mask_id

    return noisy_block, k, N


def llada_mdm_ce_loss(
    logits: torch.Tensor,        # [block_length, vocab_size]
    clean_block: torch.Tensor,   # [block_length]
    noisy_block: torch.Tensor,   # [block_length]
    trainable_mask: torch.Tensor,# [block_length] bool
    N: int,                      # number of trainable tokens
    k: int,                      # number of masked tokens
) -> torch.Tensor:
    """Masked-diffusion cross-entropy loss for one block, ELBO-weighted."""
    if k == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    masked_positions = (clean_block != noisy_block) & trainable_mask
    if masked_positions.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    ce = F.cross_entropy(logits[masked_positions], clean_block[masked_positions], reduction="sum")
    return ce / k * N   # ELBO weight: N/k


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorForLLaDAPreference(DataCollatorMixin):
    return_tensors: str = "pt"

    def torch_call(self, examples: list) -> dict:
        output = {
            "prompt_input_ids":       [torch.tensor(ex["prompt_input_ids"])       for ex in examples],
            "chosen_input_ids":       [torch.tensor(ex["chosen_input_ids"])       for ex in examples],
            "rejected_input_ids":     [torch.tensor(ex["rejected_input_ids"])     for ex in examples],
            "chosen_trainable_mask":  [torch.tensor(ex["chosen_trainable_mask"],  dtype=torch.bool) for ex in examples],
            "rejected_trainable_mask":[torch.tensor(ex["rejected_trainable_mask"],dtype=torch.bool) for ex in examples],
        }
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"]   = torch.tensor([ex["ref_chosen_logps"]   for ex in examples])
            output["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples])
        return output


# =============================================================================
# LLaDA DPO Trainer
# =============================================================================

class LLaDADPOTrainer(DPOTrainer):

    def __init__(
        self,
        model: Union[str, nn.Module],
        ref_model: Optional[nn.Module] = None,
        args: Optional[DPOConfig] = None,
        block_length: int = 64,        # match SFT block_size=64
        num_mc: int = 2,
        mask_token_id: Optional[int] = None,
        tool_resp_left:  str = "<|im_start|>user\n<tool_response>",
        tool_resp_right: str = "</tool_response><|im_end|>\n<|im_start|>assistant\n",
        **kwargs
    ):
        self.block_length    = block_length
        self.num_mc          = num_mc
        self.tool_resp_left  = tool_resp_left
        self.tool_resp_right = tool_resp_right

        # ZeRO-3 workaround: 不能用 None ref_model（TRL 会报错），
        # 也不能加载完整 8B ref_model（OOM）。
        # 用空 dummy + 临时将 prepare_deepspeed 替换为 no-op，
        # 使 TRL init 通过而不消耗 GPU 内存。
        # 训练中从 batch 读取外部预计算的 ref logps，dummy 永远不会被调用。
        if ref_model is None:
            import trl.models.utils as _tmu
            _orig_pds = _tmu.prepare_deepspeed

            def _noop_prepare(m, acc, **kw):
                return m  # 空 dummy 不需要 deepspeed 初始化

            _tmu.prepare_deepspeed = _noop_prepare
            # 也需要 patch trl.trainer.dpo_trainer 里的引用
            import trl.trainer.dpo_trainer as _tdpo
            _orig_pds2 = getattr(_tdpo, "prepare_deepspeed", None)
            _tdpo.prepare_deepspeed = _noop_prepare

            class _DummyRef(nn.Module):
                pass
            _dummy = _DummyRef()
            if hasattr(model, "config"):
                _dummy.config = model.config

            try:
                super().__init__(model=model, ref_model=_dummy, args=args, **kwargs)
            finally:
                _tmu.prepare_deepspeed = _orig_pds
                if _orig_pds2 is not None:
                    _tdpo.prepare_deepspeed = _orig_pds2
        else:
            super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)

        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif hasattr(self.processing_class, "mask_token_id") and self.processing_class.mask_token_id is not None:
            self.mask_token_id = self.processing_class.mask_token_id
        elif hasattr(self.model, "config") and hasattr(self.model.config, "mask_token_id"):
            self.mask_token_id = self.model.config.mask_token_id
        else:
            raise ValueError("Cannot find mask_token_id in tokenizer or model config")

        self.eos_token_id  = self.processing_class.eos_token_id
        self.data_collator = DataCollatorForLLaDAPreference()

        logger.info(
            f"LLaDA DPO: block_length={block_length}, num_mc={num_mc}, "
            f"mask_id={self.mask_token_id}"
        )

    # ── tokenize_row ──────────────────────────────────────────────────────────
    def tokenize_row(
        self,
        features: dict,
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int = None,
        max_completion_length: int = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> dict:
        tokenizer = processing_class

        prompt_input_ids   = tokenizer(features["prompt"],   add_special_tokens=False)["input_ids"]
        chosen_input_ids   = tokenizer(features["chosen"],   add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        chosen_input_ids   = chosen_input_ids   + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids   = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        chosen_trainable_mask = get_trainable_mask(
            features["chosen"], tokenizer, self.tool_resp_left, self.tool_resp_right
        )
        chosen_trainable_mask = (chosen_trainable_mask + [True])[:len(chosen_input_ids)]

        rejected_trainable_mask = get_trainable_mask(
            features["rejected"], tokenizer, self.tool_resp_left, self.tool_resp_right
        )
        rejected_trainable_mask = (rejected_trainable_mask + [True])[:len(rejected_input_ids)]

        return {
            "prompt_input_ids":        prompt_input_ids,
            "chosen_input_ids":        chosen_input_ids,
            "rejected_input_ids":      rejected_input_ids,
            "chosen_trainable_mask":   chosen_trainable_mask,
            "rejected_trainable_mask": rejected_trainable_mask,
        }

    # ── ELBO: single-sequence LLaDA style ─────────────────────────────────────
    def _get_elbo(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,        # [1, prompt_len + completion_len]
        prompt_length: int,
        completion_length: int,
        trainable_mask: torch.Tensor,  # [prompt_len + completion_len] bool
        seed: int,
    ) -> torch.Tensor:
        """
        Compute one Monte-Carlo ELBO sample for the completion.

        Steps:
          1. For each 64-token block in the completion, independently mask
             k ~ Uniform(1,N) trainable tokens.
          2. Run ONE forward pass on the resulting [prompt | noisy_completion].
             No attention_mask is passed — LLaDA's internal block-causal
             attention handles everything (same as in SFT training).
          3. Accumulate per-block ELBO = -CE * N/k.
        """
        set_seed(seed)
        device = full_ids.device
        block_length = self.block_length

        # Pad completion to a multiple of block_length
        pad_len = (block_length - completion_length % block_length) % block_length
        if pad_len > 0:
            full_ids = F.pad(full_ids, (0, pad_len), value=self.eos_token_id)
            extra    = torch.zeros(pad_len, dtype=torch.bool, device=device)
            trainable_mask = torch.cat([trainable_mask, extra])

        total_len        = full_ids.shape[1]
        num_comp_blocks  = (completion_length + block_length - 1) // block_length

        noisy_ids = full_ids.clone()
        block_ks  = []
        block_Ns  = []

        for bi in range(num_comp_blocks):
            bs = prompt_length + bi * block_length
            be = min(bs + block_length, total_len)

            block        = full_ids[:, bs:be]                   # [1, block_len]
            block_tm     = trainable_mask[bs:be].unsqueeze(0)   # [1, block_len]

            # Pad short last block
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

        # Single forward pass — no attention_mask (LLaDA internal block attention)
        use_bf16 = (os.environ.get("DEBUG_DPO_FP32", "0") != "1")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=noisy_ids).logits   # [1, total_len, V]

        elbo = torch.tensor(0.0, device=device, dtype=torch.float32)

        for bi in range(num_comp_blocks):
            if block_Ns[bi] == 0 or block_ks[bi] == 0:
                continue

            bs = prompt_length + bi * block_length
            be = min(bs + block_length, total_len)

            block_logits  = logits[0, bs:be]              # [block_len, V]
            clean_block   = full_ids[0, bs:be]             # [block_len]
            noisy_block   = noisy_ids[0, bs:be]            # [block_len]
            block_tm      = trainable_mask[bs:be]          # [block_len]

            loss_blk = llada_mdm_ce_loss(
                block_logits, clean_block, noisy_block, block_tm,
                N=block_Ns[bi], k=block_ks[bi]
            )
            elbo = elbo - loss_blk   # ELBO = -loss

        return elbo

    def _get_elbo_mc(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,        # [1, seq_len]
        prompt_length: int,
        completion_length: int,
        trainable_mask: torch.Tensor,  # [seq_len] bool
        seeds: torch.Tensor,           # [num_mc]
    ) -> torch.Tensor:
        elbos = [
            self._get_elbo(model, full_ids, prompt_length, completion_length,
                           trainable_mask, seeds[m].item())
            for m in range(self.num_mc)
        ]
        return torch.stack(elbos).mean()

    # ── concatenated_forward ──────────────────────────────────────────────────
    def _prepare_batch_seeds(self, batch: dict, device: torch.device) -> dict:
        n = len(batch["prompt_input_ids"])
        return {
            i: {
                "chosen_seeds":   torch.randint(0, 2**20, (self.num_mc,), device=device),
                "rejected_seeds": torch.randint(0, 2**20, (self.num_mc,), device=device),
            }
            for i in range(n)
        }

    def concatenated_forward(self, model: nn.Module, batch: dict, batch_seeds: dict) -> dict:
        device = next(model.parameters()).device
        chosen_lps, rejected_lps = [], []

        for i in range(len(batch["prompt_input_ids"])):
            prompt_ids       = batch["prompt_input_ids"][i].to(device)
            chosen_ids       = batch["chosen_input_ids"][i].to(device)
            rejected_ids     = batch["rejected_input_ids"][i].to(device)
            chosen_tm        = batch["chosen_trainable_mask"][i].to(device)
            rejected_tm      = batch["rejected_trainable_mask"][i].to(device)

            prompt_len   = len(prompt_ids)
            chosen_len   = len(chosen_ids)
            rejected_len = len(rejected_ids)

            # Prepend prompt-length zeros to trainable mask (prompt never trained)
            chosen_full_tm   = torch.cat([torch.zeros(prompt_len, dtype=torch.bool, device=device), chosen_tm])
            rejected_full_tm = torch.cat([torch.zeros(prompt_len, dtype=torch.bool, device=device), rejected_tm])

            chosen_full   = torch.cat([prompt_ids, chosen_ids]).unsqueeze(0)
            rejected_full = torch.cat([prompt_ids, rejected_ids]).unsqueeze(0)

            chosen_elbo   = self._get_elbo_mc(model, chosen_full,   prompt_len, chosen_len,   chosen_full_tm,   batch_seeds[i]["chosen_seeds"])
            rejected_elbo = self._get_elbo_mc(model, rejected_full, prompt_len, rejected_len, rejected_full_tm, batch_seeds[i]["rejected_seeds"])

            chosen_N   = max(chosen_tm.sum().item(),   1)
            rejected_N = max(rejected_tm.sum().item(), 1)

            chosen_lp   = chosen_elbo   / chosen_N
            rejected_lp = rejected_elbo / rejected_N

            if torch.isnan(chosen_elbo) or torch.isnan(rejected_elbo):
                rank = getattr(self.accelerator, "process_index", 0)
                logger.warning(
                    f"[NaN] concatenated_forward rank={rank} i={i} "
                    f"chosen_elbo={chosen_elbo.item()} rejected_elbo={rejected_elbo.item()} "
                    f"chosen_N={chosen_N} rejected_N={rejected_N} "
                    f"prompt_len={prompt_len} chosen_len={chosen_len} rejected_len={rejected_len}"
                )

            chosen_lps.append(chosen_lp)
            rejected_lps.append(rejected_lp)
            torch.cuda.empty_cache()

        chosen_logps   = torch.stack(chosen_lps)
        rejected_logps = torch.stack(rejected_lps)
        return {
            "chosen_logps":          chosen_logps,
            "rejected_logps":        rejected_logps,
            "mean_chosen_logits":    chosen_logps.mean(),
            "mean_rejected_logits":  rejected_logps.mean(),
        }

    def compute_ref_log_probs(self, batch: dict, batch_seeds: dict) -> tuple:
        ctx = autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), ctx:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_out = self.concatenated_forward(self.model, batch, batch_seeds)
            else:
                ref_out = self.concatenated_forward(self.ref_model, batch, batch_seeds)
        return ref_out["chosen_logps"], ref_out["rejected_logps"]

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict,
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple:
        metrics = {}
        device  = next(model.parameters()).device

        batch_seeds = self._prepare_batch_seeds(batch, device)
        model_out   = self.concatenated_forward(model, batch, batch_seeds)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps   = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch, batch_seeds)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_out["chosen_logps"], model_out["rejected_logps"],
            ref_chosen_logps, ref_rejected_logps,
        )

        prefix = "eval_" if train_eval == "eval" else ""
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"]    = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"]  = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"]= reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"]   = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"]      = model_out["chosen_logps"].mean().item()
        metrics[f"{prefix}logps/rejected"]    = model_out["rejected_logps"].mean().item()

        return losses.mean(), metrics

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "chosen_trainable_mask",
                "rejected_trainable_mask",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]
