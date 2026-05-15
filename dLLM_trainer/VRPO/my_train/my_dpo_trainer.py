

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
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
# Trainable Mask (aligned with tokenize_row tokenization)
# =============================================================================

def get_trainable_mask(
    completion_text: str,
    tokenizer: PreTrainedTokenizerBase,
    tool_resp_left: str = "<|im_start|>user\n<tool_response>",
    tool_resp_right: str = "</tool_response><|im_end|>\n<|im_start|>assistant\n",
) -> List[bool]:
    """
    Build trainable mask aligned 1:1 with tokenizer(completion_text).
    Tokens inside <tool_response>...</tool_response> are non-trainable (False).
    Uses token-space search: same tokenizer(completion) as tokenize_row, no offset_mapping/decode.
    """
    token_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    left_ids = tokenizer(tool_resp_left, add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(tool_resp_right, add_special_tokens=False)["input_ids"]

    if not left_ids or not right_ids:
        return [True] * len(token_ids)

    # Find [start, end) token spans of tool_response regions
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
# Block Diffusion 
# =============================================================================

def make_basic_block_attention(N: int, start_pos: int, block_size: int) -> torch.Tensor:
    """Create block attention mask for block diffusion."""
    L0 = start_pos
    L1 = (N - L0) // 2
    assert L0 + 2 * L1 == N, f"input length must be L0 + 2*L1, got N={N}, L0={L0}, L1={L1}"

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


def blk_forward_process_with_trainable(
    batch: torch.Tensor,           # [batch_size, block_length]
    trainable_mask: torch.Tensor,  # [batch_size, block_length] bool
    mask_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def mdm_ce_loss_with_trainable(
    logits: torch.Tensor,           # [batch_size, block_length, vocab_size]
    clean_batch: torch.Tensor,      # [batch_size, block_length]
    noisy_batch: torch.Tensor,      # [batch_size, block_length]
    trainable_mask: torch.Tensor,   # [batch_size, block_length] bool
    block_lengths: torch.Tensor,    # [batch_size] - trainable tokens per block
    num_trainable_masks: torch.Tensor,  # [batch_size]
) -> torch.Tensor:
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

        if torch.isnan(ce_loss) or torch.isnan(loss[i]):
            targets = clean_batch[i][valid_positions]
            vocab_size = logits.shape[-1]
            logger.warning(
                f"[NaN] mdm_ce_loss: i={i} ce_loss={ce_loss.item()} loss_i={loss[i].item()} "
                f"valid_cnt={valid_positions.sum().item()} num_masks={num_trainable_masks[i].item()} "
                f"logits_nan={torch.isnan(logits[i]).any().item()} logits_inf={torch.isinf(logits[i]).any().item()} "
                f"targets_min={targets.min().item()} targets_max={targets.max().item()} vocab_size={vocab_size}"
            )

    return loss

# =============================================================================
# Data Collator with Trainable Mask
# =============================================================================

@dataclass
class DataCollatorForSDARPreference(DataCollatorMixin):
    return_tensors: str = "pt"

    def torch_call(self, examples: list) -> dict:
        output = {
            "prompt_input_ids": [torch.tensor(ex["prompt_input_ids"]) for ex in examples],
            "chosen_input_ids": [torch.tensor(ex["chosen_input_ids"]) for ex in examples],
            "rejected_input_ids": [torch.tensor(ex["rejected_input_ids"]) for ex in examples],
            "chosen_trainable_mask": [torch.tensor(ex["chosen_trainable_mask"], dtype=torch.bool) for ex in examples],
            "rejected_trainable_mask": [torch.tensor(ex["rejected_trainable_mask"], dtype=torch.bool) for ex in examples],
        }
        
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([ex["ref_chosen_logps"] for ex in examples])
            output["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples])

        return output


# =============================================================================
# SDAR DPO Trainer
# =============================================================================

class SDARDPOTrainer(DPOTrainer):
    
    def __init__(
        self,
        model: Union[str, nn.Module],
        ref_model: Optional[nn.Module] = None,
        args: Optional[DPOConfig] = None,
        block_length: int = 128,
        num_mc: int = 2,
        mask_token_id: Optional[int] = None,
        tool_resp_left: str = "<|im_start|>user\n<tool_response>",
        tool_resp_right: str = "</tool_response><|im_end|>\n<|im_start|>assistant\n",
        **kwargs
    ):

        self.block_length = block_length
        self.num_mc = num_mc
        self.tool_resp_left = tool_resp_left
        self.tool_resp_right = tool_resp_right

        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)
        
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif hasattr(self.processing_class, 'mask_token_id') and self.processing_class.mask_token_id is not None:
            self.mask_token_id = self.processing_class.mask_token_id
        else:
            self.mask_token_id = 128002  
        
        self.eos_token_id = self.processing_class.eos_token_id
        
        self.data_collator = DataCollatorForSDARPreference()
        
        logger.info(f"SDAR DPO: block_length={block_length}, num_mc={num_mc}, mask_id={self.mask_token_id}")

    def tokenize_row(
        self,
        features: dict,
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int = None,
        max_completion_length: int = None,
        add_special_tokens: bool = False,
        **kwargs,  # accept is_chat, etc. from TRL DPOTrainer
    ) -> dict:
        tokenizer = processing_class
        
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]
        
        # Add EOS
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]
        
        # Truncate
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]
        

        # Build mask aligned with tokenizer(completion) + [eos], then truncate to match input_ids
        chosen_trainable_mask = get_trainable_mask(
            features["chosen"], tokenizer,
            self.tool_resp_left, self.tool_resp_right
        )
        chosen_trainable_mask = chosen_trainable_mask + [True]  # EOS token is trainable
        chosen_trainable_mask = chosen_trainable_mask[:len(chosen_input_ids)]

        rejected_trainable_mask = get_trainable_mask(
            features["rejected"], tokenizer,
            self.tool_resp_left, self.tool_resp_right
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

    def _get_elbo_blk_with_trainable(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,           # [num_mc, seq_len]
        prompt_length: int,
        completion_length: int,
        trainable_mask: torch.Tensor,      # [seq_len] bool
        mask_seeds: torch.Tensor,          # [num_mc]
    ) -> torch.Tensor:
        block_length = self.block_length
        device = input_ids.device
        batch_size = input_ids.shape[0]  # = num_mc
        
        # Pad to block boundary
        batch = F.pad(input_ids, (0, block_length - 1), value=self.eos_token_id)
        trainable_mask_padded = F.pad(trainable_mask, (0, block_length - 1), value=False)
        
        num_blocks = (prompt_length + completion_length + block_length - 1) // block_length
        prompt_completion_len_with_pad = num_blocks * block_length
        
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        completion_len_with_pad = prompt_completion_len_with_pad - prefill_length
        num_completion_blocks = num_blocks - prefill_blocks
        
        blk_num_masks = torch.zeros((num_completion_blocks, batch_size), dtype=torch.long, device=device)
        blk_trainable_lengths = torch.zeros((num_completion_blocks, batch_size), dtype=torch.long, device=device)
        
        noisy_batch = batch.clone()
        
        for i in range(batch_size):
            set_seed(mask_seeds[i].item())
            
            for j in range(num_completion_blocks):
                if j == 0:
                    block_start = prompt_length
                    block_end = (prefill_blocks + 1) * block_length
                else:
                    block_start = prefill_length + j * block_length
                    block_end = block_start + block_length
                
                block_content = batch[i:i+1, block_start:block_end]
                block_trainable = trainable_mask_padded[block_start:block_end].unsqueeze(0)
                
                noisy_block, num_masks, _ = blk_forward_process_with_trainable(
                    block_content,
                    block_trainable,
                    self.mask_token_id,
                    self.eos_token_id,
                )
                
                noisy_batch[i:i+1, block_start:block_end] = noisy_block
                blk_num_masks[j, i] = num_masks[0]
                blk_trainable_lengths[j, i] = block_trainable.sum()
        
        batch_concat = torch.cat(
            (batch[:, :prompt_completion_len_with_pad], 
             noisy_batch[:, prefill_length:prompt_completion_len_with_pad]),
            dim=1
        )
        
        # Block attention mask
        attn_mask = make_basic_block_attention(
            N=prompt_completion_len_with_pad + completion_len_with_pad,
            start_pos=prefill_length,
            block_size=block_length,
        ).to(dtype=torch.bool, device=device)

        # Debug: 若需排查 NaN，可设 DEBUG_DPO_FP32=1 禁用 bf16
        use_bf16 = (os.environ.get("DEBUG_DPO_FP32", "0") != "1")

        # Forward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=batch_concat, attention_mask=attn_mask).logits

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            m = attn_mask[0, 0] if attn_mask.dim() == 4 else attn_mask
            row_sums = m.sum(dim=-1) if m.dtype == torch.bool else (m > 0).sum(dim=-1)
            n_all_false = (row_sums == 0).sum().item()
            logger.warning(
                f"[NaN] _get_elbo_blk: model logits have nan/inf "
                f"logits_nan={torch.isnan(logits).any().item()} logits_inf={torch.isinf(logits).any().item()} "
                f"shape={logits.shape} attn_mask_all_false_rows={n_all_false} "
                f"N={attn_mask.shape[2] if attn_mask.dim()==4 else '?'}"
            )

        loss_expand = torch.zeros((batch_size, num_completion_blocks), device=device, dtype=torch.float32)
        
        block_boundaries = []
        for j in range(num_completion_blocks):
            if j == 0:
                block_boundaries.append((prompt_length, (prefill_blocks + 1) * block_length))
            else:
                start = prefill_length + j * block_length
                block_boundaries.append((start, start + block_length))
        
        for idx in range(num_completion_blocks):
            block_start, block_end = block_boundaries[idx]
            actual_block_len = block_end - block_start
            
            if blk_num_masks[idx].sum() == 0:
                continue
            
            logits_offset = block_start - prefill_length
            logits_start = prompt_completion_len_with_pad + logits_offset
            logits_end = logits_start + actual_block_len
            
            block_logits = logits[:, logits_start:logits_end, :]
            if block_logits.shape[1] < block_length:
                pad_size = block_length - block_logits.shape[1]
                block_logits = F.pad(block_logits, (0, 0, 0, pad_size), value=0)
            
            clean_block = batch_concat[:, block_start:block_end]
            if clean_block.shape[1] < block_length:
                clean_block = F.pad(clean_block, (0, block_length - clean_block.shape[1]), value=self.eos_token_id)
            
            noisy_block = batch_concat[:, logits_start:logits_end]
            if noisy_block.shape[1] < block_length:
                noisy_block = F.pad(noisy_block, (0, block_length - noisy_block.shape[1]), value=self.eos_token_id)
            
            block_trainable = trainable_mask_padded[block_start:block_end]
            if len(block_trainable) < block_length:
                block_trainable = F.pad(block_trainable, (0, block_length - len(block_trainable)), value=False)
            block_trainable = block_trainable.unsqueeze(0).expand(batch_size, -1)
            
            block_lengths = blk_trainable_lengths[idx].float().clamp(min=1)
            
            local_loss = mdm_ce_loss_with_trainable(
                block_logits,
                clean_block,
                noisy_block,
                block_trainable,
                block_lengths,
                blk_num_masks[idx],
            )

            if torch.isnan(local_loss).any() or torch.isinf(local_loss).any():
                logger.warning(
                    f"[NaN] _get_elbo_blk: block_idx={idx} local_loss={local_loss.tolist()} "
                    f"blk_num_masks={blk_num_masks[idx].tolist()} block_trainable_sum={block_trainable.sum().item()}"
                )

            loss_expand[:, idx] = -local_loss  # ELBO = -loss

        loss = loss_expand.sum(dim=-1)  # [batch_size] = [num_mc]
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning(
                f"[NaN] _get_elbo_blk: final loss={loss.tolist()} loss_expand_nan={torch.isnan(loss_expand).any().item()}"
            )
        return loss.unsqueeze(0)  # [1, num_mc]

    def _get_elbo_mc(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,      # [1, seq_len]
        prompt_length: int,
        completion_length: int,
        trainable_mask: torch.Tensor, # [seq_len] bool
        mask_seeds: torch.Tensor,     
    ) -> torch.Tensor:
        input_ids_expanded = input_ids.expand(self.num_mc, -1).clone()
        
        elbo = self._get_elbo_blk_with_trainable(
            model,
            input_ids_expanded,
            prompt_length,
            completion_length,
            trainable_mask,
            mask_seeds,
        )  # [1, num_mc]
        
        return elbo.mean()  # scalar

    def _prepare_batch_seeds(self, batch: dict, device: torch.device) -> dict:

        prompt_input_ids = batch["prompt_input_ids"]
        if isinstance(prompt_input_ids, list):
            num_examples = len(prompt_input_ids)
        else:
            num_examples = prompt_input_ids.shape[0]
            
        batch_seeds = {}
        
        for i in range(num_examples):
            batch_seeds[i] = {
                "chosen_seeds": torch.randint(0, 2**20, (self.num_mc,), device=device),
                "rejected_seeds": torch.randint(0, 2**20, (self.num_mc,), device=device),
            }
        
        return batch_seeds

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict,
        batch_seeds: dict,
    ) -> dict:
        num_examples = len(batch["prompt_input_ids"])
        device = next(model.parameters()).device
        
        chosen_logps_list = []
        rejected_logps_list = []
        
        for i in range(num_examples):
            prompt_ids = batch["prompt_input_ids"][i].to(device)
            chosen_ids = batch["chosen_input_ids"][i].to(device)
            rejected_ids = batch["rejected_input_ids"][i].to(device)
            chosen_trainable = batch["chosen_trainable_mask"][i].to(device)
            rejected_trainable = batch["rejected_trainable_mask"][i].to(device)
            
            prompt_len = len(prompt_ids)
            chosen_len = len(chosen_ids)
            rejected_len = len(rejected_ids)
            
            chosen_full = torch.cat([prompt_ids, chosen_ids]).unsqueeze(0)
            rejected_full = torch.cat([prompt_ids, rejected_ids]).unsqueeze(0)
            
            chosen_full_trainable = torch.cat([
                torch.zeros(prompt_len, dtype=torch.bool, device=device),
                chosen_trainable
            ])
            rejected_full_trainable = torch.cat([
                torch.zeros(prompt_len, dtype=torch.bool, device=device),
                rejected_trainable
            ])
            
            chosen_seeds = batch_seeds[i]["chosen_seeds"]
            rejected_seeds = batch_seeds[i]["rejected_seeds"]
            
            chosen_elbo = self._get_elbo_mc(
                model, chosen_full, prompt_len, chosen_len, chosen_full_trainable, chosen_seeds
            )
            rejected_elbo = self._get_elbo_mc(
                model, rejected_full, prompt_len, rejected_len, rejected_full_trainable, rejected_seeds
            )
            
            chosen_trainable_len = max(chosen_trainable.sum().item(), 1)
            rejected_trainable_len = max(rejected_trainable.sum().item(), 1)

            chosen_lp = chosen_elbo / chosen_trainable_len
            rejected_lp = rejected_elbo / rejected_trainable_len

            if torch.isnan(chosen_elbo) or torch.isnan(rejected_elbo):
                rank = getattr(self.accelerator, "process_index", 0)
                logger.warning(
                    f"[NaN] concatenated_forward: rank={rank} example={i} "
                    f"chosen_elbo={chosen_elbo.item()} rejected_elbo={rejected_elbo.item()} "
                    f"chosen_trainable_sum={chosen_trainable.sum().item()} rejected_trainable_sum={rejected_trainable.sum().item()} "
                    f"prompt_len={prompt_len} chosen_len={chosen_len} rejected_len={rejected_len}"
                )

            chosen_logps_list.append(chosen_lp)
            rejected_logps_list.append(rejected_lp)

            torch.cuda.empty_cache()
        
        chosen_logps = torch.stack(chosen_logps_list)
        rejected_logps = torch.stack(rejected_logps_list)
        
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logps.mean(),
            "mean_rejected_logits": rejected_logps.mean(),
        }

    def compute_ref_log_probs(self, batch: dict, batch_seeds: dict) -> tuple:
        ctx = autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        
        with torch.no_grad(), ctx:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_output = self.concatenated_forward(self.model, batch, batch_seeds)
            else:
                ref_output = self.concatenated_forward(self.ref_model, batch, batch_seeds)
        
        return ref_output["chosen_logps"], ref_output["rejected_logps"]

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict,
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple:
        metrics = {}
        # device = batch["prompt_input_ids"].device
        device = next(model.parameters()).device
        
        batch_seeds = self._prepare_batch_seeds(batch, device)
        
        # Policy model ELBO
        model_output = self.concatenated_forward(model, batch, batch_seeds)

        if torch.isnan(model_output["chosen_logps"]).any() or torch.isnan(model_output["rejected_logps"]).any():
            rank = getattr(self.accelerator, "process_index", 0)
            logger.warning(
                f"[NaN] get_batch_loss_metrics: policy output has NaN rank={rank} "
                f"chosen_logps={model_output['chosen_logps'].tolist()} rejected_logps={model_output['rejected_logps'].tolist()}"
            )

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch, batch_seeds)

        if torch.isnan(ref_chosen_logps).any() or torch.isnan(ref_rejected_logps).any():
            rank = getattr(self.accelerator, "process_index", 0)
            logger.warning(
                f"[NaN] get_batch_loss_metrics: ref output has NaN rank={rank} "
                f"ref_chosen={ref_chosen_logps.tolist()} ref_rejected={ref_rejected_logps.tolist()}"
            )

        # DPO loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )

        if torch.isnan(losses).any() or torch.isnan(chosen_rewards).any() or torch.isnan(rejected_rewards).any():
            rank = getattr(self.accelerator, "process_index", 0)
            logger.warning(
                f"[NaN] get_batch_loss_metrics: after DPO loss rank={rank} "
                f"losses={losses.tolist()} chosen_rewards={chosen_rewards.tolist()} rejected_rewards={rejected_rewards.tolist()}"
            )

        # Metrics
        prefix = "eval_" if train_eval == "eval" else ""
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].mean().item()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].mean().item()
        
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