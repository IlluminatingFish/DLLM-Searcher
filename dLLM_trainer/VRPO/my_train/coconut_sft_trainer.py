"""
COCONUT + SIM-CoT + LLaDA SFT Trainer
======================================

Sequence layout (Stage 1):
  [context (L_ctx)] [latent×N (H_plan injected)] [context_copy×L_ctx + action×L_act]
                                                  |<-- "doubling" block diffusion --|

Full sequence fed to model:
  [context | latent | context_copy | action]
                               ^-- noisy_action_block_j attends clean context_copy
                                   and clean action_blocks < j

Block diffusion within action:
  - Action divided into blocks of size `block_size` (default 64)
  - Each block's noisy version attends to [context | latent | clean_action_blocks_<j | noisy_block_j]
  - This uses the doubling trick: [context | latent | clean_action | noisy_action]

Attention constraints (boolean bias):
  - latent → clean_action    : BLOCKED
  - latent → noisy_action_j  : BLOCKED (for all j)
  - noisy_block_j → context  : allowed
  - noisy_block_j → latent   : allowed
  - noisy_block_j → clean_blocks_<j : allowed
  - noisy_block_j → noisy_block_<j  : BLOCKED (see clean copy instead)
  - noisy_block_j → clean/noisy_blocks_>j : BLOCKED

Loss = mdm CE on masked positions + λ * SIM-CoT aux CE on latent output hidden states

Stage 0 (warm-up): plan text prepended to context, standard LLaDA SFT (no latent)
"""

import os
import re
import glob
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

TOOL_RESP_LEFT  = "<tool_response>"
TOOL_RESP_RIGHT = "</tool_response>"


# =============================================================================
# Sentence split
# =============================================================================

def split_plan_sentences(plan: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', plan.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [plan.strip()]


# =============================================================================
# Block-causal attention bias (boolean, True=attend)
# =============================================================================

def build_coconut_attention_bias(
    L_ctx: int,
    N: int,
    L_act: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Boolean attention_bias for the doubled sequence:
      [context(L_ctx) | latent(N) | clean_action(L_act) | noisy_action(L_act)]

    True = attend, False = blocked (LLaDA converts False → -inf internally).

    Block diffusion rules for noisy_action:
      - noisy_block_j can attend to: context, latent, clean_action blocks < j, noisy_block_j itself
      - clean_action_block_j can attend to: context, latent, clean_action blocks ≤ j
      - latent rows cannot attend to any action columns (clean or noisy)
    """
    L_total = L_ctx + N + 2 * L_act
    bias = torch.zeros(1, 1, L_total, L_total, dtype=torch.bool, device=device)

    lat_s = L_ctx
    lat_e = L_ctx + N
    ca_s  = lat_e
    ca_e  = lat_e + L_act
    na_s  = ca_e

    n_blocks = (L_act + block_size - 1) // block_size

    # Context: bidirectional within ctx
    bias[:, :, :L_ctx, :L_ctx] = True

    # Latent: attend to context + latent only (NOT any action)
    bias[:, :, lat_s:lat_e, :lat_e] = True

    # Clean action block b: context + latent + clean blocks ≤ b
    for b in range(n_blocks):
        b_s = ca_s + b * block_size
        b_e = min(ca_s + (b + 1) * block_size, ca_e)
        bias[:, :, b_s:b_e, :b_e] = True

    # Noisy action block b: context + latent + clean blocks < b + noisy block b
    for b in range(n_blocks):
        nb_s = na_s + b * block_size
        nb_e = min(na_s + (b + 1) * block_size, L_total)
        clean_vis_end = ca_s + b * block_size  # clean blocks 0..b-1
        bias[:, :, nb_s:nb_e, :lat_e] = True
        if clean_vis_end > ca_s:
            bias[:, :, nb_s:nb_e, ca_s:clean_vis_end] = True
        bias[:, :, nb_s:nb_e, nb_s:nb_e] = True

    return bias


# =============================================================================
# Forward process: randomly mask one block in action
# =============================================================================

def block_forward_process(
    action_ids: torch.Tensor,      # [L_act]
    trainable_mask: torch.Tensor,  # [L_act] bool
    mask_token_id: int,
    eos_token_id: int,
    block_size: int,
    block_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Mask trainable positions in ONE randomly chosen block that has trainable tokens.
    Tries all blocks in random order until one with trainable positions is found.
    Returns (noisy_action, block_idx, n_masked, n_trainable_in_block).
    """
    L_act    = action_ids.shape[0]
    n_blocks = (L_act + block_size - 1) // block_size
    noisy    = action_ids.clone()

    if block_idx is None:
        order = torch.randperm(n_blocks).tolist()
    else:
        order = [block_idx]

    for blk in order:
        b_s   = blk * block_size
        b_e   = min((blk + 1) * block_size, L_act)
        tr_in = torch.where(trainable_mask[b_s:b_e])[0]
        n_tr  = tr_in.shape[0]
        if n_tr == 0:
            continue
        k    = torch.randint(1, n_tr + 1, ()).item()
        perm = torch.randperm(n_tr)
        noisy[b_s + tr_in[perm[:k]]] = mask_token_id
        return noisy, blk, k, n_tr

    return noisy, 0, 0, 0


# =============================================================================
# CE loss on the noisy block
# =============================================================================

def block_mdm_ce_loss(
    action_logits: torch.Tensor,  # [L_act, V]
    clean_ids: torch.Tensor,      # [L_act]
    noisy_ids: torch.Tensor,      # [L_act]
    n_trainable: int,
    n_masked: int,
) -> torch.Tensor:
    if n_masked == 0 or n_trainable == 0:
        return torch.tensor(0.0, device=action_logits.device, requires_grad=True)
    mask_pos = clean_ids != noisy_ids
    if mask_pos.sum() == 0:
        return torch.tensor(0.0, device=action_logits.device, requires_grad=True)
    loss = F.cross_entropy(
        action_logits[mask_pos], clean_ids[mask_pos], reduction="sum"
    )
    return loss / n_masked * n_trainable


# =============================================================================
# Pool teacher H_plan → [N, d]
# =============================================================================

def pool_plan_to_latent(
    H_plan: torch.Tensor,
    sent_ids: List[List[int]],
    N: int,
) -> torch.Tensor:
    d   = H_plan.shape[-1]
    K   = len(sent_ids)
    out = torch.zeros(N, d, device=H_plan.device, dtype=H_plan.dtype)
    lens   = [len(s) for s in sent_ids]
    starts = [sum(lens[:i]) for i in range(K)]
    ends   = [sum(lens[:i+1]) for i in range(K)]
    slot   = N // K
    for i in range(K):
        l_s = i * slot
        l_e = (i + 1) * slot if i < K - 1 else N
        p_s, p_e = starts[i], min(ends[i], H_plan.shape[0])
        if p_e > p_s:
            out[l_s:l_e] = H_plan[p_s:p_e].mean(dim=0, keepdim=True)
    return out


# =============================================================================
# Auxiliary Decoder (SIM-CoT)
# =============================================================================

class AuxDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# Dataset
# =============================================================================

class CoconutSFTDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        if max_samples:
            self.samples = self.samples[:max_samples]
        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class CoconutDataCollator:
    tokenizer:    Any
    mask_token_id: int
    max_ctx_len:  int  = 1024
    max_act_len:  int  = 1536
    stage:        int  = 1

    def _tok(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _build_context(self, messages: List[Dict], plan_str: str = "") -> List[int]:
        parts = []
        for msg in messages[:2]:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if plan_str and self.stage == 0:
            parts.append(f"Search plan: {plan_str}\n")
        parts.append("<|im_start|>assistant\n")
        return self._tok("".join(parts))[-self.max_ctx_len:]

    def _build_action(self, messages: List[Dict]) -> Tuple[List[int], List[bool]]:
        ids, trainable = [], []
        for msg in messages[2:]:
            role, content = msg["role"], msg["content"]
            header = f"<|im_start|>{role}\n"
            footer = "<|im_end|>\n"

            if role == "user" and TOOL_RESP_LEFT in content:
                pre, rest   = content.split(TOOL_RESP_LEFT, 1)
                inner, post = rest.split(TOOL_RESP_RIGHT, 1)
                for seg, tr in [
                    (header + pre + TOOL_RESP_LEFT, False),
                    (inner,                          False),
                    (TOOL_RESP_RIGHT + post + footer, False),
                ]:
                    t = self._tok(seg)
                    ids.extend(t); trainable.extend([tr] * len(t))
            else:
                t = self._tok(header + content + footer)
                ids.extend(t); trainable.extend([True] * len(t))

        return ids[:self.max_act_len], trainable[:self.max_act_len]

    def __call__(self, examples: List[Dict]) -> Dict[str, Any]:
        batch = []
        for ex in examples:
            msgs  = ex["messages"]
            plan  = ex["plan"]
            sents = split_plan_sentences(plan)
            pstr  = " ".join(sents)

            ctx_ids          = self._build_context(msgs, pstr if self.stage == 0 else "")
            act_ids, act_tr  = self._build_action(msgs)
            plan_ids         = self._tok(pstr)
            sent_ids         = [self._tok(s) for s in sents]

            batch.append(dict(
                ctx_ids=ctx_ids, act_ids=act_ids, act_tr=act_tr,
                plan_ids=plan_ids, sent_ids=sent_ids,
            ))

        pad_id  = self.tokenizer.eos_token_id or 0
        max_ctx = max(len(x["ctx_ids"]) for x in batch)
        max_act = max(len(x["act_ids"]) for x in batch)

        ctx_t = torch.full((len(batch), max_ctx), pad_id, dtype=torch.long)
        act_t = torch.full((len(batch), max_act), pad_id, dtype=torch.long)
        tr_t  = torch.zeros((len(batch), max_act), dtype=torch.bool)
        ctx_l = torch.zeros(len(batch), dtype=torch.long)
        act_l = torch.zeros(len(batch), dtype=torch.long)

        for i, b in enumerate(batch):
            c, a, at = b["ctx_ids"], b["act_ids"], b["act_tr"]
            ctx_t[i, :len(c)] = torch.tensor(c, dtype=torch.long)
            act_t[i, :len(a)] = torch.tensor(a, dtype=torch.long)
            tr_t[i, :len(at)] = torch.tensor(at, dtype=torch.bool)
            ctx_l[i] = len(c)
            act_l[i] = len(a)

        return {
            "context_ids":   ctx_t,
            "ctx_lengths":   ctx_l,
            "action_ids":    act_t,
            "act_lengths":   act_l,
            "act_trainable": tr_t,
            "plan_ids":      [b["plan_ids"]  for b in batch],
            "plan_sents":    [b["sent_ids"]  for b in batch],
        }


# =============================================================================
# COCONUT SFT Trainer
# =============================================================================

class CoconutSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        n_latent:        int   = 64,
        block_size:      int   = 64,
        aux_loss_weight: float = 0.1,
        stage:           int   = 1,
        mask_token_id:   int   = 126336,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_latent        = n_latent
        self.block_size      = block_size
        self.aux_loss_weight = aux_loss_weight
        self.stage           = stage
        self.mask_token_id   = mask_token_id

        # aux_decoder is registered on the model (set in main() before trainer init)
        # so ZeRO manages it; we just keep a reference here
        pass

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.model.module if hasattr(self.model, "module") else self.model
        return m.model.transformer.wte(ids)

    # ── Stage 0: explicit plan in context, standard LLaDA block diffusion ─

    def _stage0_attention_bias(self, L_ctx, L_act, block_size, device):
        """Bool attention_bias for [context | clean_action | noisy_action]."""
        L_total = L_ctx + 2 * L_act
        bias    = torch.zeros(1, 1, L_total, L_total, dtype=torch.bool, device=device)
        n_blks  = (L_act + block_size - 1) // block_size

        bias[:, :, :L_ctx, :L_ctx] = True
        for b in range(n_blks):
            b_s = L_ctx + b * block_size
            b_e = min(L_ctx + (b+1) * block_size, L_ctx + L_act)
            bias[:, :, b_s:b_e, :b_e] = True
        na_s = L_ctx + L_act
        for b in range(n_blks):
            nb_s = na_s + b * block_size
            nb_e = min(na_s + (b+1) * block_size, L_total)
            cv   = L_ctx + b * block_size
            bias[:, :, nb_s:nb_e, :L_ctx] = True
            if cv > L_ctx:
                bias[:, :, nb_s:nb_e, L_ctx:cv] = True
            bias[:, :, nb_s:nb_e, nb_s:nb_e] = True
        return bias

    def _loss_stage0(self, model, inputs):
        ctx_ids = inputs["context_ids"].to(model.device)
        act_ids = inputs["action_ids"].to(model.device)
        act_mask = inputs["act_trainable"].to(model.device)
        eos_id  = self.tokenizer.eos_token_id or 0
        total, valid = torch.tensor(0.0, device=model.device), 0

        for i in range(ctx_ids.shape[0]):
            L_ctx = inputs["ctx_lengths"][i].item()
            L_act = inputs["act_lengths"][i].item()
            c, a, at = ctx_ids[i, :L_ctx], act_ids[i, :L_act], act_mask[i, :L_act]

            noisy_a, _, n_masked, n_tr = block_forward_process(
                a, at, self.mask_token_id, eos_id, self.block_size
            )
            if n_masked == 0:
                continue

            full_ids  = torch.cat([c, a, noisy_a]).unsqueeze(0)
            attn_bias = self._stage0_attention_bias(L_ctx, L_act, self.block_size, model.device)

            out     = model(input_ids=full_ids, attention_bias=attn_bias)
            na_logits = out.logits[0, L_ctx + L_act:]

            total = total + block_mdm_ce_loss(na_logits, a, noisy_a, n_tr, n_masked)
            valid += 1

        return total / max(valid, 1)

    # ── Stage 1: COCONUT + SIM-CoT + LLaDA block diffusion ───────────────

    def _loss_stage1(self, model, inputs):
        ctx_ids  = inputs["context_ids"].to(model.device)
        act_ids  = inputs["action_ids"].to(model.device)
        act_mask = inputs["act_trainable"].to(model.device)
        eos_id   = self.tokenizer.eos_token_id or 0
        N, bs    = self.n_latent, self.block_size

        total_ce, total_aux, valid = (
            torch.tensor(0.0, device=model.device),
            torch.tensor(0.0, device=model.device),
            0,
        )

        for i in range(ctx_ids.shape[0]):
            L_ctx    = inputs["ctx_lengths"][i].item()
            L_act    = inputs["act_lengths"][i].item()
            sent_ids = inputs["plan_sents"][i]
            plan_ids = inputs["plan_ids"][i]

            c, a, at = ctx_ids[i, :L_ctx], act_ids[i, :L_act], act_mask[i, :L_act]

            # ── Teacher pass: context + plan → H_plan (no grad) ──────────
            plan_t = torch.tensor(plan_ids, device=model.device, dtype=torch.long)
            with torch.no_grad():
                t_out = model(
                    input_ids=torch.cat([c, plan_t]).unsqueeze(0),
                    output_hidden_states=True,
                )
            # Sync CUDA before Python-heavy ops to avoid GIL+CUDA deadlock with ZeRO-3
            torch.cuda.synchronize()
            H_plan   = t_out.hidden_states[-1][0, L_ctx:].detach()  # [L_plan, d]
            H_plan_n = pool_plan_to_latent(H_plan, sent_ids, N)     # [N, d]

            # ── Block forward process ─────────────────────────────────────
            noisy_a, _, n_masked, n_tr = block_forward_process(
                a, at, self.mask_token_id, eos_id, bs
            )

            # ── Student sequence: [ctx | latent_ph | clean_action | noisy_action] ──
            # Always build student sequence so all ranks do the same number of
            # model.forward() calls (ZeRO-3 requires symmetric collective ops).
            lat_ph   = torch.full((N,), self.mask_token_id, device=model.device, dtype=torch.long)
            noisy_for_student = noisy_a if n_masked > 0 else a  # dummy noisy if skipping
            full_ids = torch.cat([c, lat_ph, a, noisy_for_student]).unsqueeze(0)

            embeds = self._embed(full_ids).clone()
            embeds[0, L_ctx : L_ctx + N] = H_plan_n.to(embeds.dtype)

            attn_bias = build_coconut_attention_bias(L_ctx, N, L_act, bs, model.device)

            out = model(inputs_embeds=embeds, attention_bias=attn_bias,
                        output_hidden_states=True)

            if n_masked == 0:
                # Zero loss but backward still runs (maintains ZeRO-3 symmetry)
                total_ce  = total_ce  + 0.0 * out.logits.sum()
                total_aux = total_aux + 0.0 * out.hidden_states[-1].sum()
                continue

            # L_CE: CE on noisy_action logit slice (last L_act of the sequence)
            na_logits = out.logits[0, L_ctx + N + L_act:]         # [L_act, V]
            L_ce = block_mdm_ce_loss(na_logits, a, noisy_a, n_tr, n_masked)

            # L_aux: SIM-CoT on latent output hidden states
            H_lat     = out.hidden_states[-1][0, L_ctx : L_ctx + N]  # [N, d]
            L_aux     = self._aux_loss(H_lat, sent_ids, N, model.device)

            total_ce  = total_ce  + L_ce
            total_aux = total_aux + L_aux
            valid += 1

        if valid == 0:
            return total_ce + total_aux, total_ce.detach(), total_aux.detach()
        return (total_ce / valid + self.aux_loss_weight * (total_aux / valid),
                total_ce.detach() / valid,
                total_aux.detach() / valid)

    def _aux_decoder(self):
        """Return aux_decoder regardless of DDP/DeepSpeed wrapping."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        return m.aux_decoder_coconut

    def _aux_loss(self, H_lat, sent_ids, N, device):
        K    = len(sent_ids)
        loss = torch.tensor(0.0, device=device)
        slot = N // K
        aux_dec   = self._aux_decoder()
        proj_dtype = aux_dec.proj.weight.dtype
        for i, s_ids in enumerate(sent_ids):
            if not s_ids:
                continue
            l_s = i * slot
            l_e = (i + 1) * slot if i < K - 1 else N
            H_g    = H_lat[l_s:l_e].mean(dim=0).to(proj_dtype).unsqueeze(0)  # [1, d]
            logits = aux_dec(H_g).squeeze(0)                                 # [V]
            tgt    = torch.tensor(s_ids, dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(
                logits.unsqueeze(0).expand(len(tgt), -1), tgt, reduction="mean"
            )
        return loss / K

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.stage == 0:
            loss = self._loss_stage0(model, inputs)
        else:
            loss, l_ce, l_aux = self._loss_stage1(model, inputs)
            if not hasattr(self, '_ce_accum'):
                self._ce_accum, self._aux_accum = [], []
            self._ce_accum.append(l_ce.item())
            self._aux_accum.append(l_aux.item())
        return (loss, {}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs and hasattr(self, '_ce_accum') and self._ce_accum:
            logs['loss/ce']  = sum(self._ce_accum)  / len(self._ce_accum)
            logs['loss/aux'] = sum(self._aux_accum) / len(self._aux_accum)
            self._ce_accum, self._aux_accum = [], []
        super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        import threading
        import glob
        import shutil
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        torch.cuda.empty_cache()

        # ZeRO-2: all ranks hold full params. Only rank-0 saves.
        # The actual write is launched in a background thread so rank-0 can
        # reach the barrier immediately — preventing NCCL timeout when NFS is slow.
        # cpu_state is a detached CPU copy, so the thread is safe to write
        # it while training continues on the GPU params.
        if self.accelerator.is_main_process:
            # Wait for any still-running previous save before starting a new one.
            prev = getattr(self, '_save_thread', None)
            if prev is not None and prev.is_alive():
                prev.join()

            unwrapped = self.accelerator.unwrap_model(model)
            cpu_state = {k: v.detach().cpu() for k, v in unwrapped.named_parameters()}
            ckpt_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
            os.makedirs(ckpt_dir, exist_ok=True)

            # Capture everything the thread needs; avoid closing over `self`
            # for mutable attributes that may change.
            tokenizer     = self.tokenizer
            state_json    = os.path.join(ckpt_dir, "trainer_state.json")
            output_dir    = self.args.output_dir
            save_limit    = self.args.save_total_limit
            step          = self.state.global_step
            # trainer_state must be serialised now (it mutates each step)
            state_dict_snapshot = asdict(self.state)

            def _bg_save():
                # cpu_state drops out of scope (and is freed) when this
                # function returns; no explicit `del` needed. (A `del
                # cpu_state` here would make Python treat it as local to
                # this closure throughout, breaking the read above with
                # UnboundLocalError.)
                unwrapped.save_pretrained(ckpt_dir, state_dict=cpu_state, safe_serialization=True)
                tokenizer.save_pretrained(ckpt_dir)
                with open(state_json, "w") as f:
                    json.dump(state_dict_snapshot, f, indent=2)
                # Rotate checkpoints: keep only save_total_limit newest
                if save_limit and save_limit > 0:
                    existing = sorted(
                        glob.glob(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-*")),
                        key=lambda p: int(p.rsplit("-", 1)[-1]),
                    )
                    for old in existing[: max(0, len(existing) - save_limit)]:
                        shutil.rmtree(old, ignore_errors=True)

            self._save_thread = threading.Thread(
                target=_bg_save, name=f"ckpt-saver-{step}", daemon=True
            )
            self._save_thread.start()

        self.accelerator.wait_for_everyone()


# =============================================================================
# Config
# =============================================================================

@dataclass
class CoconutSFTConfig(TrainingArguments):
    dataset_path:    str           = field(default=None)
    max_samples:     Optional[int] = field(default=None)
    max_ctx_len:     int           = field(default=1024)
    max_act_len:     int           = field(default=1536)
    n_latent:        int           = field(default=64)
    block_size:      int           = field(default=64)
    aux_loss_weight: float         = field(default=0.1)
    stage:           int           = field(default=1)
    mask_token_id:   int           = field(default=126336)
    remove_unused_columns: bool    = field(default=False)
    dataloader_num_workers: int    = field(default=0)


# =============================================================================
# Main
# =============================================================================

def main():
    import sys
    from dataclasses import dataclass as _dc
    from transformers import HfArgumentParser

    # Force memory-efficient SDPA backend to avoid O(L²) math backend OOM with long sequences
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)   # flash_attn only handles no-mask/causal; skip it
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    @_dc
    class ModelArgs:
        model_name_or_path: str = field(default=None)
        dtype:              str = field(default="bfloat16")

    parser = HfArgumentParser((ModelArgs, CoconutSFTConfig))
    model_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=getattr(torch, model_args.dtype),
        trust_remote_code=True,
    )
    # Register aux_decoder as a model submodule so ZeRO-3 manages it correctly
    # Must match model dtype so ZeRO-3 doesn't see mixed dtypes
    _model_dtype = next(model.parameters()).dtype
    model.aux_decoder_coconut = AuxDecoder(
        model.config.d_model, model.config.vocab_size
    ).to(dtype=_model_dtype)

    # aux_decoder_coconut didn't exist yet when from_pretrained() ran above, so
    # if model_name_or_path is one of our checkpoints, its aux_decoder_coconut.*
    # weights were silently dropped (logged as "not used when initializing").
    # Pull them back in now that the submodule exists.
    _aux_shards = sorted(glob.glob(os.path.join(model_args.model_name_or_path, "*.safetensors")))
    if _aux_shards:
        _aux_state = {}
        for _shard in _aux_shards:
            with safe_open(_shard, framework="pt") as _f:
                for _k in _f.keys():
                    if _k.startswith("aux_decoder_coconut."):
                        _aux_state[_k[len("aux_decoder_coconut."):]] = _f.get_tensor(_k)
        if _aux_state:
            _missing, _unexpected = model.aux_decoder_coconut.load_state_dict(_aux_state, strict=False)
            logger.info(
                f"Restored aux_decoder_coconut from {model_args.model_name_or_path} "
                f"({len(_aux_state)} tensors; missing={_missing}, unexpected={_unexpected})"
            )

    # LLaDA doesn't support HF gradient_checkpointing_enable; patch it to use its own API
    def _llada_gc_enable(self, gradient_checkpointing_kwargs=None):
        self.model.set_activation_checkpointing("whole_layer")
    def _llada_gc_disable(self):
        self.model.set_activation_checkpointing(None)
    import types
    model.gradient_checkpointing_enable  = types.MethodType(_llada_gc_enable,  model)
    model.gradient_checkpointing_disable = types.MethodType(_llada_gc_disable, model)

    mask_token_id = training_args.mask_token_id
    if getattr(tokenizer, "mask_token_id", None) is not None:
        mask_token_id = tokenizer.mask_token_id

    dataset = CoconutSFTDataset(
        path=training_args.dataset_path,
        max_samples=training_args.max_samples,
    )
    collator = CoconutDataCollator(
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        max_ctx_len=training_args.max_ctx_len,
        max_act_len=training_args.max_act_len,
        stage=training_args.stage,
    )
    trainer = CoconutSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        n_latent=training_args.n_latent,
        block_size=training_args.block_size,
        aux_loss_weight=training_args.aux_loss_weight,
        stage=training_args.stage,
        mask_token_id=mask_token_id,
    )

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    if last_ckpt is not None:
        # Our _save_checkpoint only writes HF-format weights + trainer_state.json
        # (to avoid the DeepSpeed state_dict() OOM), not a DeepSpeed-native
        # checkpoint. transformers.Trainer unconditionally calls
        # deepspeed_load_checkpoint() when resume_from_checkpoint is set and
        # DeepSpeed is enabled, which fails looking for files we never wrote.
        # global_step/epoch restoration reads trainer_state.json separately
        # and doesn't depend on this call, so make it a no-op instead.
        import transformers.trainer as _hf_trainer_module
        _hf_trainer_module.deepspeed_load_checkpoint = lambda *a, **k: None

        # Same story for optimizer/scheduler restore: with DeepSpeed enabled,
        # _load_optimizer_and_scheduler unconditionally torch.load()s
        # scheduler.pt without checking it exists, which we never saved
        # (save_only_model=True). We can't just no-op this: the trainer's
        # "skip to global_step" catch-up loop only skips *data* batches
        # (skip_first_batches) and never calls lr_scheduler.step(), so the
        # scheduler's internal counter starts at 0 on every resume and only
        # advances via steps genuinely executed in *this* process. Left
        # alone that replays the warmup ramp from near-zero LR every resume
        # instead of continuing from the true global_step position — which
        # is exactly what corrupted the run we just killed. Fast-forward it
        # manually using the step count recorded in the checkpoint instead.
        def _fast_forward_scheduler(checkpoint):
            if checkpoint is None or trainer.lr_scheduler is None:
                return
            state_path = os.path.join(checkpoint, "trainer_state.json")
            if not os.path.isfile(state_path):
                return
            with open(state_path) as f:
                target_step = json.load(f)["global_step"]
            for _ in range(target_step):
                trainer.lr_scheduler.step()
            logger.info(f"Fast-forwarded lr_scheduler by {target_step} steps to match resumed global_step")

        trainer._load_optimizer_and_scheduler = _fast_forward_scheduler

    trainer.train(resume_from_checkpoint=last_ckpt)
    # No post-training save_model() call: our _save_checkpoint override already
    # saves each checkpoint in HF format. Calling trainer.save_model() here would
    # trigger DeepSpeed's state_dict() → 16GB GPU copy × 8 ranks → OOM → SSH kill.
    # The last checkpoint (incl. the forced end-of-training save) writes in a
    # daemon thread that the main process does NOT wait for by default —
    # without this join, the process can exit and kill that thread mid-write,
    # leaving a truncated final checkpoint (missing shards/index/tokenizer).
    _save_thread = getattr(trainer, "_save_thread", None)
    if _save_thread is not None:
        logger.info("Waiting for final checkpoint save to finish writing...")
        _save_thread.join()
    logger.info("Done.")


if __name__ == "__main__":
    main()
