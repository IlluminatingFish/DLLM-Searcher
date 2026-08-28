"""
COCONUT latent + CE combined training (SIM-CoT faithful generation)
=====================================================================
Combines the self-referential continuous-thought generation from
coconut_latent_only_trainer.py with the block-diffusion action CE task from
coconut_sft_trainer.py, so the two can be compared: does adding the CE task
on top of the same latent-generation mechanism help or hurt, and does the
latent-only checkpoint's z_k already contain what the CE task needs?

Per step k (k = 0..K-1, K = number of plan sentences for this example):

  z_k = last hidden state of a CAUSAL forward over [context, z_0, ..., z_{k-1}]

  z_k is BLIND — the explicit text of step k is never part of the sequence
  that produces it (see coconut_latent_only_trainer.py's docstring for the
  full rationale and paper references).

Two losses, both backpropagating through the SAME z_k's:

  1. L_aux: causal, teacher-forced reconstruction of step k's own tokens,
     conditioned on z_k as a prefix (SIM-CoT Eq. 6). Decoder = the base LLM
     itself.

  2. L_ce: the z_k's are injected as the "latent" segment of a student
     sequence [context | z_0..z_{K-1} | clean_action | noisy_action], using
     the same bidirectional block-diffusion attention and masking scheme as
     coconut_sft_trainer.py (one randomly chosen 64-token block of the action
     is masked; CE is computed only on that block). This phase uses LLaDA's
     default bidirectional attention, NOT causal — only the latent-generation
     and reconstruction phases above need the explicit causal override.

total_loss = L_ce + aux_loss_weight * L_aux

Gradients from BOTH losses flow back through the shared z_k generation, so
z_k is pushed to be simultaneously (a) decodable back into its source
sentence and (b) useful for predicting the masked action — unlike the
latent-only file, where only (a) is trained.
"""

import os
import re
import json
import logging
import threading
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

TOOL_RESP_LEFT  = "<tool_response>"
TOOL_RESP_RIGHT = "</tool_response>"


# =============================================================================
# Sentence split (segmentation for reconstruction targets only)
# =============================================================================

def split_plan_sentences(plan: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', plan.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [plan.strip()]


# =============================================================================
# Causal attention bias (boolean, True=attend) — used ONLY for the
# self-referential generation loop and the reconstruction decode.
# =============================================================================

def build_causal_bias(L: int, device: torch.device) -> torch.Tensor:
    """LLaDA is bidirectional by default (no bias => is_causal=False), so
    this must be passed explicitly wherever genuine causal/blind behavior is
    required — without it position i could see positions > i."""
    bias = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    return bias.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Block-diffusion attention bias (boolean, True=attend) — used for the
# student CE phase. Identical structure to coconut_sft_trainer.py's version,
# except N is now K (number of latent steps for this example) instead of a
# fixed 64.
# =============================================================================

def build_coconut_attention_bias(
    L_ctx: int,
    N: int,
    L_act: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Boolean attention_bias for [context(L_ctx) | latent(N) | clean_action(L_act) | noisy_action(L_act)].
    True = attend, False = blocked.
      - latent rows: context + latent only (no action, clean or noisy)
      - clean_action block b: context + latent + clean blocks <= b
      - noisy_action block b: context + latent + clean blocks < b + itself
    """
    L_total = L_ctx + N + 2 * L_act
    bias = torch.zeros(1, 1, L_total, L_total, dtype=torch.bool, device=device)

    lat_s = L_ctx
    lat_e = L_ctx + N
    ca_s  = lat_e
    ca_e  = lat_e + L_act
    na_s  = ca_e

    n_blocks = (L_act + block_size - 1) // block_size

    bias[:, :, :L_ctx, :L_ctx] = True
    bias[:, :, lat_s:lat_e, :lat_e] = True

    for b in range(n_blocks):
        b_s = ca_s + b * block_size
        b_e = min(ca_s + (b + 1) * block_size, ca_e)
        bias[:, :, b_s:b_e, :b_e] = True

    for b in range(n_blocks):
        nb_s = na_s + b * block_size
        nb_e = min(na_s + (b + 1) * block_size, L_total)
        clean_vis_end = ca_s + b * block_size
        bias[:, :, nb_s:nb_e, :lat_e] = True
        if clean_vis_end > ca_s:
            bias[:, :, nb_s:nb_e, ca_s:clean_vis_end] = True
        bias[:, :, nb_s:nb_e, nb_s:nb_e] = True

    return bias


# =============================================================================
# Forward process: randomly mask one block in action (unchanged from
# coconut_sft_trainer.py)
# =============================================================================

def block_forward_process(
    action_ids: torch.Tensor,
    trainable_mask: torch.Tensor,
    mask_token_id: int,
    eos_token_id: int,
    block_size: int,
) -> Tuple[torch.Tensor, int, int, int]:
    L_act    = action_ids.shape[0]
    n_blocks = (L_act + block_size - 1) // block_size
    noisy    = action_ids.clone()
    order    = torch.randperm(n_blocks).tolist()

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


def block_mdm_ce_loss(
    action_logits: torch.Tensor,
    clean_ids: torch.Tensor,
    noisy_ids: torch.Tensor,
    n_trainable: int,
    n_masked: int,
) -> torch.Tensor:
    if n_masked == 0 or n_trainable == 0:
        return torch.tensor(0.0, device=action_logits.device, requires_grad=True)
    mask_pos = clean_ids != noisy_ids
    if mask_pos.sum() == 0:
        return torch.tensor(0.0, device=action_logits.device, requires_grad=True)
    loss = F.cross_entropy(action_logits[mask_pos], clean_ids[mask_pos], reduction="sum")
    return loss / n_masked * n_trainable


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
# Data Collator — batch size is always 1: K varies per example and each
# unrolls its own causal generation, so there is nothing to pad/stack.
# =============================================================================

@dataclass
class LatentCECollator:
    tokenizer:    Any
    max_ctx_len:  int = 4096
    max_act_len:  int = 4096
    max_k:        int = 4  # cap on latent steps; real data: mean 2.4, p90 3, max 7 —
                            # only the small tail beyond p90 is affected

    def _tok(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _build_context(self, messages: List[Dict]) -> List[int]:
        parts = []
        for msg in messages[:2]:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
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
        ex = examples[0]
        sents = split_plan_sentences(ex["plan"])[:self.max_k]
        act_ids, act_tr = self._build_action(ex["messages"])
        return {
            "context_ids":   torch.tensor(self._build_context(ex["messages"]), dtype=torch.long),
            "action_ids":    torch.tensor(act_ids, dtype=torch.long),
            "act_trainable": torch.tensor(act_tr, dtype=torch.bool),
            "plan_sents":    [self._tok(s) for s in sents],
        }


# =============================================================================
# Trainer
# =============================================================================

class CoconutLatentCETrainer(Trainer):
    def __init__(self, *args, block_size: int = 64, aux_loss_weight: float = 0.1,
                 mask_token_id: int = 126336, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size      = block_size
        self.aux_loss_weight = aux_loss_weight
        self.mask_token_id   = mask_token_id

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.model.module if hasattr(self.model, "module") else self.model
        return m.model.transformer.wte(ids)

    def _generate_latents(self, model, context_ids, K):
        """Self-referential, causal, blind generation of K continuous
        thoughts — identical mechanism to coconut_latent_only_trainer.py."""
        device = context_ids.device
        embeds = self._embed(context_ids.unsqueeze(0))  # [1, L_ctx, d]
        z_list = []
        for _ in range(K):
            bias = build_causal_bias(embeds.shape[1], device)
            out = model(inputs_embeds=embeds, attention_bias=bias, output_hidden_states=True)
            z_k = out.hidden_states[-1][:, -1:, :]  # [1, 1, d]
            z_list.append(z_k)
            embeds = torch.cat([embeds, z_k], dim=1)
        return z_list

    def _reconstruct_loss_batched(self, model, z_list, sent_ids_list, device):
        """Causal teacher-forced reconstruction of ALL K steps' tokens in a
        SINGLE forward call, instead of K separate ones. The K steps are
        mutually independent (each depends only on its own z_k, not on any
        other step), so they can be stacked along the batch dimension rather
        than run sequentially — this halves the number of separate model()
        calls this trainer needs per example (2K+1 -> K+2), which matters
        under ZeRO-3 where every separate call pays its own fresh parameter
        all-gather cost regardless of how short its sequence is.

        z_k receives no direct loss of its own (SIM-CoT Eq. 6) — only the
        reconstructed-token cross-entropy backprops through it.
        """
        d = z_list[0].shape[-1]
        dec_ins, targets, lengths = [], [], []
        for k in range(len(z_list)):
            step_ids = sent_ids_list[k]
            if len(step_ids) == 0:
                continue
            tgt = torch.tensor(step_ids, dtype=torch.long, device=device)
            if tgt.shape[0] > 1:
                tgt_embeds = self._embed(tgt[:-1].unsqueeze(0))[0]  # [T-1, d]
                dec_in = torch.cat([z_list[k][0], tgt_embeds], dim=0)  # [T, d]
            else:
                dec_in = z_list[k][0]  # [1, d]
            dec_ins.append(dec_in)
            targets.append(tgt)
            lengths.append(dec_in.shape[0])

        if not dec_ins:
            return None

        Kb = len(dec_ins)
        T_max = max(lengths)
        batch_embeds = dec_ins[0].new_zeros(Kb, T_max, d)
        batch_targets = torch.full((Kb, T_max), -100, dtype=torch.long, device=device)
        for i, (de, tgt, L) in enumerate(zip(dec_ins, targets, lengths)):
            batch_embeds[i, :L] = de
            batch_targets[i, :L] = tgt

        # Right-padded + causal: real positions never attend into the
        # trailing padding regardless (causality already prevents that), so
        # the same lower-triangular mask is correct for every batch row —
        # only the loss needs the padding masked out (ignore_index=-100).
        bias = build_causal_bias(T_max, device)  # [1, 1, T_max, T_max], broadcasts over batch
        dec_out = model(inputs_embeds=batch_embeds, attention_bias=bias)
        return F.cross_entropy(
            dec_out.logits.reshape(-1, dec_out.logits.shape[-1]),
            batch_targets.reshape(-1),
            ignore_index=-100,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        context_ids   = inputs["context_ids"].to(model.device)
        action_ids    = inputs["action_ids"].to(model.device)
        act_trainable = inputs["act_trainable"].to(model.device)
        sent_ids      = inputs["plan_sents"]
        K = len(sent_ids)
        L_ctx = context_ids.shape[0]
        L_act = action_ids.shape[0]
        eos_id = self.tokenizer.eos_token_id or 0

        if K == 0:
            # Degenerate example: still run one forward so ZeRO's collective
            # ops stay symmetric across ranks.
            bias = build_causal_bias(L_ctx, model.device)
            dummy = model(inputs_embeds=self._embed(context_ids.unsqueeze(0)), attention_bias=bias)
            loss = 0.0 * dummy.logits.sum()
            return (loss, {}) if return_outputs else loss

        # ── Phase 1: self-referential blind latent generation ──────────────
        z_list = self._generate_latents(model, context_ids, K)

        # ── Aux loss: causal reconstruction, batched across all K steps ──
        L_aux = self._reconstruct_loss_batched(model, z_list, sent_ids, model.device)
        if L_aux is None:
            L_aux = 0.0 * z_list[0].sum()

        # ── Phase 2: inject z's into student sequence, block-diffusion CE ──
        noisy_action, _, n_masked, n_tr = block_forward_process(
            action_ids, act_trainable, self.mask_token_id, eos_id, self.block_size
        )
        noisy_for_student = noisy_action if n_masked > 0 else action_ids

        lat_ph = torch.full((K,), self.mask_token_id, device=model.device, dtype=torch.long)
        full_ids = torch.cat([context_ids, lat_ph, action_ids, noisy_for_student]).unsqueeze(0)
        embeds = self._embed(full_ids).clone()
        Z = torch.cat(z_list, dim=1).squeeze(0)  # [K, d] — gradients flow through to phase 1
        embeds[0, L_ctx:L_ctx + K] = Z.to(embeds.dtype)

        attn_bias = build_coconut_attention_bias(L_ctx, K, L_act, self.block_size, model.device)
        out = model(inputs_embeds=embeds, attention_bias=attn_bias)

        if n_masked == 0:
            L_ce = 0.0 * out.logits.sum()
        else:
            na_logits = out.logits[0, L_ctx + K + L_act:]
            L_ce = block_mdm_ce_loss(na_logits, action_ids, noisy_action, n_tr, n_masked)

        loss = L_ce + self.aux_loss_weight * L_aux

        if not hasattr(self, '_ce_accum'):
            self._ce_accum, self._aux_accum = [], []
        self._ce_accum.append(L_ce.detach().item())
        self._aux_accum.append(L_aux.detach().item() if torch.is_tensor(L_aux) else float(L_aux))

        return (loss, {}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs and hasattr(self, '_ce_accum') and self._ce_accum:
            logs['loss/ce']  = sum(self._ce_accum)  / len(self._ce_accum)
            logs['loss/aux'] = sum(self._aux_accum) / len(self._aux_accum)
            self._ce_accum, self._aux_accum = [], []
        super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        # See coconut_sft_trainer.py for the full rationale (background
        # thread to dodge NCCL timeouts on slow NFS + the join() in main()
        # guarding against a truncated final checkpoint). No separate aux
        # decoder module here — the "decoder" is the model itself.
        #
        # Under ZeRO-3 (unlike ZeRO-2), each rank only holds a SHARD of the
        # parameters — unwrapped.named_parameters() on rank 0 alone would
        # silently return a partial/wrong model. accelerator.get_state_dict()
        # is the ZeRO-stage-agnostic way to gather the full state dict; it is
        # a collective op that every rank must call together, so it cannot be
        # deferred into the rank-0-only background thread the way the actual
        # disk write is.
        torch.cuda.empty_cache()
        full_state_dict = self.accelerator.get_state_dict(model)

        if self.accelerator.is_main_process:
            prev = getattr(self, '_save_thread', None)
            if prev is not None and prev.is_alive():
                prev.join()

            unwrapped = self.accelerator.unwrap_model(model)
            cpu_state = {k: v.detach().cpu() for k, v in full_state_dict.items()}
            ckpt_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
            os.makedirs(ckpt_dir, exist_ok=True)

            tokenizer  = self.tokenizer
            state_json = os.path.join(ckpt_dir, "trainer_state.json")
            output_dir = self.args.output_dir
            save_limit = self.args.save_total_limit
            step       = self.state.global_step
            state_dict_snapshot = asdict(self.state)

            def _bg_save():
                unwrapped.save_pretrained(ckpt_dir, state_dict=cpu_state, safe_serialization=True)
                tokenizer.save_pretrained(ckpt_dir)
                with open(state_json, "w") as f:
                    json.dump(state_dict_snapshot, f, indent=2)
                if save_limit and save_limit > 0:
                    import glob
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
class CoconutLatentCEConfig(TrainingArguments):
    dataset_path:    str           = field(default=None)
    max_samples:     Optional[int] = field(default=None)
    max_ctx_len:     int           = field(default=4096)
    max_act_len:     int           = field(default=4096)
    block_size:      int           = field(default=64)
    aux_loss_weight: float         = field(default=0.1)
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

    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    @_dc
    class ModelArgs:
        model_name_or_path: str = field(default=None)
        dtype:              str = field(default="bfloat16")

    parser = HfArgumentParser((ModelArgs, CoconutLatentCEConfig))
    model_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)

    if training_args.per_device_train_batch_size != 1:
        logger.warning(
            "per_device_train_batch_size must be 1 for this trainer (K varies "
            "per example, each unrolls its own causal generation); overriding."
        )
        training_args.per_device_train_batch_size = 1

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=getattr(torch, model_args.dtype),
        trust_remote_code=True,
    )

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

    dataset = CoconutSFTDataset(path=training_args.dataset_path, max_samples=training_args.max_samples)
    collator = LatentCECollator(
        tokenizer=tokenizer,
        max_ctx_len=training_args.max_ctx_len,
        max_act_len=training_args.max_act_len,
    )

    trainer = CoconutLatentCETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        block_size=training_args.block_size,
        aux_loss_weight=training_args.aux_loss_weight,
        mask_token_id=mask_token_id,
    )

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    if last_ckpt is not None:
        # Same two patches as coconut_sft_trainer.py / coconut_latent_only_trainer.py.
        import transformers.trainer as _hf_trainer_module
        _hf_trainer_module.deepspeed_load_checkpoint = lambda *a, **k: None

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

    _save_thread = getattr(trainer, "_save_thread", None)
    if _save_thread is not None:
        logger.info("Waiting for final checkpoint save to finish writing...")
        _save_thread.join()
    logger.info("Done.")


if __name__ == "__main__":
    main()
