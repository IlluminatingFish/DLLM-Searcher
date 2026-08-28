"""
COCONUT latent-only pretraining (SIM-CoT faithful)
====================================================
Trains ONLY the self-referential continuous-thought generation + auxiliary
reconstruction loss (Hao et al. arXiv:2412.06769 "Coconut" / arXiv:2509.20317
"SIM-CoT"). There is no CE / block-diffusion action-prediction task in this
file — see coconut_sft_trainer.py for the combined variant.

Per step k (k = 0..K-1, K = number of plan sentences for this example):

  z_k = last hidden state of a CAUSAL forward over [context, z_0, ..., z_{k-1}]

  z_k is BLIND: the explicit text of step k (or any later step) is never part
  of the sequence that produces it. This mirrors Coconut's official training
  loop (facebookresearch/coconut, `forward()`): each continuous thought is the
  hidden state at the position just before it, computed under a strictly
  causal mask, with the replaced explicit tokens absent from the sequence
  altogether — not merely masked-but-present.

Auxiliary loss (SIM-CoT Eq. 6): causal, teacher-forced reconstruction of step
k's own tokens, conditioned on z_k as an initial prefix vector. The decoder is
the base LLM itself ("architecturally identical to the LLM" per the SIM-CoT
paper) — there is no separate small decoder module. z_k itself receives no
direct loss; only the reconstructed-token cross-entropy backprops through it
into the shared model weights.

LLaDA defaults to bidirectional attention when no attention_bias is passed
(see modeling_llada.py: `is_causal=False` unless a bias is supplied), so both
the generation loop and the reconstruction pass below must pass an explicit
causal bias — this is not optional here as it would be for a naturally causal
LM.
"""

import os
import re
import json
import logging
import threading
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

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
# Sentence split (same segmentation as the combined trainer, reused only as
# reconstruction targets here — never as a generation input)
# =============================================================================

def split_plan_sentences(plan: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', plan.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [plan.strip()]


# =============================================================================
# Causal attention bias (boolean, True=attend)
# =============================================================================

def build_causal_bias(L: int, device: torch.device) -> torch.Tensor:
    """Standard lower-triangular causal mask. LLaDA is bidirectional by
    default (no bias => is_causal=False), so this must be passed explicitly
    for both the self-referential generation loop and the reconstruction
    decode below — without it position i could see positions > i, which
    would make the "generation" and "teacher-forced reconstruction" steps
    trivially leak the very thing they're supposed to produce/predict."""
    bias = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    return bias.unsqueeze(0).unsqueeze(0)


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
# Data Collator — batch size is always 1 here: K (number of latent steps)
# varies per example and each unrolls its own causal generation, so there is
# nothing to pad/stack across examples.
# =============================================================================

@dataclass
class LatentOnlyCollator:
    tokenizer:   Any
    max_ctx_len: int = 4096

    def _tok(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _build_context(self, messages: List[Dict]) -> List[int]:
        parts = []
        for msg in messages[:2]:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return self._tok("".join(parts))[-self.max_ctx_len:]

    def __call__(self, examples: List[Dict]) -> Dict[str, Any]:
        ex = examples[0]
        sents = split_plan_sentences(ex["plan"])
        return {
            "context_ids": torch.tensor(self._build_context(ex["messages"]), dtype=torch.long),
            "plan_sents":  [self._tok(s) for s in sents],
        }


# =============================================================================
# Trainer
# =============================================================================

class CoconutLatentOnlyTrainer(Trainer):
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.model.module if hasattr(self.model, "module") else self.model
        return m.model.transformer.wte(ids)

    def _generate_latents(self, model, context_ids, K):
        """Self-referential, causal, blind generation of K continuous
        thoughts. z_k never sees any explicit step text — only context and
        the z's it has generated so far."""
        device = context_ids.device
        embeds = self._embed(context_ids.unsqueeze(0))  # [1, L_ctx, d]
        z_list = []
        for _ in range(K):
            bias = build_causal_bias(embeds.shape[1], device)
            out = model(inputs_embeds=embeds, attention_bias=bias, output_hidden_states=True)
            z_k = out.hidden_states[-1][:, -1:, :]  # [1, 1, d] — last position only
            z_list.append(z_k)
            embeds = torch.cat([embeds, z_k], dim=1)
        return z_list

    def _reconstruct_loss(self, model, z_k, step_ids, device):
        """Teacher-forced causal reconstruction of one step's tokens from
        z_k (SIM-CoT Eq. 6). z_k is used purely as an initial prefix vector;
        it receives no direct loss of its own."""
        tgt = torch.tensor(step_ids, dtype=torch.long, device=device)
        if tgt.shape[0] == 0:
            return None
        if tgt.shape[0] > 1:
            tgt_embeds = self._embed(tgt[:-1].unsqueeze(0))       # [1, T-1, d]
            dec_in = torch.cat([z_k, tgt_embeds], dim=1)          # [1, T, d]
        else:
            dec_in = z_k                                          # [1, 1, d]
        bias = build_causal_bias(dec_in.shape[1], device)
        dec_out = model(inputs_embeds=dec_in, attention_bias=bias)
        return F.cross_entropy(dec_out.logits[0], tgt)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        context_ids = inputs["context_ids"].to(model.device)
        sent_ids    = inputs["plan_sents"]
        K = len(sent_ids)

        if K == 0:
            # Degenerate example: still run one forward so ZeRO's collective
            # ops stay symmetric across ranks.
            bias = build_causal_bias(context_ids.shape[0], model.device)
            dummy = model(inputs_embeds=self._embed(context_ids.unsqueeze(0)), attention_bias=bias)
            loss = 0.0 * dummy.logits.sum()
            return (loss, {}) if return_outputs else loss

        z_list = self._generate_latents(model, context_ids, K)

        total, valid = 0.0, 0
        for k in range(K):
            step_loss = self._reconstruct_loss(model, z_list[k], sent_ids[k], model.device)
            if step_loss is not None:
                total = total + step_loss
                valid += 1

        loss = total / max(valid, 1)

        if not hasattr(self, '_recon_accum'):
            self._recon_accum = []
        self._recon_accum.append(loss.detach().item())
        return (loss, {}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs and hasattr(self, '_recon_accum') and self._recon_accum:
            logs['loss/latent_recon'] = sum(self._recon_accum) / len(self._recon_accum)
            self._recon_accum = []
        super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        # See coconut_sft_trainer.py for the full rationale (background
        # thread to avoid NCCL timeout on slow NFS + the join() in main()
        # that guards against a truncated final checkpoint). No separate
        # aux_decoder module here — the "decoder" is the model itself, so
        # this just saves the model's own state_dict.
        torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
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
        torch.cuda.empty_cache()


# =============================================================================
# Config
# =============================================================================

@dataclass
class CoconutLatentOnlyConfig(TrainingArguments):
    dataset_path: str           = field(default=None)
    max_samples:  Optional[int] = field(default=None)
    max_ctx_len:  int           = field(default=4096)
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)


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

    parser = HfArgumentParser((ModelArgs, CoconutLatentOnlyConfig))
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

    dataset = CoconutSFTDataset(path=training_args.dataset_path, max_samples=training_args.max_samples)
    collator = LatentOnlyCollator(tokenizer=tokenizer, max_ctx_len=training_args.max_ctx_len)

    trainer = CoconutLatentOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    if last_ckpt is not None:
        # Same two patches as coconut_sft_trainer.py — see there for the
        # full rationale. deepspeed_load_checkpoint expects a DeepSpeed-
        # native checkpoint we never wrote; _load_optimizer_and_scheduler
        # would torch.load() a scheduler.pt we never wrote either, and
        # without fast-forwarding it manually the LR schedule restarts its
        # warmup from ~0 on every resume instead of continuing from the
        # true global_step position.
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

    # DeepSpeed's _has_inf_or_nan does x.float() on the gradient tensor, which
    # doubles its size (bf16 → fp32). For the embedding layer (~0.965 GiB bf16)
    # this allocates 1.93 GiB — exactly the amount that OOMs on GPU 7 after the
    # first checkpoint save. bf16 isnan/isinf work without a float32 cast.
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

        @staticmethod
        def _patched_has_inf_or_nan(x, j=None):
            return (~x.isfinite()).any().float()

        DeepSpeedZeroOptimizer._has_inf_or_nan = _patched_has_inf_or_nan
        logger.info("Patched DeepSpeed _has_inf_or_nan: skipping float32 cast")
    except Exception as e:
        logger.warning(f"Could not patch DeepSpeed _has_inf_or_nan: {e}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    _save_thread = getattr(trainer, "_save_thread", None)
    if _save_thread is not None:
        logger.info("Waiting for final checkpoint save to finish writing...")
        _save_thread.join()
    logger.info("Done.")


if __name__ == "__main__":
    main()
