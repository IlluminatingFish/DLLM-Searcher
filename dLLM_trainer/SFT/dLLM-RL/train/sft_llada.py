import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import Dataset, DataLoader
from train.utils import get_config, flatten_omega_conf, AverageMeter

logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
        )


def main():
    config = get_config()

    project_name = config.experiment.project
    base_model_path = config.model.pretrained_model
    pretrained_model = config.model.pretrained_model
    if config.experiment.get("resume_from_checkpoint"):
        pretrained_model = config.experiment.resume_from_checkpoint
        print(f"Resuming from checkpoint: {pretrained_model}")

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            project=config.experiment.project,
            entity=config.wandb.get("entity", None),
            name="sft_llada_64",
            notes=f"Pretrained model: {pretrained_model}",
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)
        try:
            wandb.init(**wandb_init_kwargs, config=wandb_config)
        except Exception as e:
            logger.warning(f"wandb.init failed ({e}), continuing without wandb logging")
            wandb.init(mode="disabled")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.project) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ── Model & Tokenizer ─────────────────────────────────────────────────────
    logger.info("Loading LLaDA model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # LLaDA registers as AutoModelForCausalLM (despite being a diffusion LM)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Disable fused CE if supported (SDAR has this; LLaDA may not)
    if hasattr(model, "config") and hasattr(model.config, "fuse_cross_entropy"):
        model.config.fuse_cross_entropy = False

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # LLaDA stores mask_token_id in model config, not tokenizer
    if tokenizer.mask_token_id is not None:
        mask_id = tokenizer.mask_token_id
    elif hasattr(model, "config") and hasattr(model.config, "mask_token_id") and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id
    else:
        raise ValueError("Cannot find mask_token_id in tokenizer or model config")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading dataset")
    data_path = "../data/" + config.dataset.optimization_data + ".json"
    with open(data_path, "r") as f:
        dataset_load = json.load(f)

    prompt_list   = [x["prompt"]   for x in dataset_load]
    response_list = [x["response"] for x in dataset_load]

    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))
    _, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num   = config.training.post_num
    block_size = config.training.block_size

    # ── Tool response token IDs (for Agentic Noising) ─────────────────────────
    # Support multi-token <tool_response> / </tool_response> sequences.
    tool_resp_start_ids = tokenizer.encode("<tool_response>",  add_special_tokens=False)
    tool_resp_end_ids   = tokenizer.encode("</tool_response>", add_special_tokens=False)
    use_tool_masking    = bool(tool_resp_start_ids and tool_resp_end_ids)
    logger.info(
        f"tool_response tokens: start={tool_resp_start_ids} end={tool_resp_end_ids} "
        f"use_tool_masking={use_tool_masking}"
    )

    # ── Block attention bias for LLaDA (single sequence L0+L1) ──────────────
    # LLaDA's causal-block attention:
    #   - prompt (L0): each token attends to all prior tokens (causal)
    #   - response (L1): split into blocks of block_size;
    #     block k attends to all prompt + blocks 0..k-1 + within block k (bidirectional)
    def make_llada_block_attention(L0, L1, block_size):
        N    = L0 + L1
        bias = torch.zeros(1, 1, N, N, dtype=torch.bool)
        # Prompt: causal (lower-triangular)
        for i in range(L0):
            bias[0, 0, i, :i+1] = True
        # Response: block causal
        for bi in range((L1 + block_size - 1) // block_size):
            rs = L0 + bi * block_size
            re = min(L0 + (bi + 1) * block_size, N)
            # attend to all prompt tokens
            bias[0, 0, rs:re, :L0] = True
            # attend to all previous response blocks
            bias[0, 0, rs:re, L0:rs] = True
            # bidirectional within this block
            bias[0, 0, rs:re, rs:re] = True
        return bias  # (1, 1, N, N) bool

    basic_block_attention = make_llada_block_attention(L0, L1, block_size).cpu()

    # ── Helper: find multi-token span positions ───────────────────────────────
    def find_span_positions(input_ids, start_ids, end_ids, seq_start):
        """
        For each batch item, find all (span_start, span_end) pairs where
        span_start is the absolute index of the first token of start_ids and
        span_end is the absolute index just past the last token of end_ids.
        Also returns has_unclosed[b] = True if a start_ids was found with no
        matching end_ids.
        """
        B = input_ids.shape[0]
        slen, elen = len(start_ids), len(end_ids)
        all_spans, has_unclosed = [], []
        for b in range(B):
            seq = input_ids[b, seq_start:].tolist()
            spans, unclosed = [], False
            i = 0
            while i <= len(seq) - slen:
                if seq[i:i + slen] == start_ids:
                    start_abs = seq_start + i
                    j = i + slen
                    found = False
                    while j <= len(seq) - elen:
                        if seq[j:j + elen] == end_ids:
                            spans.append((start_abs, seq_start + j + elen))
                            i = j + elen
                            found = True
                            break
                        j += 1
                    if not found:
                        unclosed = True
                        break
                else:
                    i += 1
            all_spans.append(spans)
            has_unclosed.append(unclosed)
        return all_spans, has_unclosed

    # ── Agentic noising for LLaDA (single sequence, no doubling) ────────────
    # LLaDA input: [prompt + noisy_response], shape (B, L0+L1)
    # Labels: original clean [prompt + response], loss only on masked positions
    def collect_training_data(input_ids):
        B, L = input_ids.shape
        lower = config.training.lower_p
        upper = config.training.upper_p

        if use_tool_masking:
            spans, _ = find_span_positions(input_ids, tool_resp_start_ids, tool_resp_end_ids, start_pos)
        else:
            spans = [[] for _ in range(B)]

        noisy_ids_list, pmask_list = [], []

        for b in range(B):
            prob_ramp  = torch.empty(L1).uniform_(lower, upper)
            rand_tail  = torch.rand(L1)
            pmask_tail = rand_tail <= prob_ramp

            # Don't mask tool_response spans (Agentic Noising)
            for span_start, span_end in spans[b]:
                pmask_tail[span_start - start_pos : span_end - start_pos] = False

            pmask_b = torch.cat([torch.zeros(L0, dtype=torch.bool), pmask_tail], dim=0)

            noisy_b = input_ids[b].clone()
            noisy_b[L0:].masked_fill_(pmask_tail, mask_id)

            noisy_ids_list.append(noisy_b)
            pmask_list.append(pmask_b)

        noisy_input_ids = torch.stack(noisy_ids_list, dim=0)   # (B, L)
        p_mask          = torch.stack(pmask_list, dim=0).to(torch.bool)

        # Suppress post_num padding tokens from loss
        pad_resp = (input_ids == pad_id) & p_mask
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))

        labels = input_ids.clone()  # clean labels

        # tok_idx_ext not used for LLaDA (no position_ids), keep shape consistent
        tok_idx_ext = torch.zeros_like(noisy_input_ids)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        return (
            noisy_input_ids[keep],
            p_mask[keep],
            tok_idx_ext[keep],
            labels[keep],
        )

    # Filter samples with unclosed <tool_response> spans
    if use_tool_masking:
        _, unclosed = find_span_positions(input_ids_lm, tool_resp_start_ids, tool_resp_end_ids, start_pos)
        valid_indices = [i for i, bad in enumerate(unclosed) if not bad]
        n_filtered = len(input_ids_lm) - len(valid_indices)
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} samples with unclosed tool_response span")
            input_ids_lm = input_ids_lm[valid_indices]

    extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(input_ids_lm)

    def simple_collate(batch):
        eii, pm, tie, lbl = zip(*batch)
        return {
            "extended_input_ids": torch.stack(eii),
            "p_mask":             torch.stack(pm),
            "tok_idx_ext":        torch.stack(tie),
            "labels":             torch.stack(lbl),
        }

    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels)

    total_batch_size = (
        config.training.batch_size_lm
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size)
    num_train_epochs  = config.training.num_train_epochs
    max_train_steps   = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    train_dataloader = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=0,
    )

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("***** Running LLaDA SFT training *****")
    logger.info(f"  Dataset size:      {len(dataset_load)}")
    logger.info(f"  Dropped (too long):{drop_num}")
    logger.info(f"  Training samples:  {len(dataset_lm)}")
    logger.info(f"  Max steps:         {max_train_steps}")
    logger.info(f"  Batch per device:  {config.training.batch_size_lm}")
    logger.info(f"  Total batch size:  {total_batch_size}")
    logger.info(f"  Block size:        {block_size}")

    import torch.nn.functional as F

    def forward_process(noisy_input_ids, p_mask, _tok_idx_ext, labels):
        B, L = p_mask.shape
        device = noisy_input_ids.device

        # Effective denominator: trainable tokens per sample (exclude tool_response spans)
        if use_tool_masking:
            spans, _ = find_span_positions(noisy_input_ids, tool_resp_start_ids, tool_resp_end_ids, start_pos)
            L1_s = [
                max(1, L1 - sum(end - start for start, end in spans[b]))
                for b in range(B)
            ]
        else:
            L1_s = [L1] * B

        # 不传 attention_mask：避免 LLaDA 内部创建 (1,1,L,L) 的 bidirectional_attention_bias
        # 触发该路径会导致 F.scaled_dot_product_attention 退化为 O(L²) 内存，引发 OOM。
        # Loss 计算通过 p_mask 已经忽略 padding 位置，不需要 attention_mask。
        logits = model(
            input_ids=noisy_input_ids,
        ).logits  # (B, L, V)

        log_probs = F.log_softmax(logits, dim=-1)
        logp_tok  = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
        loss = -(logp_tok * p_mask).sum(dim=1) / torch.tensor(L1_s, dtype=torch.float, device=device)
        return loss.sum() / B

    from tqdm.auto import tqdm
    global_steps = 0
    first_epoch  = config.training.get("resume_epoch", 0) or 0

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(progress_bar, start=1):
            eii     = batch["extended_input_ids"].to(accelerator.device)
            pm      = batch["p_mask"].to(accelerator.device)
            tie     = batch["tok_idx_ext"].to(accelerator.device)
            lbl     = batch["labels"].to(accelerator.device)

            loss = forward_process(eii, pm, tie, lbl)
            global_steps += 1
            accelerator.backward(loss)

            if accelerator.is_main_process:
                wandb.log({"train_loss": loss.item()}, step=global_steps)

            if step % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

        accelerator.wait_for_everyone()
        save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name, epoch=epoch)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()  # 释放 ZeRO-3 gather 后残留的显存，防止 NCCL shutdown OOM
    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name, epoch=None):
    import glob, importlib, inspect, time, json, shutil
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            for p in ckpts[: len(ckpts) - checkpoints_total_limit + 1]:
                shutil.rmtree(p, ignore_errors=True)

    ckpt_name = f"ckpt_{epoch}" if epoch is not None else "ckpt"
    save_base  = output_dir / ckpt_name
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict    = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))
        # copy modeling code so the checkpoint is self-contained
        modeling_src = Path(config.model.pretrained_model) / "modeling_llada.py"
        if modeling_src.exists():
            shutil.copy2(modeling_src, save_dir / "modeling_llada.py")
        with (save_base / "metadata.json").open("w") as f:
            json.dump({"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        logger.info(f"Saved checkpoint to {save_dir}")


if __name__ == "__main__":
    main()
