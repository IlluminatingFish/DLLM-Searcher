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
import torch.distributed as dist
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
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, think_mask):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.think_mask = think_mask  # bool (B,L): positions for continuous diffusion

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.think_mask[idx],
        )


def main():
    config = get_config()

    project_name = config.experiment.project
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
            name="sft_llada_reasoning",
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
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

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

    # ── Reasoning hyperparams ─────────────────────────────────────────────────
    lambda_c  = float(config.training.get("lambda_c",  0.1))
    sigma_min = float(config.training.get("sigma_min", 0.1))
    sigma_max = float(config.training.get("sigma_max", 1.0))
    logger.info(f"Reasoning config: lambda_c={lambda_c} sigma=[{sigma_min},{sigma_max}]")

    # ── cont_head: ε-prediction head for continuous think tokens ─────────────
    # Kept outside DeepSpeed model (small: d_model x d_model ~64MB for 8B).
    # Gradient sync is done manually after each backward (see training loop).
    d_model = getattr(model.config, "d_model", getattr(model.config, "hidden_size", 4096))
    cont_head = torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
    torch.nn.init.xavier_uniform_(cont_head.weight.float())
    cont_head = cont_head.to(accelerator.device)
    logger.info(f"cont_head: Linear({d_model},{d_model}) — {d_model*d_model/1e6:.1f}M params")

    # ── sigma_proj: conditions the model on the noise level σ ─────────────────
    # Without this, x_noisy = x_0 + sigma*eps gives no information about sigma,
    # so the optimal eps predictor collapses to E[eps]=0 (see derivation in
    # discussion: same x_noisy can arise from different (sigma, eps) pairs).
    # Zero-init final layer so the model starts as if no conditioning exists,
    # and learns to use it only if it helps (stable warm start).
    sigma_proj = torch.nn.Sequential(
        torch.nn.Linear(1, d_model),
        torch.nn.SiLU(),
        torch.nn.Linear(d_model, d_model),
    ).to(dtype=torch.bfloat16, device=accelerator.device)
    torch.nn.init.zeros_(sigma_proj[-1].weight)
    torch.nn.init.zeros_(sigma_proj[-1].bias)
    logger.info(f"sigma_proj: MLP(1->{d_model}->{d_model}) for sigma conditioning")

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
    # Separate optimizer for cont_head + sigma_proj (not wrapped by DeepSpeed)
    aux_modules = [cont_head, sigma_proj]
    aux_optimizer = AdamW(
        [p for m in aux_modules for p in m.parameters()],
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
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
    tool_resp_start_ids = tokenizer.encode("<tool_response>",  add_special_tokens=False)
    tool_resp_end_ids   = tokenizer.encode("</tool_response>", add_special_tokens=False)
    use_tool_masking    = bool(tool_resp_start_ids and tool_resp_end_ids)
    logger.info(
        f"tool_response tokens: start={tool_resp_start_ids} end={tool_resp_end_ids} "
        f"use_tool_masking={use_tool_masking}"
    )

    # ── Think span token IDs (for continuous latent diffusion) ────────────────
    # <think>...</think> positions use Gaussian embedding noise + MSE loss,
    # NOT discrete mask_id + CE loss.
    think_start_ids = tokenizer.encode("<think>",  add_special_tokens=False)
    think_end_ids   = tokenizer.encode("</think>", add_special_tokens=False)
    use_think_latent = bool(think_start_ids and think_end_ids)
    logger.info(
        f"think tokens: start={think_start_ids} end={think_end_ids} "
        f"use_think_latent={use_think_latent}"
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

    # ── Agentic noising with think-latent separation ─────────────────────────
    # think positions: pmask=False (no discrete mask), tracked in think_mask
    # tool_response positions: pmask=False (Agentic Noising), NOT in think_mask
    # all other response positions: discrete random mask → CE loss
    def collect_training_data(input_ids):
        B, L = input_ids.shape
        lower = config.training.lower_p
        upper = config.training.upper_p

        if use_tool_masking:
            spans, _ = find_span_positions(input_ids, tool_resp_start_ids, tool_resp_end_ids, start_pos)
        else:
            spans = [[] for _ in range(B)]

        if use_think_latent:
            think_spans, _ = find_span_positions(input_ids, think_start_ids, think_end_ids, start_pos)
        else:
            think_spans = [[] for _ in range(B)]

        noisy_ids_list, pmask_list, think_mask_list = [], [], []

        for b in range(B):
            prob_ramp  = torch.empty(L1).uniform_(lower, upper)
            rand_tail  = torch.rand(L1)
            pmask_tail = rand_tail <= prob_ramp

            # Exclude tool_response spans from discrete mask (Agentic Noising)
            for span_start, span_end in spans[b]:
                pmask_tail[span_start - start_pos : span_end - start_pos] = False

            # Exclude think spans from discrete mask; they use continuous diffusion
            think_mask_tail = torch.zeros(L1, dtype=torch.bool)
            for span_start, span_end in think_spans[b]:
                pmask_tail[span_start - start_pos : span_end - start_pos] = False
                think_mask_tail[span_start - start_pos : span_end - start_pos] = True

            pmask_b      = torch.cat([torch.zeros(L0, dtype=torch.bool), pmask_tail], dim=0)
            think_mask_b = torch.cat([torch.zeros(L0, dtype=torch.bool), think_mask_tail], dim=0)

            noisy_b = input_ids[b].clone()
            # Only mask non-think, non-tool_response positions
            noisy_b[L0:].masked_fill_(pmask_tail, mask_id)

            noisy_ids_list.append(noisy_b)
            pmask_list.append(pmask_b)
            think_mask_list.append(think_mask_b)

        noisy_input_ids = torch.stack(noisy_ids_list, dim=0)
        p_mask          = torch.stack(pmask_list, dim=0).to(torch.bool)
        think_mask      = torch.stack(think_mask_list, dim=0).to(torch.bool)

        # Suppress post_num padding tokens from loss
        pad_resp = (input_ids == pad_id) & p_mask
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))

        labels = input_ids.clone()
        tok_idx_ext = torch.zeros_like(noisy_input_ids)

        # Keep samples that have at least one discrete-masked token OR think token
        keep = (
            p_mask.view(p_mask.size(0), -1).any(dim=1) |
            think_mask.view(think_mask.size(0), -1).any(dim=1)
        )
        return (
            noisy_input_ids[keep],
            p_mask[keep],
            tok_idx_ext[keep],
            labels[keep],
            think_mask[keep],
        )

    # Filter samples with unclosed <tool_response> spans
    if use_tool_masking:
        _, unclosed = find_span_positions(input_ids_lm, tool_resp_start_ids, tool_resp_end_ids, start_pos)
        valid_indices = [i for i, bad in enumerate(unclosed) if not bad]
        n_filtered = len(input_ids_lm) - len(valid_indices)
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} samples with unclosed tool_response span")
            input_ids_lm = input_ids_lm[valid_indices]

    extended_input_ids, p_mask, tok_idx_ext, labels, think_mask = collect_training_data(input_ids_lm)

    def simple_collate(batch):
        eii, pm, tie, lbl, thm = zip(*batch)
        return {
            "extended_input_ids": torch.stack(eii),
            "p_mask":             torch.stack(pm),
            "tok_idx_ext":        torch.stack(tie),
            "labels":             torch.stack(lbl),
            "think_mask":         torch.stack(thm),
        }

    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, think_mask)

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
    # Cosine schedule for cont_head aux optimizer (mirrors main LR shape)
    aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        aux_optimizer, T_max=max_train_steps,
        eta_min=optimizer_config.learning_rate * 0.1,
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
    logger.info("***** Running LLaDA Reasoning SFT training *****")
    logger.info(f"  Dataset size:      {len(dataset_load)}")
    logger.info(f"  Dropped (too long):{drop_num}")
    logger.info(f"  Training samples:  {len(dataset_lm)}")
    logger.info(f"  Max steps:         {max_train_steps}")
    logger.info(f"  Batch per device:  {config.training.batch_size_lm}")
    logger.info(f"  Total batch size:  {total_batch_size}")
    logger.info(f"  Block size:        {block_size}")
    logger.info(f"  lambda_c={lambda_c}  sigma=[{sigma_min},{sigma_max}]")

    import torch.nn.functional as F

    # Get the embedding module once for hook registration.
    # LLaDA model hierarchy: HF wrapper -> .model (LLaDAModel) -> .transformer.wte
    _unwrapped = accelerator.unwrap_model(model)
    try:
        _embed_layer = _unwrapped.model.transformer.wte
    except AttributeError:
        _embed_layer = _unwrapped.get_input_embeddings()

    # Final-norm module (output feeds directly into ff_out/lm_head). We hook
    # this instead of passing output_hidden_states=True to the model, because
    # the latter keeps a Python-level reference to ALL 32 layers' hidden
    # states for the lifetime of `outputs`, which blocks PyTorch's autograd
    # from progressively freeing each layer's activations during backward —
    # a major (and avoidable) peak-memory increase under ZeRO-3 + CPU offload
    # with no gradient checkpointing (LLaDA does not support it).
    _final_norm_layer = _unwrapped.model.transformer.ln_f

    def forward_process(noisy_input_ids, p_mask, _tok_idx_ext, labels, think_mask_b):
        B, L  = p_mask.shape
        device = noisy_input_ids.device

        # Effective CE denominator per sample (exclude tool_response spans)
        if use_tool_masking:
            spans, _ = find_span_positions(noisy_input_ids, tool_resp_start_ids, tool_resp_end_ids, start_pos)
            L1_s = [
                max(1, L1 - sum(end - start for start, end in spans[b]))
                for b in range(B)
            ]
        else:
            L1_s = [L1] * B

        # ── Continuous noise for think positions ──────────────────────────────
        # Per-sample σ ~ Uniform(sigma_min, sigma_max); ε ~ N(0,I) in embedding space.
        # The hook below injects x_noisy = x_clean + σ*ε + sigma_emb(σ) at think
        # positions. sigma_emb is what lets cont_head calibrate its prediction
        # to the actual noise level instead of collapsing to E[eps]=0 (the
        # same x_noisy is otherwise consistent with many different (σ,ε) pairs).
        sigma = torch.empty(B, device=device).uniform_(sigma_min, sigma_max)   # (B,)
        eps   = torch.randn(B, L, d_model, dtype=torch.bfloat16, device=device) # (B,L,d)

        # ── Embedding forward hook ────────────────────────────────────────────
        # Fires inside model.forward() after ZeRO-3 has gathered the embedding
        # weights. We modify the output tensor in-place (returned = replaced output).
        def _embed_hook(module, args, output):
            # output: (B, L, d_model) clean embeddings from lookup — dtype is bf16
            # Cast sigma and eps to match output dtype to avoid "Index put dtype mismatch"
            sig = sigma.to(output.dtype).view(B, 1, 1)
            noise = eps.to(output.dtype)
            sig_emb = sigma_proj(sigma.view(B, 1).to(output.dtype)).view(B, 1, d_model)
            injected = output + sig * noise + sig_emb
            output_noisy = output.clone()
            output_noisy[think_mask_b] = injected[think_mask_b]
            return output_noisy

        # ── Final hidden state capture (replaces output_hidden_states=True) ───
        _final_hidden = {}
        def _capture_final_hidden(module, args, output):
            _final_hidden["h"] = output

        _handle_embed = _embed_layer.register_forward_hook(_embed_hook)
        _handle_final = _final_norm_layer.register_forward_hook(_capture_final_hidden)
        try:
            outputs = model(input_ids=noisy_input_ids)
        finally:
            _handle_embed.remove()
            _handle_final.remove()

        logits = outputs.logits  # (B, L, V)

        # ── Discrete CE loss at masked non-think positions ────────────────────
        p_mask_disc = p_mask & ~think_mask_b
        log_probs   = F.log_softmax(logits, dim=-1)
        logp_tok    = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,L)
        loss_ce = -(logp_tok * p_mask_disc).sum(dim=1) / torch.tensor(
            L1_s, dtype=torch.float, device=device
        )
        loss_ce = loss_ce.sum() / B

        # ── Continuous MSE loss at think positions (ε-prediction) ─────────────
        loss_mse = torch.tensor(0.0, device=device)
        if use_think_latent and think_mask_b.any() and "h" in _final_hidden:
            hidden_last = _final_hidden["h"]   # (B, L, d_model)
            eps_pred    = cont_head(hidden_last)       # (B, L, d_model)
            loss_mse    = F.mse_loss(
                eps_pred[think_mask_b].float(),
                eps[think_mask_b].float(),
            )

        return loss_ce, loss_mse

    from tqdm.auto import tqdm
    global_steps = 0
    first_epoch  = config.training.get("resume_epoch", 0) or 0

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        cont_head.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(progress_bar, start=1):
            eii = batch["extended_input_ids"].to(accelerator.device)
            pm  = batch["p_mask"].to(accelerator.device)
            tie = batch["tok_idx_ext"].to(accelerator.device)
            lbl = batch["labels"].to(accelerator.device)
            thm = batch["think_mask"].to(accelerator.device)

            loss_ce, loss_mse = forward_process(eii, pm, tie, lbl, thm)
            loss = loss_ce + lambda_c * loss_mse
            global_steps += 1

            # Backward: DeepSpeed handles LLaDA grads; cont_head grads also computed
            accelerator.backward(loss)

            if accelerator.is_main_process:
                wandb.log({
                    "train_loss":      loss.item(),
                    "loss_ce":         loss_ce.item(),
                    "loss_mse":        loss_mse.item(),
                    "loss_mse_scaled": (lambda_c * loss_mse).item(),
                }, step=global_steps)

            if step % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                aux_optimizer.step()
                # Broadcast cont_head/sigma_proj params from rank 0 → all ranks AFTER
                # optimizer step (safe: happens after DeepSpeed's reduce-scatter is done)
                if accelerator.num_processes > 1 and dist.is_initialized():
                    for m in aux_modules:
                        for p in m.parameters():
                            dist.broadcast(p.data, src=0)
                lr_scheduler.step()
                aux_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                aux_optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

        accelerator.wait_for_everyone()
        save_checkpoint(model, cont_head, sigma_proj, tokenizer, config, accelerator, config.model.optimized_name, epoch=epoch)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        wandb.finish()
    accelerator.end_training()


def save_checkpoint(model, cont_head, sigma_proj, tokenizer, config, accelerator, name, epoch=None):
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

        # Save cont_head/sigma_proj separately; not part of the HF model state_dict
        torch.save(cont_head.state_dict(), save_dir / "cont_head.pt")
        torch.save(sigma_proj.state_dict(), save_dir / "sigma_proj.pt")

        with (save_base / "metadata.json").open("w") as f:
            json.dump({"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        logger.info(f"Saved checkpoint to {save_dir} (incl. cont_head.pt)")


if __name__ == "__main__":
    main()
