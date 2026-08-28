#!/usr/bin/env python3
"""
GRPO 主训练脚本（LLaDA diffusion model）。

输入数据格式（由 my_eval/make_grpo_data.py 生成）：
    {"prompt": str, "completion": str, "advantage": float}

典型调用（见 run_grpo.sh）：
    accelerate launch --config_file ... my_train/llada_grpo_train.py \
        --config recipes/grpo_llada.yaml
"""

import logging, os, sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, DPOConfig

# ── transformers 5.x 兼容补丁（LLaDA tie_weights / all_tied_weights_keys）──
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# 1. all_tied_weights_keys property（新版 transformers 需要，LLaDA 原模型无此属性）
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self):
        return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val):
        object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

# 2. _get_tied_weight_keys：兼容 dict / list / tuple 格式
def _patched_get_tied_weight_keys(module):
    tied_weight_keys = []
    for name, submodule in module.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", None) or {}
        if isinstance(tied, dict):
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied.keys()])
        elif isinstance(tied, (list, tuple)):
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied])
    return tied_weight_keys

_mu._get_tied_weight_keys = _patched_get_tied_weight_keys

# 3. _finalize_model_loading：让 tie_weights() 忽略新版 transformers 传入的 missing_keys 等 kwargs
if "_finalize_model_loading" in PreTrainedModel.__dict__:
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

# 4. _adjust_tied_keys_with_tied_pointers：安全包装（可选）
if hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
    _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers
    def _safe_adjust(self, *args, **kwargs):
        return _orig_adjust(self, *args, **kwargs)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

# 5. caching_allocator_warmup：tp_plan=None 时跳过（LLaDA 无 tensor parallel plan）
if hasattr(_mu, "caching_allocator_warmup"):
    _orig_warmup = _mu.caching_allocator_warmup
    def _safe_warmup(model, device_map, hf_quantizer=None):
        try:
            return _orig_warmup(model, device_map, hf_quantizer)
        except TypeError:
            pass  # tp_plan=None，跳过 warmup
    _mu.caching_allocator_warmup = _safe_warmup
# ────────────────────────────────────────────────────────────────────────────

from my_grpo_trainer import SDARGRPOTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class GRPOScriptArgs:
    dataset_path: str = field(default=None, metadata={"help": "GRPO 训练数据 jsonl"})
    max_samples: Optional[int] = field(default=None)
    block_length: int = field(default=128)
    num_mc: int = field(default=2)
    mask_token_id: Optional[int] = field(default=None)
    advantage_clip: float = field(default=5.0, metadata={"help": "advantage clipping 范围"})
    normalize_elbo: bool  = field(
        default=False,
        metadata={"help": (
            "将 ELBO 除以 policy token 数再计算 loss（sequence-level normalization）。"
            "False = 原始 ELBO（Round 1 行为）；True = per-token ELBO（ESPO 风格）。"
        )},
    )
    tool_resp_left:  str = field(default="<|im_start|>user\n<tool_response>")
    tool_resp_right: str = field(default="</tool_response><|im_end|>\n<|im_start|>assistant\n")


@dataclass
class GRPOConfig(DPOConfig):
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)
    max_prompt_length: Optional[int] = field(default=1024)
    max_completion_length: Optional[int] = field(default=4096)
    wandb_project: Optional[str] = field(default="llada_grpo")


# =============================================================================
# Dataset
# =============================================================================

def load_grpo_dataset(path, max_samples=None):
    if path is None:
        raise ValueError("--dataset_path 必须指定 GRPO 训练数据 jsonl")
    dataset = datasets.load_dataset("json", data_files=path, split="train")
    required = {"prompt", "completion", "advantage"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"数据集缺少字段: {missing}，现有: {dataset.column_names}")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


# =============================================================================
# Main
# =============================================================================

def main(script_args: GRPOScriptArgs, training_args: GRPOConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Script args: {script_args}")
    logger.info(f"Training args: {training_args}")

    # ── Checkpoint resume ────────────────────────────────────────────────────
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        logger.info(f"Resume from checkpoint: {last_checkpoint}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("*** Loading GRPO dataset ***")
    train_dataset = load_grpo_dataset(script_args.dataset_path, script_args.max_samples)
    logger.info(f"Train samples: {len(train_dataset)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info("*** Loading tokenizer ***")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("*** Loading model ***")
    torch_dtype = (
        getattr(torch, model_args.dtype)
        if model_args.dtype and model_args.dtype != "auto"
        else torch.bfloat16
    )
    attn_impl = getattr(model_args, "attn_implementation", None) or "eager"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ZeRO-3 检测：通过环境变量 USE_ZERO_STAGE（由 run_phase4_elbopg.sh 显式设置）
    # ZeRO-2（44GB/卡）在 A100 40GB 上放不下 8B 模型，必须用 ZeRO-3（参数也分片，每卡 ~16GB）。
    zero_stage = int(os.environ.get("USE_ZERO_STAGE", "2"))
    is_zero3 = (zero_stage == 3)
    logger.info(f"ZeRO stage: {zero_stage} (is_zero3={is_zero3})")

    if is_zero3:
        # ZeRO-3：在 deepspeed.zero.Init() context 下分片加载——每个 rank 只持有 1/8 参数（2GB）。
        # 不调用 .to(cuda)：DeepSpeed 管理参数分片和设备位置。
        # 要求 accel yaml 里 zero3_init_flag: true 且在 accelerate launch 下运行。
        import deepspeed
        with deepspeed.zero.Init(dtype=torch_dtype):
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=model_args.trust_remote_code,
                attn_implementation=attn_impl,
            )
        logger.info("Model loaded via deepspeed.zero.Init() (ZeRO-3 sharded)")
    else:
        # ZeRO-2：加载完整模型到 CPU，再手动 .to(gpu)。
        # 不用 device_map=cuda 避免触发 hf_device_map 专用路径与 DeepSpeed 的兼容问题。
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        model = model.to(f"cuda:{local_rank}")
        logger.info(f"Model loaded to cuda:{local_rank} (ZeRO-2)")

    # LLaDA config 里没有 use_cache 字段，需手动设置防止 forward 抛 AttributeError
    model.config.use_cache = False

    # 启用 LLaDA 内置的层级 activation checkpointing（每个 transformer block 单独 checkpoint）。
    # 把反向传播激活内存从 O(32层 × N²) 降到 O(1层 × N²)，解决 40GB GPU OOM 问题。
    # 使用字符串 "whole_layer"（ActivationCheckpointingStrategy 是 StrEnum，与字符串相等）。
    if hasattr(model, 'model') and hasattr(model.model, 'set_activation_checkpointing'):
        model.model.set_activation_checkpointing("whole_layer")
        logger.info("LLaDA activation checkpointing enabled: whole_layer")

    # ── Trainer ──────────────────────────────────────────────────────────────
    logger.info("*** Initializing GRPO Trainer ***")
    trainer = SDARGRPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        block_length=script_args.block_length,
        num_mc=script_args.num_mc,
        mask_token_id=script_args.mask_token_id,
        advantage_clip=script_args.advantage_clip,
        normalize_elbo=script_args.normalize_elbo,
        tool_resp_left=script_args.tool_resp_left,
        tool_resp_right=script_args.tool_resp_right,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("*** Starting GRPO training ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Saving model ***")
    import os as _os
    _os.makedirs(training_args.output_dir, exist_ok=True)

    if trainer.is_deepspeed_enabled:
        if is_zero3:
            # ZeRO-3：参数分散在各 rank，需要 consolidate 后保存。
            # trainer.save_model() 内部调用 accelerator.get_state_dict(model, unwrap=False)
            # → DeepSpeed 的 zero3_save_16bit_model 路径，在 rank0 上聚合并写出 bf16 权重。
            # 要求 accel yaml 里 zero3_save_16bit_model: true。
            trainer.save_model(training_args.output_dir)
            if local_rank == 0:
                tokenizer.save_pretrained(training_args.output_dir)
                logger.info(f"ZeRO-3 model consolidated → {training_args.output_dir}")
        else:
            # ZeRO-2：每个 rank 有完整参数，rank0 直接 save_pretrained 流式写出。
            # 避免 trainer.save_model() 在 CPU 上做全量 clone（~32GB × 2）造成 OOM。
            if local_rank == 0:
                _unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                _unwrapped.save_pretrained(
                    training_args.output_dir,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(training_args.output_dir)
                logger.info(f"ZeRO-2 model → {training_args.output_dir}")
            trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Model → {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArgs, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
