#!/usr/bin/env python3
"""
Plan A: DPO on x0pred model, x0pred head FROZEN.

原理：x0pred head 不参与 RL 更新，只优化 transformer 主体。
避免 RL 破坏 think 连续空间表示，但 head 可能与主模型逐渐不对齐。

用法:
  accelerate launch --config_file recipes/accelerate_configs/zero3_cpu_offload.yaml \
    --num_processes 8 my_train/llada_dpo_train_a.py \
    --model_path /common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized \
    --data_path  data/dpo_pairs_plan_ab_ref.jsonl \
    --output_dir /common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_a/model
"""

import argparse, json, os, sys
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).parent))
from llada_dpo_trainer import LLaDADPOTrainer

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

# transformers 5.x 兼容：LLaDA 没有 all_tied_weights_keys 属性
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--output_dir",  default="output/dpo_plan_a")
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

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    data = [json.loads(l) for l in open(args.data_path) if l.strip()]
    dataset = Dataset.from_list(data)
    if rank == 0:
        print(f"DPO 数据: {len(dataset)} 对")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # 在加载模型之前设置 flash_attention=True，使 __init__ 中 flash_attn_func 正确初始化
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    _cfg.flash_attention = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=_cfg, trust_remote_code=True,
        dtype=torch.bfloat16
    )
    model.config.use_cache = False
    if rank == 0:
        fa_count = sum(1 for m in model.modules() if getattr(m, "flash_attn_func", None) is not None)
        print(f"Flash Attention 已激活：{fa_count} 个注意力层")
    # ZeRO-3 内存不够同时存两个 8B 模型，使用预计算的 ref log probs（见 precompute_ref_logps.py）
    ref_model = None

    # ── Plan A: 冻结 x0pred head（如果存在） ─────────────────────────────────
    ckpt_dir = Path(args.model_path)
    x0_head_path = ckpt_dir / "x0_head.pt"
    if x0_head_path.exists():
        if rank == 0:
            print(f"发现 x0_head.pt，加载并冻结 (Plan A)")
        # x0_head 只是辅助组件，不是 model 的成员，训练时自然不会更新
        # 这里显式标记为不需要梯度
        import torch.nn as nn
        x0_head = nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False)
        x0_head.load_state_dict(torch.load(x0_head_path, map_location="cpu"))
        for p in x0_head.parameters():
            p.requires_grad = False
        if rank == 0:
            print("x0_head 已冻结（不参与 RL 训练）")
    else:
        if rank == 0:
            print("未找到 x0_head.pt，以标准 LLaDA DPO 模式运行")

    # ── DPO 配置 ──────────────────────────────────────────────────────────────
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

    trainer = LLaDADPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        block_length=args.block_length,
        num_mc=args.num_mc,
    )

    if rank == 0:
        print(f"\nPlan A: x0pred head frozen, DPO on main model")
        print(f"lr={args.lr}  beta={args.beta}  epochs={args.epochs}")
        print(f"block_length={args.block_length}  num_mc={args.num_mc}\n")

    trainer.train()
    trainer.save_model(args.output_dir)
    if rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n训练完成，模型保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
