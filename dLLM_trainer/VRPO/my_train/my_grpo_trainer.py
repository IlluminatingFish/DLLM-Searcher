"""
GRPO Trainer for LLaDA（扩散语言模型）。

继承 SDARDPOTrainer 以复用 block diffusion ELBO 计算，
将 DPO loss 替换为 GRPO loss：
    L = -mean( advantage_i × ELBO_i )

输入数据格式（每条是一个 rollout）:
    {
        "prompt":     str,       # 初始 prompt（system + user 问题）
        "completion": str,       # 模型完整轨迹（含 tool_response）
        "advantage":  float,     # 预计算的组内 z-score 归一化优势
    }
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOConfig
from accelerate import logging

from my_dpo_trainer import SDARDPOTrainer, get_trainable_mask

logger = logging.get_logger(__name__)


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorForGRPO(DataCollatorMixin):
    return_tensors: str = "pt"

    def torch_call(self, examples: list) -> dict:
        batch = {
            "prompt_input_ids":    [torch.tensor(ex["prompt_input_ids"])    for ex in examples],
            "completion_input_ids":[torch.tensor(ex["completion_input_ids"])for ex in examples],
            "trainable_mask":      [torch.tensor(ex["trainable_mask"], dtype=torch.bool) for ex in examples],
            "advantage":           torch.tensor([ex["advantage"] for ex in examples], dtype=torch.float32),
        }
        # old_elbo 是可选字段（Round 2+ 数据）
        if "old_elbo" in examples[0] and examples[0]["old_elbo"] is not None:
            batch["old_elbo"] = torch.tensor([ex["old_elbo"] for ex in examples], dtype=torch.float32)
        else:
            batch["old_elbo"] = None
        return batch


# =============================================================================
# GRPO Trainer
# =============================================================================

class SDARGRPOTrainer(SDARDPOTrainer):
    """
    GRPO trainer for LLaDA.
    DPO 相比的变化：
    - 输入：单条 rollout + pre-computed advantage（无 reference model）
    - loss：L = -mean(advantage × ELBO)，advantage clip 到 [-5, 5] 防止梯度爆炸
    - 若 batch 含 old_elbo 字段，使用 PPO-style ratio clipping 稳定训练
    - 无需 ref_model（不加 KL 惩罚项，简化版）
    """

    # 让 TRL/DPOTrainer 认为 is_peft_model=True，跳过 ZeRO-3 下的 ref_model 强制要求。
    # GRPO 不使用 ref_model，setter 设为 no-op 防止 DPOTrainer.__init__ 覆写此值。
    @property
    def is_peft_model(self):
        return True

    @is_peft_model.setter
    def is_peft_model(self, value):
        pass  # no-op: 保持 True，防止 DPOTrainer.__init__ 覆写

    def __init__(self, model, ref_model=None, args=None,
                 block_length=128, num_mc=2, mask_token_id=None,
                 tool_resp_left="<|im_start|>user\n<tool_response>",
                 tool_resp_right="</tool_response><|im_end|>\n<|im_start|>assistant\n",
                 advantage_clip=5.0,
                 clip_eps=0.2,
                 normalize_elbo=False,
                 **kwargs):
        self.advantage_clip = advantage_clip
        self.clip_eps = clip_eps
        self.normalize_elbo = normalize_elbo
        super().__init__(
            model=model, ref_model=None,  # GRPO 不使用 ref_model；is_peft_model=True 绕过 ZeRO-3 检查
            args=args,
            block_length=block_length, num_mc=num_mc,
            mask_token_id=mask_token_id,
            tool_resp_left=tool_resp_left,
            tool_resp_right=tool_resp_right,
            **kwargs
        )
        self.data_collator = DataCollatorForGRPO()
        logger.info(
            f"GRPO: block_length={block_length}, num_mc={num_mc}, "
            f"adv_clip={advantage_clip}, clip_eps={clip_eps}, "
            f"normalize_elbo={normalize_elbo}"
        )

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

        prompt_input_ids = tokenizer(
            features["prompt"], add_special_tokens=False
        )["input_ids"]
        completion_input_ids = tokenizer(
            features["completion"], add_special_tokens=False
        )["input_ids"]
        completion_input_ids = completion_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            completion_input_ids = completion_input_ids[:max_completion_length]

        trainable_mask = get_trainable_mask(
            features["completion"], tokenizer,
            self.tool_resp_left, self.tool_resp_right
        )
        trainable_mask = trainable_mask + [True]            # EOS trainable
        trainable_mask = trainable_mask[:len(completion_input_ids)]

        result = {
            "prompt_input_ids":     prompt_input_ids,
            "completion_input_ids": completion_input_ids,
            "trainable_mask":       trainable_mask,
            "advantage":            features["advantage"],
        }
        if "old_elbo" in features:
            result["old_elbo"] = features["old_elbo"]
        return result

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        device = next(model.parameters()).device
        metrics = {}

        advantages = batch["advantage"].to(device)             # [N]
        advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)

        elbos = []
        for i in range(len(batch["prompt_input_ids"])):
            prompt_ids     = batch["prompt_input_ids"][i].to(device)
            completion_ids = batch["completion_input_ids"][i].to(device)
            trainable_mask = batch["trainable_mask"][i].to(device)

            prompt_len     = len(prompt_ids)
            completion_len = len(completion_ids)

            # 全序列 = [prompt | completion]
            full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
            full_trainable = torch.cat([
                torch.zeros(prompt_len, dtype=torch.bool, device=device),
                trainable_mask
            ])

            seeds = torch.randint(0, 2**20, (self.num_mc,), device=device)
            elbo  = self._get_elbo_mc(
                model, full_ids, prompt_len, completion_len, full_trainable, seeds
            )
            # _get_elbo_mc 返回 mean ELBO（正值越大表示 log prob 越高）

            # FIX: sequence-level length normalization（可选）
            # 不归一化时 ELBO ∝ 序列长度（长轨迹梯度量级远大于短轨迹），
            # 导致 gradient norm 600-2600、近 100% clipping，训练信号被长度噪声淹没。
            # 除以 policy token 数后，不同长度轨迹的梯度量级变得可比。
            if self.normalize_elbo:
                n_policy = trainable_mask.sum().float().clamp(min=1)
                elbo = elbo / n_policy

            elbos.append(elbo)

            torch.cuda.empty_cache()

        elbos = torch.stack(elbos)       # [N]

        # PPO-style ratio clipping（若 batch 有 old_elbo，即 Round 2+ 数据）
        # ratio = exp(ELBO_new - ELBO_old)，clip 到 [1-ε, 1+ε] 防止策略偏离过大。
        # Round 1 数据无 old_elbo 时退化为普通 GRPO loss。
        if "old_elbo" in batch and batch["old_elbo"] is not None:
            old_elbos = batch["old_elbo"].to(device)              # [N]
            ratio = torch.exp(elbos - old_elbos.detach())
            ratio_clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            # PPO clip objective: -min(r*A, clip(r)*A)
            loss = -torch.min(ratio * advantages, ratio_clipped * advantages).mean()
            metrics[f"{'eval_' if train_eval == 'eval' else ''}ratio/mean"] = ratio.mean().item()
        else:
            # 无 old_elbo：普通 GRPO loss（Round 1 兼容）
            loss = -(advantages * elbos).mean()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"]              = loss.item()
        metrics[f"{prefix}rewards/mean"]      = advantages.mean().item()
        metrics[f"{prefix}rewards/positive"]  = (advantages > 0).float().mean().item()
        metrics[f"{prefix}elbo/mean"]         = elbos.mean().item()
        # normalize_elbo=True 时 elbo/mean 单位是 per-token ELBO（量级约 -1~-5），
        # normalize_elbo=False 时量级约 -1400（即 ~700 policy tokens × ~2 nats/token）。

        return loss, metrics

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """覆写 DPOTrainer._prepare_dataset 以支持 GRPO 格式（无 chosen/rejected）。"""
        from accelerate.state import PartialState

        map_kwargs = {}
        if hasattr(dataset, "num_rows"):  # 非 IterableDataset
            map_kwargs["num_proc"] = getattr(args, "dataset_num_proc", None)
            map_kwargs["writer_batch_size"] = 10
            map_kwargs["desc"] = f"Tokenizing GRPO {dataset_name} dataset"

        with PartialState().main_process_first():
            dataset = dataset.map(
                self.tokenize_row,
                remove_columns=dataset.column_names,  # 删除所有原始列
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": getattr(args, "max_prompt_length", None),
                    "max_completion_length": getattr(args, "max_completion_length", None),
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )
        return dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "completion_input_ids",
                "trainable_mask",
                "advantage",
            ]
