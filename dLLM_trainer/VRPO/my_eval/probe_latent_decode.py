#!/usr/bin/env python3
"""
probe_latent_decode.py — 检验 coconut latent only 模型的 z_k 解码质量

流程：
  1. 加载 checkpoint 模型
  2. 对每个样本生成 K 个 latent 向量 z_0…z_{K-1}（SIM-CoT 自指代生成）
  3. 把每个 z_k 当作 embedding prefix，greedy decode 后续 token
  4. 对比解码结果与真实 plan 句子

用法（dionysos 单卡）：
  CUDA_VISIBLE_DEVICES=0 python my_eval/probe_latent_decode.py \
    --checkpoint /common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_latent_only/checkpoint-90 \
    --data /research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data/sft_with_plans.jsonl \
    --n_samples 100 --max_decode_tokens 80 \
    --output my_eval/results/latent_probe_ckpt90.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── 与训练代码保持一致的工具函数 ──────────────────────────────────────────────

def split_plan_sentences(plan: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', plan.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [plan.strip()]


def build_causal_bias(L: int, device: torch.device) -> torch.Tensor:
    bias = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    return bias.unsqueeze(0).unsqueeze(0)


def get_wte(model):
    m = model.module if hasattr(model, "module") else model
    return m.model.transformer.wte


# ── Latent 生成（与训练代码完全一致） ────────────────────────────────────────

@torch.no_grad()
def generate_latents(model, context_ids: torch.Tensor, K: int) -> List[torch.Tensor]:
    """生成 K 个 z_k。context_ids shape: (L,)（不含 batch 维）"""
    device = context_ids.device
    wte = get_wte(model)
    embeds = wte(context_ids.unsqueeze(0))  # [1, L, d]
    z_list = []
    for _ in range(K):
        bias = build_causal_bias(embeds.shape[1], device)
        out = model(inputs_embeds=embeds, attention_bias=bias, output_hidden_states=True)
        z_k = out.hidden_states[-1][:, -1:, :]  # [1, 1, d]
        z_list.append(z_k.detach())
        embeds = torch.cat([embeds, z_k], dim=1)
    return z_list


# ── Greedy decode from z_k ────────────────────────────────────────────────────

@torch.no_grad()
def decode_from_zk(
    model,
    tokenizer,
    z_k: torch.Tensor,          # [1, 1, d]
    max_tokens: int = 80,
    stop_on_eos: bool = True,
) -> str:
    """
    把 z_k 当作第 0 号位置的 embedding，greedy decode 后续 token。
    这与训练时的重建 forward 完全对应：
        dec_in = [z_k, embed(tgt[0]), ..., embed(tgt[T-2])]
        CE 目标是 tgt[0..T-1]
    所以从 z_k 出发 greedy decode 的结果就是模型"预期"的重建文本。
    """
    device = z_k.device
    wte = get_wte(model)
    eos_id = tokenizer.eos_token_id

    current_embeds = z_k.clone()  # [1, 1, d]
    generated = []

    for _ in range(max_tokens):
        bias = build_causal_bias(current_embeds.shape[1], device)
        out = model(inputs_embeds=current_embeds, attention_bias=bias)
        logits = out.logits[0, -1, :]           # [vocab]
        next_token = int(logits.argmax())

        if stop_on_eos and next_token == eos_id:
            break
        generated.append(next_token)

        # 只追加新 token 的 embedding，不累积整个序列（节省显存）
        next_emb = wte(torch.tensor([[next_token]], device=device, dtype=torch.long))
        current_embeds = torch.cat([current_embeds, next_emb], dim=1)

        # 简单停止条件：两个换行符连续出现
        decoded_so_far = tokenizer.decode(generated, skip_special_tokens=True)
        if "\n\n" in decoded_so_far or "<|im_end|>" in decoded_so_far:
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── 相似度度量（不需要额外依赖） ──────────────────────────────────────────────

def word_overlap_f1(pred: str, gold: str) -> float:
    """简单的词级别 F1（小写）"""
    pred_words = set(pred.lower().split())
    gold_words = set(gold.lower().split())
    if not pred_words or not gold_words:
        return 0.0
    common = pred_words & gold_words
    p = len(common) / len(pred_words)
    r = len(common) / len(gold_words)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_latent_only/checkpoint-90",
    )
    parser.add_argument(
        "--data",
        default="/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data/sft_with_plans.jsonl",
    )
    parser.add_argument("--n_samples",         type=int, default=100)
    parser.add_argument("--max_decode_tokens", type=int, default=80)
    parser.add_argument("--max_ctx_len",       type=int, default=4096)
    parser.add_argument("--output",            default="my_eval/results/latent_probe_ckpt90.jsonl")
    parser.add_argument("--device",            default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"加载 tokenizer + 模型: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()
    print("模型加载完成\n")

    def tok(text: str) -> List[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def build_context_ids(messages) -> torch.Tensor:
        parts = []
        for msg in messages[:2]:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        ids = tok("".join(parts))[-args.max_ctx_len:]
        return torch.tensor(ids, dtype=torch.long, device=device)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    samples = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= args.n_samples:
                break
    print(f"共 {len(samples)} 个样本\n")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    all_f1 = []
    results = []

    with open(out_path, "w") as f_out:
        for idx, sample in enumerate(tqdm(samples, desc="probe")):
            question = sample["messages"][1]["content"] if len(sample["messages"]) > 1 else ""
            plan = sample.get("plan", "")
            if not plan:
                continue

            sents = split_plan_sentences(plan)
            if not sents:
                continue

            context_ids = build_context_ids(sample["messages"])

            try:
                z_list = generate_latents(model, context_ids, len(sents))
            except Exception as e:
                tqdm.write(f"[WARN] sample {idx}: latent gen failed: {e}")
                continue

            decoded_steps = []
            f1_per_step = []
            for k, z_k in enumerate(z_list):
                try:
                    decoded = decode_from_zk(
                        model, tokenizer, z_k,
                        max_tokens=args.max_decode_tokens,
                    )
                except Exception as e:
                    tqdm.write(f"[WARN] sample {idx} z_{k}: decode failed: {e}")
                    decoded = ""
                f1 = word_overlap_f1(decoded, sents[k])
                decoded_steps.append(decoded)
                f1_per_step.append(f1)

            avg_f1 = sum(f1_per_step) / len(f1_per_step) if f1_per_step else 0.0
            all_f1.append(avg_f1)

            record = {
                "idx":            idx,
                "question":       question[:200],
                "true_plan":      plan,
                "true_steps":     sents,
                "decoded_steps":  decoded_steps,
                "f1_per_step":    [round(x, 4) for x in f1_per_step],
                "avg_f1":         round(avg_f1, 4),
            }
            results.append(record)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"完成！{len(results)} 个样本")
    print(f"平均词 F1: {sum(all_f1)/len(all_f1):.4f}" if all_f1 else "无结果")
    print(f"结果保存到: {out_path}")
    print()

    # 打印前 5 个样本的对比
    print("=== 前 5 个样本对比 ===")
    for r in results[:5]:
        print(f"\n[样本 {r['idx']}] Q: {r['question'][:80]}...")
        for k, (true_s, dec_s, f1) in enumerate(
            zip(r["true_steps"], r["decoded_steps"], r["f1_per_step"])
        ):
            print(f"  step {k}:")
            print(f"    真实: {true_s}")
            print(f"    解码: {dec_s}")
            print(f"    F1  : {f1:.4f}")

    print(f"\n结果文件: {out_path}")


if __name__ == "__main__":
    main()
