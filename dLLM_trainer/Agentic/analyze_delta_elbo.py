#!/usr/bin/env python3
# ── 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES ──────────────────────────
# argparse 还没解析，直接手动从 sys.argv 提取 --gpu 值
import sys as _sys, os as _os
for _i, _a in enumerate(_sys.argv):
    if _a == "--gpu" and _i + 1 < len(_sys.argv):
        _os.environ["CUDA_VISIBLE_DEVICES"] = _sys.argv[_i + 1]
        break
# ───────────────────────────────────────────────────────────────────────────────

"""
analyze_delta_elbo.py — Advantage → ΔELBO update-effectiveness analysis

For each rollout trajectory in a training dataset, compute:
  ELBO_before = ELBO(trajectory | π_before)
  ELBO_after  = ELBO(trajectory | π_after)
  ΔELBO       = ELBO_after - ELBO_before   (normalized per policy token)

Then check:
  Spearman(advantage, ΔELBO)       — did the model move in the right direction?
  Sign-alignment rate              — what fraction have sign(adv) == sign(ΔELBO)?
  Effect magnitude by adv quintile — how large is the shift?

Usage:
  python analyze_delta_elbo.py \
    --before  /path/to/pi_n/model \
    --after   /path/to/pi_{n+1}/model \
    --data    /path/to/train_reward_a.jsonl \
    --output  /path/to/results.jsonl \
    --label   "Round_N_pi_n→pi_n+1" \
    --gpu     0 \
    --n_mc    4 \
    --max_samples  256

Output:
  - results.jsonl: per-sample (advantage, ΔELBO, reward, q_id, …)
  - stdout: summary statistics
"""
import argparse, json, os, sys, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# ── VRPO 路径 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2] / "dLLM_trainer" / "VRPO"
sys.path.insert(0, str(ROOT / "my_train"))

from my_dpo_trainer import (
    get_trainable_mask,
    blk_forward_process_with_trainable,
    make_basic_block_attention,
    mdm_ce_loss_with_trainable,
)

# ── ELBO 计算（复现 SDARGRPOTrainer._get_elbo_blk_with_trainable） ─────────────
@torch.no_grad()
def compute_elbo(
    model,
    full_ids: torch.Tensor,       # [seq_len]
    full_trainable: torch.Tensor, # [seq_len] bool
    prompt_len: int,
    completion_len: int,
    block_length: int = 64,
    n_mc: int = 4,
    mask_token_id: int = 126336,
    device: str = "cuda",
) -> float:
    """
    Returns normalized ELBO (per policy token), matching trainer with normalize_elbo=True.
    Uses block-diffusion MC estimation.
    """
    n_policy = full_trainable.sum().float().clamp(min=1)

    model.eval()
    total_elbo = 0.0

    seq_len = full_ids.shape[0]
    rng = torch.Generator()

    for mc_i in range(n_mc):
        seed = random.randint(0, 2**31)
        rng.manual_seed(seed)

        noisy = full_ids.clone()
        # Block-mask on completion region only (positions prompt_len : seq_len)
        comp_trainable = full_trainable[prompt_len:]
        trainable_indices = comp_trainable.nonzero(as_tuple=False).squeeze(-1)
        if len(trainable_indices) == 0:
            continue
        # Randomly mask half of them (approximate single-block behaviour)
        perm = torch.randperm(len(trainable_indices), generator=rng)
        k = max(1, len(trainable_indices) // 2)
        mask_idx = trainable_indices[perm[:k]] + prompt_len
        noisy[mask_idx] = mask_token_id

        # Forward
        inp = noisy.unsqueeze(0).to(device)
        clean = full_ids.unsqueeze(0).to(device)
        try:
            logits = model(input_ids=inp).logits  # [1, seq, vocab]
        except Exception as e:
            print(f"  [WARN] forward failed: {e}")
            continue

        # CE loss only on masked positions that are policy tokens
        mask_pos = (noisy != full_ids).to(device)  # [seq]
        valid_pos = mask_pos & full_trainable.to(device)  # [seq]
        if valid_pos.sum() == 0:
            continue

        logits_flat = logits[0][valid_pos]           # [N, vocab]
        targets_flat = clean[0][valid_pos]            # [N]
        ce = F.cross_entropy(logits_flat, targets_flat)
        elbo_mc = -ce.item() * valid_pos.sum().item() / k  # unnorm ELBO

        total_elbo += elbo_mc

    raw_elbo = total_elbo / n_mc
    return raw_elbo / n_policy.item()  # normalize per policy token


# Tokenizer fallback：部分训练输出目录无 tokenizer 文件，用任一有 tokenizer 的模型目录
_TOKENIZER_FALLBACK = str(
    Path(__file__).resolve().parents[1]
    / "Agentic/output/onpolicy_loop/pi5/model"
)


def _load_tokenizer(path: str):
    """先尝试从 path 加载，失败则 fallback 到 pi5/model。"""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"  [WARN] {path} 无 tokenizer（{e}），使用 fallback: {_TOKENIZER_FALLBACK}")
        return AutoTokenizer.from_pretrained(_TOKENIZER_FALLBACK, trust_remote_code=True)


def load_model(path: str, device: str):
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, PreTrainedModel

    print(f"  加载模型: {path}")
    tok = _load_tokenizer(path)

    # ── Patch 1: all_tied_weights_keys ─────────────────────────────────────
    # 新版 transformers 调用 model.all_tied_weights_keys.keys()，
    # 但旧版 LLaDA modeling_llada.py 未定义该属性 → 返回空 dict。
    _orig_getattr = nn.Module.__getattr__
    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _orig_getattr(self, name)
    nn.Module.__getattr__ = _patched_getattr

    # ── Patch 2: tie_weights kwargs ─────────────────────────────────────────
    # transformers 5.x 的 _finalize_model_loading 是 staticmethod，
    # 调用 model.tie_weights(missing_keys=..., recompute_mapping=False)，
    # 但旧版 LLaDA 的 tie_weights() 不接受这些 kwargs → wrap 容错版本。
    _orig_finalize = PreTrainedModel._finalize_model_loading  # staticmethod → 直接是函数

    def _safe_finalize(model, load_config, loading_info):
        orig_tw = model.__class__.tie_weights
        def wrapped_tw(self, *args, **kwargs):
            try:
                return orig_tw(self, *args, **kwargs)
            except TypeError:
                return orig_tw(self)
        model.__class__.tie_weights = wrapped_tw
        try:
            return _orig_finalize(model, load_config, loading_info)
        finally:
            model.__class__.tie_weights = orig_tw

    PreTrainedModel._finalize_model_loading = staticmethod(_safe_finalize)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device).eval()
    finally:
        # 还原两个 patch
        nn.Module.__getattr__ = _orig_getattr
        PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize)

    # ── Patch 3: config 兼容性 ───────────────────────────────────────────────
    # 旧版 LLaDA config 缺少新版 transformers 期望的属性，直接补全。
    cfg = model.config
    if not hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    if not hasattr(cfg, "output_attentions"):
        cfg.output_attentions = False
    if not hasattr(cfg, "output_hidden_states"):
        cfg.output_hidden_states = False

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before",      required=True,  help="π_n 模型路径（训练前）")
    parser.add_argument("--after",       required=True,  help="π_{n+1} 模型路径（训练后）")
    parser.add_argument("--data",        required=True,  help="train_reward_a.jsonl")
    parser.add_argument("--output",      required=True,  help="输出 results.jsonl")
    parser.add_argument("--label",       default="round_N", help="标注标签，用于区分实验")
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--n_mc",        type=int, default=4,   help="MC 采样数")
    parser.add_argument("--max_samples", type=int, default=256, help="分析样本数")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # ── 读取训练数据 ───────────────────────────────────────────────────────────
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    random.shuffle(rows)
    rows = rows[:args.max_samples]
    print(f"分析 {len(rows)} 条 rollout（来自 {args.data}）")

    # ── 顺序加载策略：先算所有 ELBO_before，卸载，再算所有 ELBO_after ─────────
    # 这样峰值显存只有 ~16GB（单模型），可以和 eval 进程共享同一张 A100

    # 先tokenize 所有样本（不需要 GPU）
    print("== 加载 tokenizer ==")
    tokenizer = _load_tokenizer(args.before)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or 126336

    samples = []
    print(f"tokenize {len(rows)} 条样本...")
    for row in rows:
        p_ids = tokenizer(row["prompt"],     add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [tokenizer.eos_token_id]
        # get_trainable_mask 返回长度 = len(c_ids)-1（不含 eos），
        # 需要补上 eos 对应的 True，保证 full_mask 与 full_ids 等长。
        tm    = list(get_trainable_mask(row["completion"], tokenizer)) + [True]
        tm    = tm[:len(c_ids)]           # 恰好 len(c_ids) 个元素
        full_ids  = torch.tensor(p_ids + c_ids, dtype=torch.long)
        full_mask = torch.tensor([False]*len(p_ids) + tm, dtype=torch.bool)
        assert full_ids.shape == full_mask.shape, \
            f"shape mismatch: full_ids={full_ids.shape}, full_mask={full_mask.shape}"
        samples.append({
            "full_ids":    full_ids,
            "full_mask":   full_mask,
            "prompt_len":  len(p_ids),
            "comp_len":    len(c_ids),
            "advantage":   row["advantage"],
            "reward":      row.get("reward", float("nan")),
            "question_id": row.get("question_id", "?"),
            "rollout_id":  row.get("rollout_id", -1),
        })

    # Phase A: 计算所有 ELBO_before
    print("\n== 加载 π_before ==")
    model_before, _ = load_model(args.before, device)
    elbos_before = []
    for i, s in enumerate(samples):
        eb = compute_elbo(model_before, s["full_ids"], s["full_mask"],
                          s["prompt_len"], s["comp_len"],
                          n_mc=args.n_mc, mask_token_id=mask_token_id, device=device)
        elbos_before.append(eb)
        if (i+1) % 32 == 0:
            print(f"  ELBO_before {i+1}/{len(samples)}")

    # 卸载 π_before，释放显存
    print("  卸载 π_before...")
    del model_before
    torch.cuda.empty_cache()

    # Phase B: 计算所有 ELBO_after
    print("\n== 加载 π_after ==")
    model_after, _ = load_model(args.after, device)
    elbos_after = []
    for i, s in enumerate(samples):
        ea = compute_elbo(model_after, s["full_ids"], s["full_mask"],
                          s["prompt_len"], s["comp_len"],
                          n_mc=args.n_mc, mask_token_id=mask_token_id, device=device)
        elbos_after.append(ea)
        if (i+1) % 32 == 0:
            print(f"  ELBO_after  {i+1}/{len(samples)}")

    del model_after
    torch.cuda.empty_cache()

    # 合并结果
    results = []
    for i, s in enumerate(samples):
        delta_elbo = elbos_after[i] - elbos_before[i]
        results.append({
            "label":        args.label,
            "question_id":  s["question_id"],
            "rollout_id":   s["rollout_id"],
            "advantage":    float(s["advantage"]),
            "reward":       float(s["reward"]),
            "elbo_before":  float(elbos_before[i]),
            "elbo_after":   float(elbos_after[i]),
            "delta_elbo":   float(delta_elbo),
            "n_policy":     int(s["full_mask"].sum().item()),
            "prompt_len":   int(s["prompt_len"]),
        })
        if (i+1) % 64 == 0:
            advs   = [r["advantage"]  for r in results]
            deltas = [r["delta_elbo"] for r in results]
            rho, _ = spearmanr(advs, deltas)
            sign_ok = np.mean([np.sign(a)==np.sign(d) for a,d in zip(advs,deltas) if a!=0 and d!=0])
            print(f"  [{i+1}/{len(results)}]  ρ={rho:.4f}  sign-align={sign_ok*100:.1f}%")

    # ── 保存 ────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n保存 {len(results)} 条 → {args.output}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    advs   = np.array([r["advantage"]  for r in results])
    deltas = np.array([r["delta_elbo"] for r in results])
    rho, pval = spearmanr(advs, deltas)

    sign_ok = np.mean(
        (np.sign(advs) == np.sign(deltas)) & (advs != 0) & (deltas != 0)
    )
    # By advantage quintile
    qs = np.percentile(advs, [0, 20, 40, 60, 80, 100])
    print(f"\n{'='*55}")
    print(f"  Label : {args.label}")
    print(f"  N     : {len(results)}")
    print(f"  Spearman(adv, ΔELBO) = {rho:.4f}  (p={pval:.4f})")
    print(f"  Sign-alignment rate  = {sign_ok*100:.1f}%")
    print(f"  mean ΔELBO           = {deltas.mean():.5f} ± {deltas.std():.5f}")
    print(f"  mean ΔELBO | adv>0   = {deltas[advs>0].mean():.5f}  (n={sum(advs>0)})")
    print(f"  mean ΔELBO | adv<0   = {deltas[advs<0].mean():.5f}  (n={sum(advs<0)})")
    print(f"\n  Advantage quintile breakdown:")
    for qi in range(5):
        mask = (advs >= qs[qi]) & (advs < qs[qi+1])
        if mask.sum() == 0: continue
        print(f"    Q{qi+1} [{qs[qi]:+.3f}, {qs[qi+1]:+.3f})  "
              f"n={mask.sum():3d}  mean_adv={advs[mask].mean():+.3f}  "
              f"mean_ΔELBO={deltas[mask].mean():+.5f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
