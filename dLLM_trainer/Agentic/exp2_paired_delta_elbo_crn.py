#!/usr/bin/env python3
"""
exp2_paired_delta_elbo_crn.py — Exp2: High-precision ΔELBO with Common Random Numbers (CRN)

CRN: pre-generate ALL corruption seeds before loading any model.
     Both before/after models see EXACTLY the same masks → noise cancels in Δ.
Runs n_mc up to n_mc_max=64, reports convergence at n_mc = 4, 16, 32, 64.
"""
import sys, os
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import argparse, json, random, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def load_tokenizer(path):
    from transformers import AutoTokenizer
    fb = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi5/model"
    try: return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except: return AutoTokenizer.from_pretrained(fb, trust_remote_code=True)


def load_model_gpu(path, device):
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, PreTrainedModel
    _og = nn.Module.__getattr__
    def _pg(self, name):
        if name == "all_tied_weights_keys": return {}
        return _og(self, name)
    nn.Module.__getattr__ = _pg
    _of = PreTrainedModel._finalize_model_loading
    def _sf(model, lc, li):
        orig = model.__class__.tie_weights
        def w(self, *a, **kw):
            try: return orig(self, *a, **kw)
            except TypeError: return orig(self)
        model.__class__.tie_weights = w
        try: return _of(model, lc, li)
        finally: model.__class__.tie_weights = orig
    PreTrainedModel._finalize_model_loading = staticmethod(_sf)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
        ).to(device).eval()
    finally:
        nn.Module.__getattr__ = _og
        PreTrainedModel._finalize_model_loading = staticmethod(_of)
    for attr in ("use_cache", "output_attentions", "output_hidden_states"):
        if not hasattr(model.config, attr): setattr(model.config, attr, False)
    return model


@torch.no_grad()
def compute_elbo_crn(model, full_ids, full_mask, prompt_len, seeds, mask_token_id, device,
                     batch_size=8):
    """Compute per-MC ELBO using CRN seeds, with batched forward passes.
    Returns list of len(seeds) ELBO values normalized per policy token."""
    n_policy = full_mask.sum().float().clamp(min=1).item()
    t_idx = full_mask[prompt_len:].nonzero(as_tuple=False).squeeze(-1)
    if len(t_idx) == 0:
        return [0.0] * len(seeds)

    rng = torch.Generator()
    clean_dev = full_ids.to(device)
    elbos = []

    for b_start in range(0, len(seeds), batch_size):
        b_seeds = seeds[b_start : b_start + batch_size]
        noisy_list, valid_list, k_list = [], [], []
        for seed in b_seeds:
            rng.manual_seed(seed)
            noisy = full_ids.clone()
            perm = torch.randperm(len(t_idx), generator=rng)
            k = max(1, len(t_idx) // 2)
            noisy[t_idx[perm[:k]] + prompt_len] = mask_token_id
            noisy_list.append(noisy)
            valid_list.append((noisy != full_ids) & full_mask)
            k_list.append(k)

        batch_inp = torch.stack(noisy_list, dim=0).to(device)  # [B, seq]
        try:
            batch_logits = model(input_ids=batch_inp).logits    # [B, seq, vocab]
        except Exception as e:
            print(f"  [WARN] batch forward error: {e}")
            elbos.extend([0.0] * len(b_seeds)); continue

        for j in range(len(b_seeds)):
            valid_d = valid_list[j].to(device)
            if valid_d.sum() == 0:
                elbos.append(0.0); continue
            ce = F.cross_entropy(
                batch_logits[j][valid_d],
                clean_dev[valid_d],
            )
            elbos.append((-ce.item() * valid_d.sum().item() / k_list[j]) / n_policy)

    return elbos


def print_nmc_table(results, n_mc_list, label):
    advs = np.array([r["adv"] for r in results])
    print(f"\n{'='*75}")
    print(f"  {label}  N={len(results)}")
    print(f"  {'n_mc':>5}  {'ρ':>8}  {'p':>8}  {'sign%':>7}  {'Δ|A>0':>12}  {'Δ|A<0':>12}  {'meanΔ':>12}  {'stdΔ':>12}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    rows = []
    for nmc in n_mc_list:
        deltas = np.array([
            np.mean(r["mc_after"][:nmc]) - np.mean(r["mc_before"][:nmc])
            for r in results
        ])
        rho, pval = spearmanr(advs, deltas)
        sign = np.mean((np.sign(advs) == np.sign(deltas)) & (advs != 0) & (deltas != 0))
        dp = deltas[advs > 0].mean() if (advs > 0).any() else float("nan")
        dn = deltas[advs < 0].mean() if (advs < 0).any() else float("nan")
        dm, ds = deltas.mean(), deltas.std()
        print(f"  {nmc:>5}  {rho:>8.4f}  {pval:>8.4f}  {sign*100:>6.1f}%  "
              f"{dp:>12.6f}  {dn:>12.6f}  {dm:>12.6f}  {ds:>12.6f}")
        rows.append({"n_mc": nmc, "rho": rho, "pval": pval, "sign_pct": sign*100,
                     "delta_pos": float(dp), "delta_neg": float(dn),
                     "mean_delta": float(dm), "std_delta": float(ds)})
    print(f"{'='*75}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before",      required=True)
    parser.add_argument("--after",       required=True)
    parser.add_argument("--data",        required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--label",       default="")
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--n_mc_max",    type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda:0"

    print(f"\nCRN ΔELBO | {args.label}")
    print(f"  before: {Path(args.before).name}  after: {Path(args.after).name}")
    print(f"  n_mc_max={args.n_mc_max}  max_samples={args.max_samples}")

    # Load data
    ROOT = Path(__file__).resolve().parents[2] / "dLLM_trainer" / "VRPO"
    sys.path.insert(0, str(ROOT / "my_train"))
    from my_dpo_trainer import get_trainable_mask

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    random.shuffle(rows); rows = rows[:args.max_samples]

    tok = load_tokenizer(args.before)
    mask_id = getattr(tok, "mask_token_id", None) or 126336

    # ── Pre-generate ALL seeds (CRN key step) ───────────────────────────────
    print("预生成 CRN seeds...")
    all_seeds = [[random.randint(0, 2**31) for _ in range(args.n_mc_max)] for _ in rows]

    # Tokenize
    samples = []
    for i, row in enumerate(rows):
        p_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(row["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        tm = list(get_trainable_mask(row["completion"], tok)) + [True]
        tm = tm[:len(c_ids)]
        full_ids  = torch.tensor(p_ids + c_ids, dtype=torch.long)
        full_mask = torch.tensor([False]*len(p_ids) + tm, dtype=torch.bool)
        samples.append({
            "full_ids": full_ids, "full_mask": full_mask,
            "prompt_len": len(p_ids), "adv": row["advantage"],
            "seeds": all_seeds[i],
        })
    print(f"tokenized {len(samples)} samples")

    n_mc_list = [v for v in [4, 16, 32, 64] if v <= args.n_mc_max]

    # ── Phase A: ELBO_before ─────────────────────────────────────────────────
    print(f"\n== π_before: {Path(args.before).name}")
    model = load_model_gpu(args.before, device)
    mc_before = []
    for i, s in enumerate(samples):
        mc_before.append(compute_elbo_crn(
            model, s["full_ids"], s["full_mask"], s["prompt_len"],
            s["seeds"], mask_id, device
        ))
        if (i + 1) % 64 == 0:
            print(f"  before {i+1}/{len(samples)}")
    del model; torch.cuda.empty_cache()
    print("  π_before 完成，已卸载")

    # ── Phase B: ELBO_after ──────────────────────────────────────────────────
    print(f"\n== π_after: {Path(args.after).name}")
    model = load_model_gpu(args.after, device)
    mc_after = []
    for i, s in enumerate(samples):
        mc_after.append(compute_elbo_crn(
            model, s["full_ids"], s["full_mask"], s["prompt_len"],
            s["seeds"], mask_id, device
        ))
        if (i + 1) % 64 == 0:
            print(f"  after {i+1}/{len(samples)}")
    del model; torch.cuda.empty_cache()

    # Compile
    results = [
        {"adv": float(s["adv"]), "mc_before": mc_before[i], "mc_after": mc_after[i]}
        for i, s in enumerate(samples)
    ]

    # Report convergence table
    summary = print_nmc_table(results, n_mc_list, args.label)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "label": args.label, "before": args.before, "after": args.after,
            "n_mc_max": args.n_mc_max, "n_samples": len(results),
            "summary": summary,
        }, f, indent=2)
    print(f"\n保存 → {args.output}")


if __name__ == "__main__":
    main()
