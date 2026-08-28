#!/usr/bin/env python3
"""
exp1_policy_distances.py — Experiment 1: Policy Distance Analysis
  1A: Parameter L2 distance (CPU, state dict)
  1B: Fixed masked-state KL divergence (GPU, both models simultaneously)
"""
import sys, os
# Must set CUDA device BEFORE any torch imports
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import argparse, json, random, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


# ── State dict loader (CPU only) ──────────────────────────────────────────────
def load_state_dict_cpu(model_path: str) -> dict:
    path = Path(model_path)
    from safetensors.torch import load_file

    sf = path / "model.safetensors"
    if sf.exists():
        print(f"    单文件: {sf.name}")
        return load_file(str(sf), device="cpu")

    idx = path / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {}
        for key, sname in weight_map.items():
            if sname not in shards:
                print(f"    分片: {sname}")
                shards[sname] = load_file(str(path / sname), device="cpu")
        return {key: shards[sf][key] for key, sf in weight_map.items()}

    raise FileNotFoundError(f"No safetensors in {model_path}")


def get_layer_group(key: str) -> str:
    parts = key.split(".")
    # LLaMA: model.layers.N.xxx  OR  transformer.blocks.N.xxx
    for i, p in enumerate(parts):
        if p in ("layers", "blocks") and i + 1 < len(parts) and parts[i+1].isdigit():
            return f"layer_{int(parts[i+1]):02d}"
    k = key.lower()
    if any(x in k for x in ("embed_tokens", "wte", "embedding")) and "layer" not in k and "block" not in k:
        return "embed"
    if "lm_head" in k or "embed_out" in k:
        return "lm_head"
    return "other"


# ── 1A: Parameter distances ───────────────────────────────────────────────────
def exp1a(sd1: dict, sd2: dict) -> dict:
    print("  [1A] 计算参数距离...")
    groups: dict = {}
    g_sq_d = g_sq_n = 0.0
    g_n = 0
    g_max = 0.0

    for key in sd1:
        if key not in sd2:
            continue
        p1 = sd1[key].float()
        p2 = sd2[key].float()
        d = p2 - p1
        sq_d = (d ** 2).sum().item()
        sq_n = (p1 ** 2).sum().item()
        n = p1.numel()
        ma = d.abs().max().item()

        grp = get_layer_group(key)
        if grp not in groups:
            groups[grp] = [0.0, 0.0, 0, 0.0]
        groups[grp][0] += sq_d
        groups[grp][1] += sq_n
        groups[grp][2] += n
        groups[grp][3] = max(groups[grp][3], ma)

        g_sq_d += sq_d; g_sq_n += sq_n; g_n += n; g_max = max(g_max, ma)

    def fmt(sq_d, sq_n, n, ma):
        return {
            "rel_l2": round((sq_d**0.5) / max(sq_n**0.5, 1e-12), 8),
            "rms":    round((sq_d / max(n, 1))**0.5, 10),
            "max_abs": round(ma, 8),
            "n_params_M": round(n / 1e6, 2),
        }

    return {
        "global": fmt(g_sq_d, g_sq_n, g_n, g_max),
        "layer_wise": {g: fmt(*v) for g, v in sorted(groups.items())},
    }


# ── GPU model loader (same patches as analyze_delta_elbo.py) ─────────────────
def load_model_gpu(path: str, device: str):
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
            path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager"
        ).to(device).eval()
    finally:
        nn.Module.__getattr__ = _og
        PreTrainedModel._finalize_model_loading = staticmethod(_of)

    for attr in ("use_cache", "output_attentions", "output_hidden_states"):
        if not hasattr(model.config, attr): setattr(model.config, attr, False)
    return model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    fb = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi5/model"
    try: return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except: return AutoTokenizer.from_pretrained(fb, trust_remote_code=True)


# ── 1B: Fixed masked-state KL ─────────────────────────────────────────────────
@torch.no_grad()
def exp1b(model1, model2, samples, mask_token_id=126336, t=0.5, seed=42, device="cuda:0"):
    """Both models receive identical corrupted input; measure KL between outputs."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    kl_12, kl_21 = [], []

    for i, (full_ids, full_mask, prompt_len) in enumerate(samples):
        comp_mask = full_mask[prompt_len:]
        t_idx = comp_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(t_idx) == 0:
            continue

        noisy = full_ids.clone()
        perm = torch.randperm(len(t_idx), generator=rng)
        k = max(1, int(len(t_idx) * t))
        noisy[t_idx[perm[:k]] + prompt_len] = mask_token_id

        valid = (noisy != full_ids) & full_mask  # [seq]
        if valid.sum() == 0:
            continue

        inp = noisy.unsqueeze(0).to(device)
        valid_d = valid.to(device)

        lg1 = model1(input_ids=inp).logits[0][valid_d]  # [N, vocab]
        lg2 = model2(input_ids=inp).logits[0][valid_d]

        lp1 = F.log_softmax(lg1.float(), dim=-1)
        lp2 = F.log_softmax(lg2.float(), dim=-1)
        p1, p2 = lp1.exp(), lp2.exp()

        kl_12.append((p1 * (lp1 - lp2)).sum(-1).mean().item())
        kl_21.append((p2 * (lp2 - lp1)).sum(-1).mean().item())

        if (i + 1) % 32 == 0:
            print(f"  [1B] [{i+1}/{len(samples)}] KL(1‖2)={np.mean(kl_12):.6f}  KL(2‖1)={np.mean(kl_21):.6f}")

    kl12 = float(np.mean(kl_12)) if kl_12 else 0.0
    kl21 = float(np.mean(kl_21)) if kl_21 else 0.0
    return {"kl_12": kl12, "kl_21": kl21, "kl_sym": (kl12 + kl21) / 2, "n": len(kl_12)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1",    required=True)
    parser.add_argument("--model2",    required=True)
    parser.add_argument("--data",      required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--label",     default="")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--skip_1a",   action="store_true")
    parser.add_argument("--skip_1b",   action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda:0"
    out = {"label": args.label, "model1": args.model1, "model2": args.model2}

    print(f"\n{'='*60}")
    print(f"Policy Distance | {args.label}")
    print(f"  model1: {Path(args.model1).name}")
    print(f"  model2: {Path(args.model2).name}")
    print(f"{'='*60}")

    # ── 1A ───────────────────────────────────────────────────────────────────
    if not args.skip_1a:
        print("\n── 1A: Parameter L2 distance ──")
        sd1 = load_state_dict_cpu(args.model1)
        sd2 = load_state_dict_cpu(args.model2)
        r1a = exp1a(sd1, sd2)
        out["exp1a"] = r1a
        g = r1a["global"]
        print(f"  Global  rel_L2={g['rel_l2']:.6f}  RMS={g['rms']:.2e}  max_abs={g['max_abs']:.4e}  N={g['n_params_M']:.1f}M")
        print("  Layer-wise:")
        for ln, lv in sorted(r1a["layer_wise"].items()):
            print(f"    {ln:15s} rel_L2={lv['rel_l2']:.6f}  RMS={lv['rms']:.2e}")
        del sd1, sd2; gc.collect()

    # ── 1B ───────────────────────────────────────────────────────────────────
    if not args.skip_1b:
        print("\n── 1B: Fixed-state KL divergence ──")
        ROOT = Path(__file__).resolve().parents[2] / "dLLM_trainer" / "VRPO"
        sys.path.insert(0, str(ROOT / "my_train"))
        from my_dpo_trainer import get_trainable_mask

        tok = load_tokenizer(args.model1)
        mask_id = getattr(tok, "mask_token_id", None) or 126336

        rows = [json.loads(l) for l in open(args.data) if l.strip()]
        random.shuffle(rows); rows = rows[:args.n_samples]

        samples = []
        for row in rows:
            p_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tok(row["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            tm = list(get_trainable_mask(row["completion"], tok)) + [True]
            tm = tm[:len(c_ids)]
            full_ids  = torch.tensor(p_ids + c_ids, dtype=torch.long)
            full_mask = torch.tensor([False]*len(p_ids) + tm, dtype=torch.bool)
            samples.append((full_ids, full_mask, len(p_ids)))

        print(f"  加载 model1: {Path(args.model1).name}")
        m1 = load_model_gpu(args.model1, device)
        print(f"  加载 model2: {Path(args.model2).name}")
        m2 = load_model_gpu(args.model2, device)
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU mem: {mem:.1f}GB  samples={len(samples)}")

        r1b = exp1b(m1, m2, samples, mask_id, t=0.5, seed=args.seed, device=device)
        out["exp1b"] = r1b
        del m1, m2; torch.cuda.empty_cache()

        print(f"\n  KL(model1‖model2) = {r1b['kl_12']:.6f}")
        print(f"  KL(model2‖model1) = {r1b['kl_21']:.6f}")
        print(f"  KL symmetric      = {r1b['kl_sym']:.6f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存 → {args.output}")

    print(f"\n{'='*60}  SUMMARY  {args.label}")
    if "exp1a" in out:
        g = out["exp1a"]["global"]
        print(f"  1A  rel_L2={g['rel_l2']:.6f}  RMS={g['rms']:.2e}  max_abs={g['max_abs']:.4e}")
    if "exp1b" in out:
        b = out["exp1b"]
        print(f"  1B  KL(1‖2)={b['kl_12']:.6f}  KL(2‖1)={b['kl_21']:.6f}  sym={b['kl_sym']:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
