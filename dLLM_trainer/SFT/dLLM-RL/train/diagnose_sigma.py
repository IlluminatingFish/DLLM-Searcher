"""
Standalone diagnostic: does cont_head's eps prediction actually respond to sigma?

Loads the checkpoint saved after the 1-epoch test run (sft_llada_reasoning_test/ckpt_0)
on a single GPU (no DeepSpeed needed for inference), takes one real training sample
with a <think> span, and runs two forward passes with the SAME eps but DIFFERENT
sigma (0.1 vs 1.0). If sigma conditioning is working, eps_pred should differ
meaningfully between the two cases. If eps_pred is (almost) identical regardless
of sigma, the sigma signal is not reaching cont_head in a useful way.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT_DIR = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_llada_reasoning_test/ckpt_0/optimized"
DATA_PATH = "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/data/sft_train_final.json"

device = "cuda:0"

print("Loading tokenizer + model from checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(CKPT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
model.eval()

d_model = model.config.d_model if hasattr(model.config, "d_model") else model.config.hidden_size

cont_head = torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16).to(device)
cont_head.load_state_dict(torch.load(os.path.join(CKPT_DIR, "cont_head.pt"), map_location=device))
cont_head.eval()

sigma_proj = torch.nn.Sequential(
    torch.nn.Linear(1, d_model),
    torch.nn.SiLU(),
    torch.nn.Linear(d_model, d_model),
).to(dtype=torch.bfloat16, device=device)
sigma_proj.load_state_dict(torch.load(os.path.join(CKPT_DIR, "sigma_proj.pt"), map_location=device))
sigma_proj.eval()

# Check sigma_proj actually moved away from zero-init
w0 = sigma_proj[0].weight
w2 = sigma_proj[2].weight
b2 = sigma_proj[2].bias
print(f"\nsigma_proj[0].weight norm: {w0.float().norm().item():.6f}")
print(f"sigma_proj[2].weight norm: {w2.float().norm().item():.6f}  (started at 0.0)")
print(f"sigma_proj[2].bias   norm: {b2.float().norm().item():.6f}  (started at 0.0)")

# Probe sigma_proj output magnitude directly for a few sigma values
for sv in [0.1, 0.5, 1.0]:
    sig_in = torch.tensor([[sv]], dtype=torch.bfloat16, device=device)
    out = sigma_proj(sig_in)
    print(f"  sigma={sv}: ||sigma_proj(sigma)|| = {out.float().norm().item():.6f}")

# ── Load one real sample with a <think> span ──────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

def find_think_span(input_ids):
    seq = input_ids.tolist()
    slen, elen = len(think_start_ids), len(think_end_ids)
    for i in range(len(seq) - slen):
        if seq[i:i+slen] == think_start_ids:
            for j in range(i+slen, len(seq) - elen):
                if seq[j:j+elen] == think_end_ids:
                    return i, j + elen
    return None

sample = None
for item in data[:50]:
    full_text = item["prompt"] + item["response"]
    ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"][0]
    span = find_think_span(ids)
    if span is not None and span[1] - span[0] > 10:
        sample = (ids, span)
        break

assert sample is not None, "couldn't find a sample with a think span in first 50 examples"
input_ids, (t_start, t_end) = sample
print(f"\nUsing sample with think span [{t_start}:{t_end}] (length {t_end-t_start}), total length {len(input_ids)}")

input_ids = input_ids.unsqueeze(0).to(device)  # (1, L)
L = input_ids.shape[1]
think_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
think_mask[0, t_start:t_end] = True

embed_layer = model.model.transformer.wte
final_norm_layer = model.model.transformer.ln_f

torch.manual_seed(0)
eps = torch.randn(1, L, d_model, dtype=torch.bfloat16, device=device)

def run(actual_sigma, told_sigma):
    """Inject noise at level `actual_sigma`, but feed sigma_proj `told_sigma`.
    If conditioning actually influences the prediction, varying told_sigma
    while holding actual_sigma fixed should change eps_pred."""
    sig_actual = torch.tensor([actual_sigma], dtype=torch.bfloat16, device=device)
    sig_told   = torch.tensor([told_sigma],   dtype=torch.bfloat16, device=device)
    final_hidden = {}

    def embed_hook(module, args, output):
        sig = sig_actual.to(output.dtype).view(1, 1, 1)
        sig_emb = sigma_proj(sig_told.view(1, 1).to(output.dtype)).view(1, 1, d_model)
        injected = output + sig * eps + sig_emb
        out = output.clone()
        out[think_mask] = injected[think_mask]
        return out

    def capture_hook(module, args, output):
        final_hidden["h"] = output

    h1 = embed_layer.register_forward_hook(embed_hook)
    h2 = final_norm_layer.register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        h1.remove()
        h2.remove()

    hidden_last = final_hidden["h"]
    eps_pred = cont_head(hidden_last)
    return hidden_last[think_mask].float(), eps_pred[think_mask].float()

eps_at_think = eps[0, t_start:t_end].float()

print("\n=== Test A: does eps_pred change with sigma_proj's input, holding actual noise fixed? ===")
print("(actual_sigma=1.0 in both; told_sigma varies 0.1 vs 1.0)")
hidden_A1, pred_A1 = run(actual_sigma=1.0, told_sigma=0.1)
hidden_A2, pred_A2 = run(actual_sigma=1.0, told_sigma=1.0)
diff_hidden_A = (hidden_A2 - hidden_A1).norm(dim=-1).mean().item()
diff_pred_A   = (pred_A2   - pred_A1).norm(dim=-1).mean().item()
print(f"||hidden_last(told=1.0) - hidden_last(told=0.1)|| (avg): {diff_hidden_A:.6f}")
print(f"||eps_pred(told=1.0)   - eps_pred(told=0.1)||     (avg): {diff_pred_A:.6f}")
print(f"hidden_last typical norm (told=1.0 case):                {hidden_A2.norm(dim=-1).mean().item():.4f}")
print("  -> if diff_hidden_A is tiny relative to typical norm, sigma_proj's signal")
print("     is being washed out somewhere between the embedding layer and ln_f.")

print("\n=== Test B: does eps_pred track the TRUE injected noise level (real scenario)? ===")
print("(actual_sigma and told_sigma both correct: 0.1 vs 1.0)")
hidden_01, pred_01 = run(actual_sigma=0.1, told_sigma=0.1)
hidden_10, pred_10 = run(actual_sigma=1.0, told_sigma=1.0)
mse_01 = F.mse_loss(pred_01, eps_at_think).item()
mse_10 = F.mse_loss(pred_10, eps_at_think).item()
mse_zero = F.mse_loss(torch.zeros_like(eps_at_think), eps_at_think).item()
print(f"true eps norm (per-token avg):            {eps_at_think.norm(dim=-1).mean().item():.4f}")
print(f"eps_pred norm @ sigma=0.1 (per-tok avg):   {pred_01.norm(dim=-1).mean().item():.4f}")
print(f"eps_pred norm @ sigma=1.0 (per-tok avg):   {pred_10.norm(dim=-1).mean().item():.4f}")
print(f"MSE(eps_pred, true_eps) @ sigma=0.1:       {mse_01:.4f}")
print(f"MSE(eps_pred, true_eps) @ sigma=1.0:       {mse_10:.4f}")
print(f"MSE(zero_pred, true_eps) [trivial baseline]: {mse_zero:.4f}  <- this is what we saw flatlined to in training")
