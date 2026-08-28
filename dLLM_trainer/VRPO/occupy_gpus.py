"""
Occupy GPU memory on specified devices to reserve them.
Allocates ~38 GB on each GPU (out of 40 GB) and sits in a loop.
"""
import torch
import time
import signal
import sys

GPUS = [0, 1, 2, 3]
GB_TO_OCCUPY = 38  # GB per GPU (leave ~2 GB headroom)

tensors = []

def cleanup(sig, frame):
    print("\nReleasing GPU memory and exiting...")
    del tensors[:]
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print(f"Occupying GPUs: {GPUS}")
for gpu_id in GPUS:
    device = torch.device(f"cuda:{gpu_id}")
    n_bytes = GB_TO_OCCUPY * 1024 ** 3
    n_floats = n_bytes // 4  # float32
    t = torch.zeros(n_floats, dtype=torch.float32, device=device)
    tensors.append(t)
    mem = torch.cuda.memory_allocated(device) / 1024**3
    print(f"  GPU {gpu_id}: allocated {mem:.1f} GB")

print(f"\nAll {len(GPUS)} GPUs occupied. Press Ctrl+C or send SIGTERM to release.")
while True:
    time.sleep(60)
    for gpu_id in GPUS:
        mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"  [heartbeat] GPU {gpu_id}: {mem:.1f} GB held", flush=True)
