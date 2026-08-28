import torch
import torch.distributed as dist
dist.init_process_group('nccl')
rank = dist.get_rank()
t = torch.ones(3).cuda(rank)
dist.all_reduce(t)
print(f'rank {rank} all_reduce OK: {t}', flush=True)
dist.destroy_process_group()
