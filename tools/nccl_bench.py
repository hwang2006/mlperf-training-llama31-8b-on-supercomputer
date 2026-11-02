#!/usr/bin/env python3
"""All-reduce bus bandwidth with torch.distributed (1 process per GPU).
Rank/world come from Slurm (srun --ntasks-per-node=4); MASTER_ADDR/PORT from the sbatch."""
import os
import socket
import time

import torch
import torch.distributed as dist

rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
local = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local)
cpus = sorted(os.sched_getaffinity(0))
print(f"AFFINITY rank={rank} host={socket.gethostname()} gpu={local} ncpus={len(cpus)} "
      f"cpus={cpus[0]}-{cpus[-1]}", flush=True)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local))
if rank == 0:
    print(f"world={world} NCCL={torch.cuda.nccl.version()} plugin={os.environ.get('NCCL_NET_PLUGIN')}", flush=True)
for mb in (8, 64, 512, 2048):
    n = mb * 2**20 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    iters = 20
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    busbw = (mb * 2**20) / dt * 2 * (world - 1) / world / 1e9
    if rank == 0:
        print(f"allreduce {mb:5d} MB  time {dt*1e3:8.2f} ms  busbw {busbw:7.1f} GB/s", flush=True)
dist.destroy_process_group()
