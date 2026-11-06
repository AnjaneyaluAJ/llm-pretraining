"""Toy DDP overlap test: prints compute-only, all-reduce-only and DDP step time. DDP step ~= sum means no overlap.
Run: [HOOK=allreduce|bf16] torchrun --standalone --nproc_per_node=2 scripts/repro/ddp_overlap_toy.py"""
import torch,torch.distributed as dist,time,torch.nn as nn,os
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("nccl"); r=dist.get_rank(); torch.cuda.set_device(r)
m=nn.Sequential(*[nn.Linear(2048,2048) for _ in range(24)]).cuda()
d=DDP(m,device_ids=[r],bucket_cap_mb=int(os.environ.get("BKT","25")))
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as H
hk=os.environ.get("HOOK")
if hk=="allreduce": d.register_comm_hook(None,H.allreduce_hook)
def nothen(state,bucket):
    t=bucket.buffer(); t.div_(dist.get_world_size())
    return dist.all_reduce(t,async_op=True).get_future()
if hk=="nothen": d.register_comm_hook(None,nothen)
if hk=="bf16": d.register_comm_hook(None,H.bf16_compress_hook)
x=torch.randn(8192,2048,device="cuda")
def step(sync):
    with torch.autocast("cuda",torch.bfloat16):
        if sync: d(x).sum().backward()
        else:
            with d.no_sync(): d(x).sum().backward()
def tm(f,n=5):
    for _ in range(2): f()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n*1e3
g=torch.randn(sum(p.numel() for p in m.parameters()),device="cuda")
a=tm(lambda:step(False)); b=tm(lambda:step(True)); c=tm(lambda:dist.all_reduce(g))
if r==0: print(f"compute-only {a:.0f} ms | allreduce-only {c:.0f} ms | DDP step {b:.0f} ms | sum {a+c:.0f} | perfect-overlap {max(a,c):.0f}")
dist.destroy_process_group()
