"""Minimal repro: Future.then() on an NCCL all-reduce blocks the compute stream like .wait() (torch 2.15 + NCCL 2.30).
Run: torchrun --standalone --nproc_per_node=2 scripts/repro/then_blocks_overlap.py
On torch 2.8 + NCCL 2.27, async and async+then print about the same time; on 2.15 async+then equals async+wait."""
import torch,torch.distributed as dist,time
dist.init_process_group("nccl"); r=dist.get_rank(); torch.cuda.set_device(r)
a=torch.randn(8192,8192,device="cuda",dtype=torch.bfloat16); g=torch.randn(6_250_000,device="cuda")
def mm(n=2):
    for _ in range(n): a@a
def tm(f):
    f(); torch.cuda.synchronize(); t=time.time(); f(); torch.cuda.synchronize(); return (time.time()-t)*1e3
plain=tm(lambda:[(dist.all_reduce(g,async_op=True),mm()) for _ in range(10)])
then=tm(lambda:[(dist.all_reduce(g,async_op=True).get_future().then(lambda f:f.value()),mm()) for _ in range(10)])
wait=tm(lambda:[(dist.all_reduce(g,async_op=True).wait(),mm()) for _ in range(10)])
mmo=tm(lambda:[mm() for _ in range(10)]); aro=tm(lambda:[dist.all_reduce(g) for _ in range(10)])
if r==0: print(f"mm-only {mmo:.0f} ar-only {aro:.0f} | async {plain:.0f} | async+then {then:.0f} | async+wait {wait:.0f}")
