"""Measure what the hardware can actually do, as the reference for MFU and for the
communication model of each parallel strategy.

    python scripts/bench_hw.py matmul                      # every GPU, one at a time
    torchrun --standalone --nproc_per_node=2 scripts/bench_hw.py comm
    torchrun --standalone --nproc_per_node=4 scripts/bench_hw.py comm

matmul: dense square matmuls in fp32, TF32 and bf16; best TFLOP/s over sizes.
comm:   NCCL all_reduce, all_gather, reduce_scatter and send/recv over message sizes.
        Bus bandwidth uses the nccl-tests convention (all_reduce: algbw * 2(n-1)/n,
        all_gather / reduce_scatter: algbw * (n-1)/n), so it is comparable across GPU counts.
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def matmul():
    out = []
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu)
        res = {"gpu": gpu, "name": torch.cuda.get_device_name(gpu)}
        for label, dtype, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True),
                                   ("bf16", torch.bfloat16, False)]:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            best = 0.0
            for n in (2048, 4096, 8192):
                a = torch.randn(n, n, device="cuda", dtype=dtype)
                b = torch.randn(n, n, device="cuda", dtype=dtype)
                dt = timeit(lambda: a @ b, 20 if n < 8192 else 10)
                best = max(best, 2 * n ** 3 / dt / 1e12)
            res[label + "_tflops"] = round(best, 1)
        print(json.dumps(res))
        out.append(res)
    return out


def comm():
    rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    rows = []
    for mb in (1, 8, 64, 256, 512):
        n = mb * 2 ** 20 // 4                       # fp32 elements
        x = torch.ones(n, device="cuda")
        shard = torch.ones(n // ws, device="cuda")
        out_full = torch.empty(n, device="cuda")
        r = {"world": ws, "mb": mb}
        dt = timeit(lambda: dist.all_reduce(x), 10)
        r["all_reduce_busbw_gbs"] = round(x.numel() * 4 / dt / 1e9 * 2 * (ws - 1) / ws, 2)
        dt = timeit(lambda: dist.all_gather_into_tensor(out_full, shard), 10)
        r["all_gather_busbw_gbs"] = round(n * 4 / dt / 1e9 * (ws - 1) / ws, 2)
        dt = timeit(lambda: dist.reduce_scatter_tensor(shard, x), 10)
        r["reduce_scatter_busbw_gbs"] = round(n * 4 / dt / 1e9 * (ws - 1) / ws, 2)
        if ws == 2:                                  # point-to-point, used by pipeline parallel
            def p2p():
                if rank == 0:
                    dist.send(x, 1)
                else:
                    dist.recv(x, 0)
            dt = timeit(p2p, 10)
            r["send_recv_gbs"] = round(x.numel() * 4 / dt / 1e9, 2)
        if rank == 0:
            print(json.dumps(r))
        rows.append(r)
    if rank == 0:
        print("nccl", ".".join(map(str, torch.cuda.nccl.version())))
    dist.destroy_process_group()


if __name__ == "__main__":
    {"matmul": matmul, "comm": comm}[sys.argv[1]]()
