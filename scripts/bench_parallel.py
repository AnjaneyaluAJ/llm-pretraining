"""Compare multi-GPU training strategies on the same model, data and global batch.

    python scripts/bench_parallel.py --strategy single                     # 1-GPU reference
    torchrun --standalone --nproc_per_node=2 scripts/bench_parallel.py --strategy ddp
    torchrun --standalone --nproc_per_node=2 scripts/bench_parallel.py --strategy fsdp
    torchrun --standalone --nproc_per_node=2 scripts/bench_parallel.py --strategy tp
    torchrun --standalone --nproc_per_node=2 scripts/bench_parallel.py --strategy pp

Every strategy trains GPT-2 (bf16 autocast, SDPA attention, fused AdamW, per-block
torch.compile, grad clip 1.0) on real FineWeb-Edu batches with the same global batch per
optimizer step. What differs is how the work is split:

  single  one GPU, gradient accumulation to reach the global batch
  ddp     data parallel: full model copy per GPU, gradients all-reduced once per step
  fsdp    data parallel with parameters, gradients and optimizer state sharded (FSDP2);
          gradients are reduce-scattered once per step (no sync during accumulation)
  tp      tensor parallel: attention heads and MLP columns split across GPUs (Megatron);
          --tp_sp adds sequence parallelism for the layer norms and residual stream
  pp      pipeline parallel: blocks split into stages, the global batch split into
          --microbatches and run with 1F1B, GPipe, or interleaved 1F1B (--pp_virtual 2)

Tuning knobs: --ddp_tuned (bigger buckets, gradient bucket views), --fsdp_bf16, --fsdp_no_reshard, --tp_sp, --pp_split,
--pp_virtual, --act_ckpt (activation checkpointing per block).

Measurement: median step time after --skip warm-up steps (CUDA-synchronized). With
--profile, 3 extra steps are traced with torch.profiler and each rank's GPU time is split
into compute, communication hidden under compute, exposed communication, and idle.

Correctness: --check_grads DIR runs one forward/backward on a fixed batch from the same
initial weights and saves each rank's full gradients (gathered from shards) for
scripts/compare_grads.py, which compares every strategy with the single-GPU gradients.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from scripts/
from llmtrain.data import split_dataset, train_loader
from llmtrain.dist import cleanup_distributed, setup_distributed
from llmtrain.evaluation import lm_loss
from llmtrain.models import build_model, count_params, flops_per_token, resolve_model_config

STRATEGIES = ("single", "ddp", "fsdp", "tp", "pp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=STRATEGIES, required=True)
    p.add_argument("--preset", default="gpt2-124m")
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--data", default="/mnt/nvme/tok/fineweb-edu-300k")
    p.add_argument("--global_batch", type=int, default=32, help="sequences per optimizer step")
    p.add_argument("--micro_batch", type=int, default=16, help="sequences per forward (not pp)")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--act_ckpt", action="store_true", help="activation checkpointing per block")
    p.add_argument("--ddp_bf16_grads", action="store_true", help="DDP: bf16_compress_hook for the gradient all-reduce")
    p.add_argument("--ddp_tuned", action="store_true",
                   help="100 MB buckets + gradient bucket views (static_graph=True is left out: it trips an "
                        "internal assert together with no_sync() accumulation in torch 2.15)")
    p.add_argument("--fsdp_bf16", action="store_true", help="gather params in bf16, reduce grads in fp32")
    p.add_argument("--fsdp_no_reshard", action="store_true", help="keep params gathered after forward (ZeRO-2 like)")
    p.add_argument("--tp_shared_input", action="store_true",
                   help="tp: pass each block input to its column-parallel layers as one DTensor (one backward reduction)")
    p.add_argument("--tp_sp", action="store_true", help="tensor parallel + sequence parallel")
    p.add_argument("--microbatches", type=int, default=8, help="pipeline microbatches per step (pp)")
    p.add_argument("--pp_schedule", choices=("1f1b", "gpipe"), default="1f1b")
    p.add_argument("--pp_split", type=int, default=None, help="blocks on the first stage (default: half)")
    p.add_argument("--pp_virtual", type=int, default=1, help="stages per GPU; 2 = interleaved 1F1B")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--skip", type=int, default=20)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--peak_flops", type=float, default=68e12, help="measured bf16 matmul peak per GPU")
    p.add_argument("--profile", default=None, help="directory for profiler traces + breakdown")
    p.add_argument("--check_grads", default=None, help="save one-step gradients to this directory and exit")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="append the result as a JSON line to this file")
    return p.parse_args()


# ------------------------------------------------------------------ helpers
class CkptBlock(nn.Module):
    """Recompute a block's activations in backward instead of storing them."""
    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, x):
        return checkpoint(self.blk, x, use_reentrant=False)


class Stage(nn.Module):
    """One pipeline stage: optional embeddings, a run of blocks, optional head."""
    def __init__(self, m, blocks, first, last):
        super().__init__()
        self.first, self.last = first, last
        if first:
            self.tok_emb, self.pos_emb = m.tok_emb, m.pos_emb
        self.blocks = nn.Sequential(*blocks)
        if last:
            self.final_norm, self.out_head = m.final_norm, m.out_head

    def forward(self, x):
        if self.first:
            pos = torch.arange(x.shape[1], device=x.device)
            x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        if self.last:
            x = self.out_head(self.final_norm(x))
        return x


def clean_name(n: str) -> str:
    for junk in ("module.", "_orig_mod.", "_checkpoint_wrapped_module."):
        n = n.replace(junk, "")
    return n.replace(".blk.", ".")


def global_grad_clip(params, max_norm, strategy, device, skip_ids=()):
    """Clip by the global norm when parameters are split across ranks (tp, pp)."""
    from torch.distributed.tensor import DTensor
    sharded = torch.zeros((), device=device)
    local = torch.zeros((), device=device)
    for p in params:
        if p.grad is None or id(p) in skip_ids:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            sq = g.to_local().float().pow(2).sum()
            if any(pl.is_shard() for pl in g.placements):
                sharded += sq
            else:
                local += sq
        else:
            local += g.float().pow(2).sum()
    if strategy == "tp":          # shards differ per rank; replicated params are identical
        dist.all_reduce(sharded)
        total = sharded + local
    else:                         # pp: every rank owns different parameters
        total = sharded + local
        dist.all_reduce(total)
    norm = total.sqrt()
    coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for p in params:
        if p.grad is not None:
            g = p.grad
            (g.to_local() if isinstance(g, DTensor) else g).mul_(coef)
    return norm


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    i = j = 0
    t = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def analyze_trace(path, wall_ms):
    """Split GPU time in a chrome trace into compute / overlapped comm / exposed comm / idle."""
    ev = json.load(open(path))["traceEvents"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    comm = union([(e["ts"], e["ts"] + e["dur"]) for e in kern if "nccl" in e["name"].lower()])
    comp = union([(e["ts"], e["ts"] + e["dur"]) for e in kern if "nccl" not in e["name"].lower()])
    busy = union([(e["ts"], e["ts"] + e["dur"]) for e in kern])
    overlap = intersect(comm, comp)
    us = 1e-3
    return {"compute_ms": total(comp) * us, "comm_ms": total(comm) * us,
            "comm_overlapped_ms": overlap * us, "comm_exposed_ms": (total(comm) - overlap) * us,
            "idle_ms": max(0.0, wall_ms - total(busy) * us), "wall_ms": wall_ms}


def main():
    a = parse_args()
    d = setup_distributed()
    ws, rank, dev = d.world_size, d.rank, d.device
    if (a.strategy == "single") != (ws == 1):
        raise SystemExit("--strategy single runs without torchrun; the others need torchrun with 2+ GPUs")
    tf32 = a.precision == "bf16"                 # strict fp32 runs disable TF32 too
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    compile_ = not a.no_compile
    if a.check_grads:
        torch.use_deterministic_algorithms(False)

    mc = resolve_model_config({"preset": a.preset, "vocab_size": a.vocab_size, "drop_rate": 0.0,
                               "tie_embeddings": True, "attn_impl": "sdpa"})
    torch.manual_seed(123)                       # identical init on every rank
    model = build_model(mc).to(dev)
    n_params = count_params(model)
    flops_tok = flops_per_token(mc, n_params)
    T = mc.context_length
    id_to_name = {id(p): n for n, p in model.named_parameters()}

    # data-parallel degree decides how the global batch is split
    dp = ws if a.strategy in ("ddp", "fsdp") else 1
    per_rank = a.global_batch // dp
    if a.strategy == "pp":
        loader_bs, accum = a.global_batch, 1
    else:
        loader_bs = min(a.micro_batch, per_rank)
        accum = per_rank // loader_bs
    assert loader_bs * accum * dp == a.global_batch, "global_batch must divide evenly"
    ds = split_dataset(a.data, "train", T, None)

    if a.check_grads:                            # fixed batch: the first global_batch windows
        X = torch.stack([ds[i][0] for i in range(a.global_batch)])
        Y = torch.stack([ds[i][1] for i in range(a.global_batch)])
        mine = slice(rank * per_rank, (rank + 1) * per_rank) if dp > 1 else slice(None)
        fixed = list(zip(X[mine].split(loader_bs), Y[mine].split(loader_bs)))
        fixed_it = iter(fixed)

        def batch():
            x, y = next(fixed_it)
            return x.to(dev), y.to(dev)
    else:
        dl = train_loader(ds, loader_bs, rank if dp > 1 else 0, dp, seed=123, num_workers=4)
        it = iter(dl)

        def batch():
            nonlocal it
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)
            return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    def wrap_blocks(blocks, shard_fn=None):
        out = []
        for blk in blocks:
            if compile_:
                blk.compile()
            if shard_fn:
                shard_fn(blk)
            out.append(CkptBlock(blk) if a.act_ckpt else blk)
        return out

    schedule, tied_skip, shared_param, first_rank_input = None, set(), None, False
    if a.strategy in ("single", "ddp"):
        model.trf_blocks = nn.Sequential(*wrap_blocks(list(model.trf_blocks)))
        train_model = model
        if a.strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            kw = dict(bucket_cap_mb=100, gradient_as_bucket_view=True) if a.ddp_tuned else {}
            train_model = DDP(model, device_ids=[d.local_rank], **kw)
            if a.ddp_bf16_grads:  # all-reduce gradients in bf16 (halves traffic; rounds gradients)
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
                train_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
        params = list(model.parameters())
    elif a.strategy == "fsdp":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if a.fsdp_bf16 \
            else MixedPrecisionPolicy()
        reshard = not a.fsdp_no_reshard
        model.trf_blocks = nn.Sequential(*wrap_blocks(
            list(model.trf_blocks), lambda b: fully_shard(b, mp_policy=mp, reshard_after_forward=reshard)))
        fully_shard(model, mp_policy=mp, reshard_after_forward=reshard)
        train_model, params = model, list(model.parameters())
    elif a.strategy == "tp":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput,
                                                       PrepareModuleOutput, RowwiseParallel,
                                                       SequenceParallel, parallelize_module)
        mesh = init_device_mesh("cuda", (ws,))
        # --tp_shared_input: hand the attention / MLP input to Q, K, V (and to the first MLP layer) as one
        # DTensor, so backward sums their input gradients first and reduces once (as torchtitan does).
        # Without it each column-parallel layer gets a plain tensor and all-reduces its own input gradient.
        shared = a.tp_shared_input
        if a.tp_sp:
            to_full = lambda: PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),),
                                                 use_local_output=not shared)
            plan = {"norm1": SequenceParallel(), "att": to_full(),
                    "att.W_query": ColwiseParallel(), "att.W_key": ColwiseParallel(),
                    "att.W_value": ColwiseParallel(), "att.out_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "norm2": SequenceParallel(), "ff": to_full(),
                    "ff.layers.0": ColwiseParallel(), "ff.layers.2": RowwiseParallel(output_layouts=Shard(1))}
        else:
            plan = {"att.W_query": ColwiseParallel(), "att.W_key": ColwiseParallel(),
                    "att.W_value": ColwiseParallel(), "att.out_proj": RowwiseParallel(),
                    "ff.layers.0": ColwiseParallel(), "ff.layers.2": RowwiseParallel()}
            if shared:
                rep = lambda: PrepareModuleInput(input_layouts=(Replicate(),), desired_input_layouts=(Replicate(),),
                                                 use_local_output=False)
                plan = {"att": rep(), "ff": rep(), **plan}
        blocks = list(model.trf_blocks)
        for blk in blocks:
            parallelize_module(blk, mesh, plan)
            blk.att.n_heads //= ws                  # each rank now holds n_heads / ws heads
        if a.tp_sp:                                 # residual stream sharded on the sequence dim
            parallelize_module(blocks[0], mesh, PrepareModuleInput(
                input_layouts=(Replicate(),), desired_input_layouts=(Shard(1),), use_local_output=True))
            parallelize_module(blocks[-1], mesh, PrepareModuleOutput(
                output_layouts=(Shard(1),), desired_output_layouts=(Replicate(),), use_local_output=True))
        model.trf_blocks = nn.Sequential(*wrap_blocks(blocks))
        train_model, params = model, list(model.parameters())
    else:  # pp
        from torch.distributed.pipelining import (PipelineStage, Schedule1F1B, ScheduleGPipe,
                                                  ScheduleInterleaved1F1B)
        blocks = list(model.trf_blocks)
        n_chunks = ws * a.pp_virtual
        if a.pp_split is not None:
            assert n_chunks == 2, "--pp_split needs 2 stages"
            sizes = [a.pp_split, len(blocks) - a.pp_split]
        else:
            assert len(blocks) % n_chunks == 0
            sizes = [len(blocks) // n_chunks] * n_chunks
        bounds = [sum(sizes[:i]) for i in range(n_chunks + 1)]
        stages, mods = [], []
        for c in range(rank, n_chunks, ws):          # looped placement: chunk c on rank c % ws
            mod = Stage(model, wrap_blocks(blocks[bounds[c]:bounds[c + 1]]), c == 0, c == n_chunks - 1)
            mods.append(mod)
            stages.append(PipelineStage(mod, stage_index=c, num_stages=n_chunks, device=dev))
            if c == 0:
                shared_param, first_rank_input = mod.tok_emb.weight, True
            if c == n_chunks - 1:
                shared_param = mod.out_head.weight   # tied copy of tok_emb on the last stage
                tied_skip = {id(shared_param)}       # count it once in the global grad norm
        del model
        if a.pp_virtual > 1:
            schedule = ScheduleInterleaved1F1B(stages, n_microbatches=a.microbatches, loss_fn=lm_loss)
        else:
            cls = Schedule1F1B if a.pp_schedule == "1f1b" else ScheduleGPipe
            schedule = cls(stages[0], n_microbatches=a.microbatches, loss_fn=lm_loss)
        train_model = nn.ModuleList(mods)
        params = list(train_model.parameters())

    # fused AdamW cannot mix DTensor (tensor-parallel shards) and plain tensors in one call,
    # so under tp the two kinds of parameters get one fused optimizer each
    from torch.distributed.tensor import DTensor
    groups = [[p for p in params if isinstance(p, DTensor)], [p for p in params if not isinstance(p, DTensor)]]
    opts = [torch.optim.AdamW(g, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
            for g in groups if g]

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.precision == "bf16" else nullcontext()

    def fwd_bwd():
        """One optimizer step's worth of forward/backward; returns the mean loss on this rank."""
        loss_val = torch.zeros((), device=dev)
        if schedule is not None:
            x, y = batch()
            mb_losses = []
            args = (x,) if first_rank_input else ()               # this rank holds the first stage
            kw = dict(target=y, losses=mb_losses) if tied_skip else {}  # this rank holds the last stage
            with autocast:
                schedule.step(*args, **kw)
            if mb_losses:
                loss_val = torch.stack([l.detach().float() for l in mb_losses]).mean()
            # the tied embedding lives on two stages: sum its two gradient contributions
            dist.all_reduce(shared_param.grad)
            return loss_val
        for micro in range(accum):
            x, y = batch()
            last = micro == accum - 1
            ctx = train_model.no_sync() if (a.strategy == "ddp" and not last) else nullcontext()
            if a.strategy == "fsdp":
                train_model.set_requires_gradient_sync(last)   # reduce-scatter once per step
            with ctx:
                with autocast:
                    logits = train_model(x)
                loss = lm_loss(logits, y)
                (loss / accum).backward()
            loss_val += loss.detach() / accum
        return loss_val

    def clip():
        if a.strategy == "tp":
            global_grad_clip(params, 1.0, "tp", dev)
        elif a.strategy == "pp":
            global_grad_clip(params, 1.0, "pp", dev, tied_skip)
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)

    # ------------------------------------------------------------ gradient check mode
    if a.check_grads:
        fwd_bwd()
        out = {}
        named = ([(id_to_name[id(p)], p) for p in params] if a.strategy == "pp"
                 else [(clean_name(n), p) for n, p in (model.named_parameters())])
        for n, p in named:
            if p.grad is None:
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            out[n] = g.detach().float().cpu()
        Path(a.check_grads).mkdir(parents=True, exist_ok=True)
        torch.save(out, Path(a.check_grads) / f"rank{rank}.pt")
        if rank == 0:
            print(f"saved {len(out)} gradients per rank to {a.check_grads}")
        cleanup_distributed()
        return

    # ------------------------------------------------------------ timed steps
    def train_step():
        loss = fwd_bwd()
        clip()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss

    times, losses = [], []
    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(1, a.steps + 1):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        losses.append(train_step())
        torch.cuda.synchronize(dev)
        if step > a.skip:
            times.append(time.perf_counter() - t0)

    breakdown = None
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        Path(a.profile).mkdir(parents=True, exist_ok=True)
        n_prof = 3
        torch.cuda.synchronize(dev)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(n_prof):
                train_step()
            torch.cuda.synchronize(dev)
            wall_ms = (time.perf_counter() - t0) * 1e3
        trace = Path(a.profile) / f"{a.label or a.strategy}_rank{rank}.json"
        prof.export_chrome_trace(str(trace))
        bd = analyze_trace(trace, wall_ms)
        breakdown = {k: v / n_prof for k, v in bd.items()}   # per step

    # ------------------------------------------------------------ report
    step_s = statistics.median(times)
    spread = (min(times), max(times))
    tok_s = a.global_batch * T / step_s
    mfu = flops_tok * tok_s / (a.peak_flops * ws)
    mem = torch.tensor([torch.cuda.max_memory_allocated(dev) / 1e9,
                        torch.cuda.max_memory_reserved(dev) / 1e9], device=dev)
    final_loss = losses[-1].clone()
    bds = [breakdown]
    if ws > 1:
        mems = [torch.zeros_like(mem) for _ in range(ws)]
        dist.all_gather(mems, mem)
        if a.strategy == "pp":                      # loss is only known on the last stage
            dist.broadcast(final_loss, src=ws - 1)
        else:
            dist.all_reduce(final_loss, op=dist.ReduceOp.AVG)
        if breakdown is not None:
            bds = [None] * ws
            dist.all_gather_object(bds, breakdown)
    else:
        mems = [mem]
    label = a.label or a.strategy
    res = {"label": label, "strategy": a.strategy, "gpus": ws, "preset": a.preset,
           "params_m": round(n_params / 1e6, 1), "global_batch": a.global_batch,
           "micro_batch": loader_bs if a.strategy != "pp" else a.global_batch // a.microbatches,
           "accum": accum, "microbatches": a.microbatches if a.strategy == "pp" else None,
           "options": {k: getattr(a, k) for k in ("act_ckpt", "ddp_tuned", "ddp_bf16_grads", "fsdp_bf16", "fsdp_no_reshard", "tp_shared_input",
                                                   "tp_sp", "pp_schedule", "pp_split", "pp_virtual",
                                                   "precision")},
           "compile": compile_, "step_s": step_s, "step_s_min_max": spread, "tok_per_s": tok_s,
           "mfu": mfu, "peak_flops": a.peak_flops,
           "peak_mem_gb": max(m[0].item() for m in mems),
           "mem_alloc_per_gpu_gb": [round(m[0].item(), 2) for m in mems],
           "mem_reserved_per_gpu_gb": [round(m[1].item(), 2) for m in mems],
           "final_loss": final_loss.item(), "steps": a.steps, "skip": a.skip,
           "breakdown_per_rank": bds if breakdown is not None else None, "torch": torch.__version__}
    if rank == 0:
        print(f"{label:22s} {ws} GPU | {tok_s:9,.0f} tok/s | MFU {100 * mfu:5.1f}% | "
              f"step {1000 * step_s:7.1f} ms | mem/GPU " +
              "/".join(f"{m[0].item():.1f}" for m in mems) + f" GB | loss {res['final_loss']:.3f}")
        if breakdown is not None:
            for r, b in enumerate(bds):
                print(f"   rank{r}: compute {b['compute_ms']:.0f} ms | comm overlapped {b['comm_overlapped_ms']:.0f}"
                      f" | comm exposed {b['comm_exposed_ms']:.0f} | idle {b['idle_ms']:.0f} | wall {b['wall_ms']:.0f}")
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")
    cleanup_distributed()


if __name__ == "__main__":
    main()
