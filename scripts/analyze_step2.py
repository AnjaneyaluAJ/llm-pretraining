"""Turn the step-2 campaign outputs into tables (and JSON) for the write-up.

    python scripts/analyze_step2.py /mnt/nvme/runs/step2 --hw /mnt/nvme/runs/hw --out results/step2_performance

Sections: ladder (cumulative) and leave-one-out on 1 GPU; parallel strategies with
median / min / max over interleaved repeats; a first-principles communication model
(bytes each strategy must move, divided by the measured NCCL bandwidth) next to the
measured overhead over perfect scaling; profiler breakdowns; memory limits; GPU clocks.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--hw", default="/mnt/nvme/runs/hw")
p.add_argument("--out", default=None)
p.add_argument("--skip", type=int, default=30, help="ladder: ignore log intervals up to this step")
a = p.parse_args()
R = Path(a.root)
summary = {}
med = statistics.median


def jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]


# ------------------------------------------------------------ hardware reference
hw = {"matmul": jsonl(Path(a.hw) / "matmul.jsonl")}
for n in (2, 4):
    rows = jsonl(Path(a.hw) / f"comm_{n}.jsonl")
    big = [r for r in rows if r["mb"] >= 64]
    if big:
        hw[n] = {k: med(r[k] for r in big if k in r) for k in big[0] if k.endswith("gbs")}
summary["hardware"] = hw
print("## Hardware")
if hw["matmul"]:
    print("bf16 matmul TFLOP/s per GPU:", [m["bf16_tflops"] for m in hw["matmul"]],
          "| tf32:", [m["tf32_tflops"] for m in hw["matmul"]], "| fp32:", [m["fp32_tflops"] for m in hw["matmul"]])
for n in (2, 4):
    if n in hw:
        print(f"{n} GPUs, GB/s (bus bandwidth, >=64 MB messages):", {k: round(v, 2) for k, v in hw[n].items()})

# ------------------------------------------------------------ ladder + leave-one-out
lad = defaultdict(list)
for f in sorted(R.glob("ladder/rep*/*/metrics.jsonl")):
    recs = [r for r in jsonl(f) if "tok_per_s" in r and r["step"] > a.skip]
    if recs:
        lad[f.parent.name].append({"tok_per_s": med(r["tok_per_s"] for r in recs),
                                   "mfu": med(r["mfu"] for r in recs),
                                   "mem": max(r.get("peak_mem_gb", 0) for r in recs),
                                   "loss": recs[-1]["train/step_loss"]})
if lad:
    print("\n## Ladder (1 GPU, GPT-2 124M, batch 8 unless noted) - median [min, max] over repeats")
    agg = {}
    for name in sorted(lad):
        xs = lad[name]
        t = [x["tok_per_s"] for x in xs]
        agg[name] = {"tok_per_s": med(t), "min": min(t), "max": max(t), "reps": len(xs),
                     "mfu": med(x["mfu"] for x in xs), "mem_gb": max(x["mem"] for x in xs),
                     "loss": med(x["loss"] for x in xs)}
    base = agg.get("L0_fp32_baseline", {}).get("tok_per_s")
    full = agg.get("L6_vocab_50304", {}).get("tok_per_s")
    prev = None
    print(f"{'config':22s} {'tok/s':>8s} {'[min, max]':>18s} {'vs base':>8s} {'step gain':>9s} {'MFU':>6s} {'mem':>6s} {'loss':>6s}")
    for name, r in agg.items():
        r["vs_baseline"] = r["tok_per_s"] / base if base else None
        if name.startswith("L"):
            r["gain_vs_previous"] = r["tok_per_s"] / prev if prev else None
            prev = r["tok_per_s"]
        else:
            r["loss_when_removed"] = 1 - r["tok_per_s"] / full if full else None
        extra = (f"{r['gain_vs_previous']:8.2f}x" if r.get("gain_vs_previous") else
                 (f"{-100 * r['loss_when_removed']:+7.1f}%" if r.get("loss_when_removed") is not None else " " * 9))
        print(f"{name:22s} {r['tok_per_s']:8,.0f} [{r['min']:7,.0f},{r['max']:7,.0f}] "
              f"{(r['vs_baseline'] or 0):7.2f}x {extra} {100 * r['mfu']:5.1f}% {r['mem_gb']:5.1f} {r['loss']:6.3f}")
    summary["ladder"] = agg

# ------------------------------------------------------------ parallel strategies
par = defaultdict(list)
for r in jsonl(R / "parallel.jsonl") + jsonl(R / "extra.jsonl"):  # extra = follow-up runs
    par[r["label"]].append(r)
if par:
    print("\n## Parallel strategies (GPT-2 124M) - median [min, max] over repeats")
    agg = {}
    for label, xs in par.items():
        t = [x["tok_per_s"] for x in xs]
        x0 = xs[0]
        agg[label] = {"gpus": x0["gpus"], "global_batch": x0["global_batch"], "tok_per_s": med(t),
                      "min": min(t), "max": max(t), "reps": len(xs), "step_s": med(x["step_s"] for x in xs),
                      "mfu": med(x["mfu"] for x in xs), "mem_per_gpu": x0["mem_alloc_per_gpu_gb"],
                      "loss": med(x["final_loss"] for x in xs), "params_m": x0["params_m"],
                      "options": x0["options"], "micro_batch": x0["micro_batch"], "accum": x0["accum"],
                      "microbatches": x0["microbatches"]}
    singles = {v["global_batch"]: v for k, v in agg.items() if v["gpus"] == 1}
    print(f"{'label':24s} {'GPUs':>4s} {'gb':>4s} {'tok/s':>8s} {'[min, max]':>18s} {'speed-up':>8s} {'eff':>5s} {'MFU':>6s} {'mem/GPU GB':>12s} {'loss':>6s}")
    for label, r in agg.items():
        s = singles.get(r["global_batch"])
        r["speedup"] = r["tok_per_s"] / s["tok_per_s"] if s else None
        r["scaling_eff"] = r["speedup"] / r["gpus"] if s else None
        print(f"{label:24s} {r['gpus']:4d} {r['global_batch']:4d} {r['tok_per_s']:8,.0f} [{r['min']:7,.0f},{r['max']:7,.0f}] "
              f"{(r['speedup'] or 0):7.2f}x {100 * (r['scaling_eff'] or 0):4.0f}% {100 * r['mfu']:5.1f}% "
              f"{'/'.join(f'{m:.1f}' for m in r['mem_per_gpu']):>12s} {r['loss']:6.3f}")
    summary["parallel"] = agg

    # -------------------------------------------------------- communication model
    print("\n## Communication model: predicted transfer time vs measured overhead over perfect scaling")
    print("(predicted = bytes / measured NCCL bandwidth, assuming no overlap with compute)")
    model_rows = {}
    for label, r in agg.items():
        n, gb = r["gpus"], r["global_batch"]
        s = singles.get(gb)
        if n == 1 or not s or n not in hw:
            continue
        N = r["params_m"] * 1e6
        bw = hw[n]
        ar = bw["all_reduce_busbw_gbs"] * 1e9
        ag = bw["all_gather_busbw_gbs"] * 1e9
        rs = bw["reduce_scatter_busbw_gbs"] * 1e9
        o = r["options"]
        T, d, L = 1024, 768, 12
        strat = label.split("_")[0].rstrip("4")
        if strat == "ddp":
            gbytes = 2 if o.get("ddp_bf16_grads") else 4
            pred = gbytes * N * 2 * (n - 1) / n / ar
            what = 'all-reduce ' + ('bf16' if gbytes == 2 else 'fp32') + ' grads once'
        elif strat == "fsdp":
            pb = 2 if o["fsdp_bf16"] else 4
            gathers = 1 if o["fsdp_no_reshard"] else 2
            pred = gathers * pb * N * (n - 1) / n / ag + 4 * N * (n - 1) / n / rs
            pred *= 1  # gradient sync happens once per step; parameter gathers happen per micro-batch:
            pred += (r["accum"] - 1) * gathers * pb * N * (n - 1) / n / ag
            what = f"{gathers} param all-gather(s) x {r['accum']} micro + 1 grad reduce-scatter"
        elif strat == "tp":
            # Counted from the profiler trace: per block per micro-batch, forward all-reduces the
            # outputs of out_proj and of the second MLP layer (bf16); backward all-reduces the input
            # gradient of every column-parallel layer: W_query, W_key, W_value (separate modules) and
            # the first MLP layer, in fp32 because their input (the LayerNorm output) is fp32.
            # With --tp_shared_input, Q/K/V (and the MLP input) share one DTensor input, so backward
            # reduces 2 summed input gradients per block instead of 4. Sequence parallel uses
            # reduce-scatter / all-gather instead and is not modeled here (see the trace counts).
            if o.get("tp_sp"):
                continue
            tokens = r["micro_batch"] * T * d
            per_block = 2 * tokens * 2 + (2 if o.get("tp_shared_input") else 4) * tokens * 4
            pred = L * r["accum"] * per_block * 2 * (n - 1) / n / ar
            what = "per block per micro-batch: 2 bf16 fwd + " + ("2" if o.get("tp_shared_input") else "4") + " fp32 bwd activation all-reduces"
        elif strat == "pp":
            m = r["microbatches"]
            act = (gb // m) * T * d * 4                             # fp32 residual stream between stages
            p2p = bw.get("send_recv_gbs", ar / 1e9) * 1e9
            comm = 2 * m * act / p2p
            ideal = s["step_s"] / n
            bubble = ideal * (n - 1) / m
            pred = comm + bubble
            what = f"activations+grads p2p ({comm * 1e3:.0f} ms) + bubble ({bubble * 1e3:.0f} ms, balanced stages)"
        else:
            continue
        measured = r["step_s"] - s["step_s"] / n
        model_rows[label] = {"pred_ms": pred * 1e3, "measured_overhead_ms": measured * 1e3, "what": what}
        print(f"{label:24s} predicted {pred * 1e3:7.0f} ms | measured {measured * 1e3:7.0f} ms | {what}")
    summary["comm_model"] = model_rows

# ------------------------------------------------------------ profiles
prof = jsonl(R / "profile.jsonl") + jsonl(R / "extra_profile.jsonl")
if prof:
    print("\n## Profiler breakdown per step (ms): compute | comm hidden under compute | comm exposed | idle | wall")
    for r in prof:
        for i, b in enumerate(r["breakdown_per_rank"] or []):
            print(f"{r['label']:22s} rank{i}: {b['compute_ms']:7.0f} | {b['comm_overlapped_ms']:7.0f} | "
                  f"{b['comm_exposed_ms']:7.0f} | {b['idle_ms']:7.0f} | {b['wall_ms']:7.0f}")
    summary["profiles"] = {r["label"]: r["breakdown_per_rank"] for r in prof}

# ------------------------------------------------------------ memory limits
mem = jsonl(R / "memory.jsonl")
if mem or (R.parent.parent / "logs").exists():
    print("\n## Memory limits (runs that fit; anything missing ran out of memory)")
    for r in mem:
        print(f"{r['label']:34s} {r['gpus']} GPU | {r['tok_per_s']:7,.0f} tok/s | MFU {100 * r['mfu']:4.1f}% | "
              f"mem/GPU {'/'.join(f'{m:.1f}' for m in r['mem_alloc_per_gpu_gb'])} GB")
    summary["memory_fits"] = {r["label"]: {"tok_per_s": r["tok_per_s"], "mem": r["mem_alloc_per_gpu_gb"],
                                           "gpus": r["gpus"]} for r in mem}

# ------------------------------------------------------------ clocks
clk = R / "gpu_clocks.csv"
if clk.exists():
    rows = list(csv.reader(open(clk)))[1:]
    sm = [int(x[2].split()[0]) for x in rows if len(x) > 5 and x[2].strip().split()[0].isdigit()]
    busy = [int(x[2].split()[0]) for x in rows if len(x) > 5 and x[5].strip().split()[0].isdigit()
            and int(x[5].split()[0]) > 50]
    temp = [int(x[3]) for x in rows if len(x) > 5 and x[3].strip().isdigit()]
    if busy:
        summary["clocks"] = {"busy_sm_mhz_min": min(busy), "busy_sm_mhz_median": med(busy),
                             "temp_max_c": max(temp), "samples": len(rows)}
        print(f"\n## GPU clocks while busy (>50% util): min {min(busy)} MHz, median {med(busy)} MHz; "
              f"max temp {max(temp)} C; {len(rows)} samples")

if a.out:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
