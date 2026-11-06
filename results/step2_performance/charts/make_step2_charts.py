"""Post 3 charts (step 2: performance engineering + multi-GPU) from results/step2_performance/summary.json.

usage: python make_step2_charts.py <summary.json> <out_dir> [extra.jsonl] [extra_profile.jsonl]
extra.jsonl holds follow-up runs (DDP with bf16 gradient all-reduce) in bench_parallel.py format.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

S = json.load(open(sys.argv[1]))
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
EXTRA = [json.loads(l) for l in open(sys.argv[3])] if len(sys.argv) > 3 else []
EXTRA_PROF = [json.loads(l) for l in open(sys.argv[4])] if len(sys.argv) > 4 else []
for lab in {r["label"] for r in EXTRA}:  # median over repeats, speed-up vs the 1-GPU run at the same batch
    rs = [r for r in EXTRA if r["label"] == lab]
    tok = statistics.median(r["tok_per_s"] for r in rs)
    ref = S["parallel"]["single" if rs[0]["global_batch"] == 32 else "single_gb128"]["tok_per_s"]
    S["parallel"][lab] = {"tok_per_s": tok, "speedup": tok / ref, "gpus": rs[0]["gpus"], "reps": len(rs)}
for r in EXTRA_PROF:
    S["profiles"][r["label"].removesuffix("_prof")] = r["breakdown_per_rank"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def figure(kicker, title, sub, rect=(0.30, 0.17, 0.63, 0.60)):
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=SURFACE)
    fig.text(0.07, 0.92, kicker.upper(), fontsize=15, color=BLUE, weight="bold")
    fig.text(0.07, 0.865, title, fontsize=34, weight="bold", va="center")
    fig.text(0.07, 0.81, sub, fontsize=17, color=INK2, va="center")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.08, 0.08], color="#dcdbd5", lw=1.5))
    fig.text(0.07, 0.045, "github.com/AnjaneyaluAJ/llm-pretraining", fontsize=16, color=INK2, va="center")
    fig.text(0.93, 0.045, "Anjaneyalu Thippaiah", fontsize=16, color=INK2, va="center", ha="right")
    ax = fig.add_axes(rect, facecolor=SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=15, length=0, pad=10)
    return fig, ax


def hbar(ax, y, x0, w, color, h=0.5):
    """Horizontal bar, square at its start, 4px-ish rounded at the data end."""
    ax.barh(y, w, left=x0, height=h, color=color, edgecolor=SURFACE, linewidth=2)


# 1. Single-GPU ladder ------------------------------------------------------------
L = S["ladder"]
steps = [("L0_fp32_baseline", "Baseline (FP32)"), ("L1_tf32", "+ TF32 matmuls"), ("L2_bf16", "+ BF16 mixed precision"),
         ("L3_flash_attention", "+ Flash attention"), ("L4_fused_adamw", "+ Fused AdamW"),
         ("L5_torch_compile", "+ torch.compile"), ("L6_vocab_50304", "+ Vocab padded to 50,304"),
         ("L7_batch_16", "+ Batch 8 -> 16")]
fig, ax = figure("Part 3: making training fast", f"{L['L7_batch_16']['vs_baseline']:.1f}x faster on the same GPU",
                 "Tokens per second, one change at a time (GPT-2 124M, median of 3 runs)",
                 rect=(0.30, 0.21, 0.63, 0.56))
labels, ys = [], []
for i, (k, name) in enumerate(steps):
    y = len(steps) - 1 - i
    r = L[k]
    hbar(ax, y, 0, r["tok_per_s"] / 1000, BLUE if i else GRAY)
    gain = "" if i == 0 else f"   {r['gain_vs_previous']:.2f}x"
    ax.text(r["tok_per_s"] / 1000 + 0.6, y, f"{r['tok_per_s'] / 1000:.1f}k{gain}", va="center", fontsize=15, color=INK)
    labels.append(name); ys.append(y)
ax.set_yticks(ys, labels, fontsize=16, color=INK)
ax.set_xlim(0, 58)
ax.set_xlabel("Thousand tokens per second", fontsize=16, labelpad=12)
mfu0, mfu1 = L["L0_fp32_baseline"]["mfu"], L["L7_batch_16"]["mfu"]
fig.text(0.07, 0.11, f"Share of the GPU's measured peak used for model maths (MFU): {100 * mfu0:.0f}% -> {100 * mfu1:.0f}%", fontsize=15, color=INK2)
fig.savefig(OUT / "step2_ladder.png", facecolor=SURFACE); plt.close(fig)

# 2. Multi-GPU speed-up -------------------------------------------------------------
P = S["parallel"]
rows = [  # (key, label, group)
    ("tp_shared", "Tensor parallel", 0), ("pp_1f1b_m8", "Pipeline (even split)", 0),
    ("pp_1f1b_m8_split9", "Pipeline (balanced split)", 0), ("ddp", "DDP", 0),
    ("ddp_bf16grads", "DDP, bf16 gradients", 0), ("fsdp_bf16_noreshard", "FSDP, bf16 params", 0),
    ("ddp_gb128", "DDP", 1), ("ddp_bf16grads_gb128", "DDP, bf16 gradients", 1), ("fsdp_bf16_gb128", "FSDP, bf16 params", 1),
    ("ddp4_gb128", "DDP", 2), ("ddp4_bf16grads_gb128", "DDP, bf16 gradients", 2), ("fsdp4_bf16_gb128", "FSDP, bf16 params", 2),
]
groups = ["2 GPUs, batch 32", "2 GPUs, batch 128", "4 GPUs, batch 128"]
colors = [BLUE, ORANGE, AQUA]
fig, ax = figure("Part 3: adding GPUs", "More GPUs is not automatically faster",
                 "Speed-up over one GPU at the same batch size (GPT-2 124M, median of 3 runs)",
                 rect=(0.38, 0.17, 0.55, 0.60))
y, ys, labels, prev = 0, [], [], None
for key, name, g in reversed(rows):
    if prev is not None and g != prev:
        y += 0.7
    r = P[key]
    hbar(ax, y, 0, r["speedup"], colors[g])
    ax.text(r["speedup"] + 0.04, y, f"{r['speedup']:.2f}x", va="center", fontsize=15)
    ys.append(y); labels.append(name); prev = g; y += 1
ax.set_yticks(ys, labels, fontsize=15, color=INK)
for n, xv in ((1, 1), (2, 2), (4, 4)):
    ax.axvline(xv, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=0)
ax.set_xlim(0, 4.3)
ax.set_xticks([0, 1, 2, 3, 4], ["0", "1x", "2x", "3x", "4x"])
ax.set_xlabel("Speed-up over a single GPU (dotted lines: perfect scaling)", fontsize=16, labelpad=12)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles[::-1], groups[::-1], loc="upper right", fontsize=14, frameon=False)
fig.savefig(OUT / "step2_multigpu.png", facecolor=SURFACE); plt.close(fig)

# 3. Where the time goes (profiler) -------------------------------------------------
PR = S["profiles"]
prow = [("single", 0, "1 GPU"), ("ddp", 0, "DDP"), ("ddp_bf16grads", 0, "DDP, bf16 gradients"),
        ("fsdp_bf16", 0, "FSDP, bf16 params"),
        ("pp_1f1b_m8", 0, "Pipeline, even split: GPU 1"), ("pp_1f1b_m8", 1, "Pipeline, even split: GPU 2"),
        ("pp_1f1b_m8_split9", 0, "Pipeline, balanced: GPU 1"), ("pp_1f1b_m8_split9", 1, "Pipeline, balanced: GPU 2"),
        ("tp_shared", 0, "Tensor parallel")]
fig, ax = figure("Part 3: where the time goes", "Compute vs waiting on the other GPU",
                 "Profiled time per training step, batch 32 (GPT-2 124M)", rect=(0.38, 0.19, 0.55, 0.58))
ys, labels = [], []
for i, (k, rank, name) in enumerate(prow):
    yy = len(prow) - 1 - i
    r = PR[k][rank]
    c, e, idle = r["compute_ms"], r["comm_exposed_ms"], r["idle_ms"]
    hbar(ax, yy, 0, c, BLUE); hbar(ax, yy, c, e, ORANGE)
    ax.text(c + e + idle + 25, yy, f"{r['wall_ms']:.0f} ms", va="center", fontsize=15)
    ys.append(yy); labels.append(name)
ax.set_yticks(ys, labels, fontsize=15, color=INK)
ax.set_xlim(0, 2300)
ax.set_xlabel("Milliseconds per training step", fontsize=16, labelpad=12)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
ax.legend(handles, ["GPU computing", "GPU waiting on communication"], loc="center right", fontsize=14, frameon=False)
fig.text(0.07, 0.115, "Communication hidden under compute is counted as compute. Pipeline 'waiting' includes the pipeline bubble.",
         fontsize=13, color=MUTED)
fig.savefig(OUT / "step2_time_breakdown.png", facecolor=SURFACE); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("step2_*.png")))
