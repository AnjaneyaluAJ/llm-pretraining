"""LinkedIn charts for the GPT-2 124M FineWeb-Edu run: HellaSwag and validation loss vs tokens."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

rows = [json.loads(l) for l in open(RUN / "metrics.jsonl")]
hs = [(r["tokens_seen"] / 1e9, 100 * r["hellaswag/acc_norm"]) for r in rows if "hellaswag/acc_norm" in r]
vl = [(r["tokens_seen"] / 1e9, r["val/loss"]) for r in rows if "val/loss" in r]
openai_hs = 100 * json.load(open(RUN / "openai_gpt2_hellaswag_val.json"))["acc_norm"]
openai_vl = json.load(open(RUN / "openai_gpt2_eval_val.json"))["loss"]
hs = [(0.0, 25.0)] + hs  # untrained model = random guessing over 4 endings


def figure(kicker, title, sub):
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=SURFACE)
    fig.text(0.07, 0.92, kicker.upper(), fontsize=15, color=BLUE, weight="bold")
    fig.text(0.07, 0.865, title, fontsize=34, weight="bold", va="center")
    fig.text(0.07, 0.81, sub, fontsize=17, color=INK2, va="center")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.08, 0.08], color="#dcdbd5", lw=1.5))
    fig.text(0.07, 0.045, "github.com/AnjaneyaluAJ/llm-pretraining", fontsize=16, color=INK2, va="center")
    fig.text(0.93, 0.045, "Anjaneyalu Thippaiah", fontsize=16, color=INK2, va="center", ha="right")
    ax = fig.add_axes([0.1, 0.21, 0.83, 0.54], facecolor=SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.tick_params(labelsize=15, length=0, pad=8)
    ax.set_xlabel("Training tokens (billions)", fontsize=16, labelpad=12)
    ax.set_xlim(0, 10.2)
    return fig, ax


# 1. HellaSwag
fig, ax = figure("GPT-2 from scratch", "My GPT-2 beats OpenAI's GPT-2",
                 "HellaSwag accuracy (common-sense benchmark, 10,042 questions)")
x, y = zip(*hs)
ax.axhline(openai_hs, color=ORANGE, lw=2, ls=(0, (6, 4)))
ax.text(0.15, openai_hs + 0.18, f"OpenAI GPT-2 124M: {openai_hs:.1f}%", color=INK2, fontsize=15)
ax.axhline(25, color=MUTED, lw=1.5, ls=(0, (2, 3)))
ax.text(10.1, 25.18, "random guessing: 25%", color=MUTED, fontsize=14, ha="right")
ax.plot(x, y, color=BLUE, lw=3, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
ax.annotate(f"Mine: {y[-1]:.1f}%", (x[-1], y[-1]), xytext=(-12, 18), textcoords="offset points",
            fontsize=19, weight="bold", ha="right", color=INK)
cross = next(t for t, v in hs if v > openai_hs)
ax.annotate(f"passes GPT-2 at\n~{cross:.1f}B tokens", (cross, openai_hs), xytext=(cross - 2.6, openai_hs + 1.0),
            fontsize=15, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, shrinkB=6))
ax.set_ylim(24.5, 32)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
fig.savefig(OUT / "gpt2_hellaswag.png", facecolor=SURFACE)
plt.close(fig)

# 2. Validation loss
fig, ax = figure("GPT-2 from scratch", "Learning curve: 124M parameters",
                 "Validation loss on held-out FineWeb-Edu text (lower is better)")
x, y = zip(*vl)
ax.axhline(openai_vl, color=ORANGE, lw=2, ls=(0, (6, 4)))
ax.text(10.1, openai_vl + 0.04, f"OpenAI GPT-2 124M: {openai_vl:.2f}", color=INK2, fontsize=15, ha="right")
ax.plot(x, y, color=BLUE, lw=3)
ax.annotate(f"Mine: {y[-1]:.2f}", (x[-1], y[-1]), xytext=(-8, -30), textcoords="offset points",
            fontsize=19, weight="bold", ha="right")
ax.set_ylim(2.8, 4.6)
fig.text(0.07, 0.1, "Measured on FineWeb-Edu, the kind of text my model was trained on, so this comparison\n"
         "favours it. HellaSwag is the neutral test.", linespacing=1.5, fontsize=13, color=MUTED)
fig.savefig(OUT / "gpt2_val_loss.png", facecolor=SURFACE)
plt.close(fig)
print("crossing at", cross, "B tokens; final", hs[-1], "openai", openai_hs, openai_vl, "start val", vl[0])
