"""Charts for the step 3 README, from the committed results.

    python scripts/step3/plots.py --dir results/step3_scaling

Writes isoflop.png (loss vs model size per budget), n_opt.png (best size vs compute, with the
pre-registered prediction) and lr_sweeps.png (the two reference learning-rate sweeps).
"""
import argparse, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, validated (light surface)
MARKERS = ["o", "s", "D"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig(w=7.2, h=4.4):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    f.patch.set_facecolor(SURFACE)
    style(ax)
    return f, ax


def vertex_fit(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return a, b, c, -b / (2 * a)


def isoflop_chart(an, pred, out):
    f, ax = fig()
    for k, fit in enumerate(an["fits"]):
        pts = fit["points"]
        n = np.array([p["n_nonemb"] for p in pts]); y = np.array([p["val_loss"] for p in pts])
        x = np.log(n); i = int(np.argmin(y)); j = min(max(i - 1, 0), len(pts) - 3)
        a, b, c, xv = vertex_fit(x[j:j + 3], y[j:j + 3])
        xs = np.linspace(x[j] - 0.15, x[j + 2] + 0.15, 100)
        col = SERIES[k]
        ax.plot(n, y, color=col, linewidth=1, alpha=0.45, zorder=2)
        ax.plot(np.exp(xs), a * xs ** 2 + b * xs + c, color=col, linewidth=2, zorder=3)
        ax.scatter(n, y, s=36, color=col, marker=MARKERS[k], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.scatter([fit["n_opt"]], [fit["l_min"]], s=90, marker="*", color=col, edgecolor=INK, linewidth=0.6, zorder=5)
        ax.annotate(f"{fit['budget']:.0e} FLOPs\nbest {fit['n_opt'] / 1e6:.0f}M".replace("e+", "e"),
                    (n[-1], y[-1]), xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
    ax.scatter([pred["n_opt"]], [pred["l_min_pure_power_law"]], s=110, marker="*", facecolor="none",
               edgecolor=INK, linewidth=1.2, zorder=6)
    ax.annotate(f"predicted before the 1e18 runs\n{pred['n_opt'] / 1e6:.0f}M, loss {pred['l_min_pure_power_law']:.3f}",
                (pred["n_opt"], pred["l_min_pure_power_law"]), xytext=(0, -22), textcoords="offset points",
                fontsize=8.5, color=MUTED, ha="center", va="top")
    ax.set_ylim(2.82, None)
    ax.set_xscale("log")
    ax.set_xticks([1e7, 2e7, 5e7, 1e8, 2e8]); ax.set_xticklabels(["10M", "20M", "50M", "100M", "200M"])
    ax.set_xlim(8e6, 4.2e8)
    ax.set_xlabel("non-embedding parameters (log scale)", color=INK, fontsize=10)
    ax.set_ylabel("validation loss", color=INK, fontsize=10)
    ax.set_title("IsoFLOP curves: each line is one compute budget; filled star = fitted best size",
                 color=INK, fontsize=10.5, loc="left")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def nopt_chart(an, pred, out):
    f, ax = fig(6.4, 4.2)
    C = np.array([fit["budget"] for fit in an["fits"]]); N = np.array([fit["n_opt"] for fit in an["fits"]])
    e, k = np.polyfit(np.log(C), np.log(N), 1)
    cs = np.logspace(16.85, 18.15, 50)
    ax.plot(cs, np.exp(k) * cs ** e, color=SERIES[0], linewidth=2, zorder=2)
    ax.scatter(C, N, s=44, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=4)
    for c, n in zip(C, N):
        ax.annotate(f"{n / 1e6:.1f}M", (c, n), xytext=(-10, 8), textcoords="offset points", fontsize=9,
                    color=INK, ha="right")
    ax.scatter([pred["budget"]], [pred["n_opt"]], s=90, marker="*", facecolor="none", edgecolor=INK,
               linewidth=1.2, zorder=5)
    ax.annotate(f"predicted from 1e17 and 3e17: {pred['n_opt'] / 1e6:.0f}M", (pred["budget"], pred["n_opt"]),
                xytext=(-8, -20), textcoords="offset points", fontsize=8.5, color=MUTED, ha="right")
    ax.annotate(f"N_opt ~ C^{e:.2f}", (cs[18], np.exp(k) * cs[18] ** e), xytext=(10, -14),
                textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1e17, 3e17, 1e18]); ax.set_xticklabels(["1e17", "3e17", "1e18"])
    ax.set_yticks([1e7, 2e7, 5e7, 1e8]); ax.set_yticklabels(["10M", "20M", "50M", "100M"])
    ax.minorticks_off()
    ax.set_xlabel("training compute, FLOPs (C = 6ND)", color=INK, fontsize=10)
    ax.set_ylabel("best non-embedding size", color=INK, fontsize=10)
    ax.set_title("Compute-optimal model size vs compute", color=INK, fontsize=10.5, loc="left")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def lr_chart(rule, out):
    f, ax = fig(6.4, 4.2)
    refs = rule["references"]
    for k, ref in enumerate(refs):
        pts = sorted(ref["points"])
        lr = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        key = f"{ref['size']} at {ref['budget']:.0e} FLOPs".replace("e+", "e")
        col = SERIES[k]
        ax.plot(lr, y, color=col, linewidth=2, zorder=3)
        ax.scatter(lr, y, s=36, color=col, marker=MARKERS[k], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axvline(ref["lr_best"], color=col, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.annotate(f"{key}\nbest {ref['lr_best']:.2g}", (lr[0], y[0]), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ticks = [3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:.1e}".replace("e-0", "e-") for t in ticks]); ax.minorticks_off()
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.06)
    ax.set_xlabel("peak learning rate (log scale)", color=INK, fontsize=10)
    ax.set_ylabel("validation loss", color=INK, fontsize=10)
    ax.set_title("Reference learning-rate sweeps (dashed: parabola vertex)", color=INK, fontsize=10.5, loc="left")
    f.tight_layout(); f.savefig(out, facecolor=SURFACE); plt.close(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/step3_scaling")
    a = p.parse_args()
    d = Path(a.dir)
    an = json.loads((d / "analysis.json").read_text())
    pred = json.loads((d / "prediction_before_1e18.json").read_text())["predictions"][0]
    isoflop_chart(an, pred, d / "isoflop.png")
    nopt_chart(an, pred, d / "n_opt.png")
    lr_chart(json.loads((d / "lr_rule_p2c.json").read_text()), d / "lr_sweeps.png")
    print("wrote", ", ".join(str(d / n) for n in ("isoflop.png", "n_opt.png", "lr_sweeps.png")))


if __name__ == "__main__":
    main()
