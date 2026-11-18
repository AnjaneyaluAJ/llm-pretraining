"""Step 3 IsoFLOP analysis: best model size per compute budget, and a prediction for the next budget.

    python scripts/step3/analyze.py results_lr.jsonl results_p2.jsonl results_p2c.jsonl --out isoflop.json

For each budget C, take each size's best run (lowest validation loss over its learning rates), fit a
parabola to loss vs ln(N) through the lowest point and its two neighbours, and read off N_opt (vertex),
L_min and D_opt = C / (6 N_opt). N is non-embedding parameters, C = 6 N D (Kaplan-style counting).
N_opt(C) and L_min(C) are fitted as power laws (least squares in log-log over all budgets) and extrapolated.
"""
import argparse, json, math
from collections import defaultdict

import numpy as np


def isoflop(rows):
    best = defaultdict(dict)                                   # budget -> size -> best row
    for r in rows:
        if r.get("val_loss") is None:
            continue
        cur = best[r["budget"]].get(r["size"])
        if cur is None or r["val_loss"] < cur["val_loss"]:
            best[r["budget"]][r["size"]] = r
    out = []
    for C in sorted(best):
        pts = sorted(best[C].values(), key=lambda r: r["n_nonemb"])
        x = np.log([r["n_nonemb"] for r in pts]); y = np.array([r["val_loss"] for r in pts])
        i = int(np.argmin(y)); j = min(max(i - 1, 0), len(pts) - 3)
        a, b, c = np.polyfit(x[j:j + 3], y[j:j + 3], 2)
        n_opt = math.exp(-b / (2 * a))
        out.append({"budget": C, "n_opt": n_opt, "l_min": c - b * b / (4 * a), "d_opt": C / (6 * n_opt),
                    "tokens_per_param": C / (6 * n_opt) / n_opt, "min_at_edge": i in (0, len(pts) - 1),
                    "points": [{"size": r["size"], "n_nonemb": r["n_nonemb"], "tokens": r["tokens"],
                                "lr": r["lr"], "val_loss": r["val_loss"]} for r in pts]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--predict", type=float, nargs="*", default=[1e18])
    p.add_argument("--out")
    a = p.parse_args()
    rows = [json.loads(l) for f in a.results for l in open(f) if l.strip()]
    fits = isoflop(rows)
    for f in fits:
        print(f"C={f['budget']:.0e}: N_opt {f['n_opt'] / 1e6:.1f}M, D_opt {f['d_opt'] / 1e9:.2f}B, "
              f"D/N {f['tokens_per_param']:.0f}, L_min {f['l_min']:.4f}" + ("  (min at edge!)" if f["min_at_edge"] else ""))
    res = {"fits": fits}
    if len(fits) >= 2:
        lc = np.log([f["budget"] for f in fits])
        a_n, k_n = np.polyfit(lc, np.log([f["n_opt"] for f in fits]), 1)      # N_opt = e^k C^a
        b_l, k_l = np.polyfit(lc, np.log([f["l_min"] for f in fits]), 1)      # L_min = e^k C^b (b < 0)
        res["n_opt_exponent"] = float(a_n)
        res["l_min_exponent"] = float(b_l)
        res["d_opt_exponent"] = float(1 - a_n)
        res["predictions"] = []
        for C in a.predict:
            n = math.exp(k_n) * C ** a_n
            l = math.exp(k_l) * C ** b_l
            res["predictions"].append({"budget": C, "n_opt": n, "d_opt": C / (6 * n), "l_min_pure_power_law": l})
            print(f"PREDICTION C={C:.0e}: N_opt {n / 1e6:.0f}M, D_opt {C / (6 * n) / 1e9:.2f}B, "
                  f"L_min {l:.3f} (pure power law in C; an irreducible-loss term would make it higher)")
        print(f"N_opt ~ C^{a_n:.2f}, D_opt ~ C^{1 - a_n:.2f} (least squares in log-log over {len(fits)} budgets)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
