"""Compare one-step gradients of each parallel strategy with the single-GPU reference.

    python scripts/compare_grads.py /mnt/nvme/runs/gradcheck/<precision> --ref single

Each subdirectory holds rank*.pt files from `bench_parallel.py --check_grads`. Ranks are
merged (pipeline stages hold different parameters; data/tensor-parallel ranks hold the
same full gradients after gathering). For each run we report the relative L2 error of
the whole gradient vector against the reference, the worst per-tensor relative error,
and the cosine similarity.
"""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("root")
p.add_argument("--ref", default="single")
p.add_argument("--json")
a = p.parse_args()


def load(run_dir):
    merged = {}
    for f in sorted(Path(run_dir).glob("rank*.pt")):
        for k, v in torch.load(f).items():
            merged.setdefault(k, v)
    return merged


root = Path(a.root)
ref = load(root / a.ref)
rows = []
print(f"{'run':28s} {'tensors':>7s} {'rel L2 err':>11s} {'worst tensor':>13s} {'cosine':>10s}")
for run in sorted(x for x in root.iterdir() if x.is_dir()):
    g = load(run)
    missing = sorted(set(ref) - set(g))
    extra = sorted(set(g) - set(ref))
    keys = [k for k in ref if k in g]
    num = sum((g[k] - ref[k]).pow(2).sum() for k in keys).sqrt()
    den = sum(ref[k].pow(2).sum() for k in keys).sqrt()
    dot = sum((g[k] * ref[k]).sum() for k in keys)
    gn = sum(g[k].pow(2).sum() for k in keys).sqrt()
    worst = max(((g[k] - ref[k]).norm() / (ref[k].norm() + 1e-12)).item() for k in keys)
    r = {"run": run.name, "tensors": len(keys), "missing": missing, "extra": extra,
         "rel_l2_err": (num / den).item(), "worst_tensor_rel_err": worst, "cosine": (dot / (gn * den)).item()}
    rows.append(r)
    flag = "" if not (missing or extra) else f"  MISSING {missing[:3]} EXTRA {extra[:3]}"
    print(f"{run.name:28s} {len(keys):7d} {r['rel_l2_err']:11.2e} {worst:13.2e} {r['cosine']:10.6f}{flag}")
if a.json:
    Path(a.json).write_text(json.dumps(rows, indent=2))
