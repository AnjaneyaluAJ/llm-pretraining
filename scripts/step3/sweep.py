"""Step 3 scaling-law sweep: train a list of GPT-2-style models, each for a fixed compute budget.

Every run gets its own full cosine schedule (warmup 5%, decay to 10% of peak) that ends exactly
at its token budget, so final losses are comparable across sizes (Chinchilla, not Kaplan).

Compute is counted Kaplan-style as C = 6 * N * D with N = non-embedding parameters
(12 * layers * d_model^2); total parameters are reported too.

    python scripts/step3/sweep.py --phase lr --dry_run          # print the run table
    python scripts/step3/sweep.py --phase lr                    # run it (inside a SageMaker job)

Outputs per run in $RESULTS/<run>/ (default /opt/ml/checkpoints, synced to S3): metrics.jsonl,
config.yaml, train.log, summary.json, DONE. A run with DONE is skipped, so a restarted job resumes.
"""
import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

VOCAB, CTX = 50304, 1024
TOKENS_PER_STEP = 4 * 16 * CTX * 4          # 4 GPUs x micro 16 x ctx x accum 4 = 262,144

#            layers d_model heads micro accum
SIZES = {
    "s10":  (6,  384,  6, 16, 4),
    "s20":  (6,  512,  8, 16, 4),
    "s35":  (8,  576,  9, 16, 4),
    "s50":  (10, 640, 10, 16, 4),
    "s85":  (12, 768, 12, 16, 4),
    "s135": (14, 896, 14, 16, 4),
    "s200": (16, 1024, 16, 8, 8),
    "s300": (24, 1024, 16, 8, 8),
}


def n_nonemb(size):
    L, d = SIZES[size][:2]
    return 12 * L * d * d


def n_total(size):
    d = SIZES[size][1]
    return n_nonemb(size) + VOCAB * d + CTX * d      # tied embeddings + learned positions


def run_spec(size, budget, lr, seed=123, tag=""):
    steps = round(budget / (6 * n_nonemb(size) * TOKENS_PER_STEP))
    name = f"{size}_C{budget:.0e}_lr{lr:.1e}".replace("+", "") + (f"_seed{seed}" if seed != 123 else "") + tag
    return dict(name=name, size=size, budget=budget, lr=lr, seed=seed, steps=steps,
                tokens=steps * TOKENS_PER_STEP, n_nonemb=n_nonemb(size), n_total=n_total(size))


PHASES = {
    # learning-rate sweep at C = 1e17 on two sizes; these runs double as IsoFLOP points
    "lr": [run_spec(s, 1e17, lr) for s in ("s20", "s85") for lr in (6e-4, 1.2e-3, 2.4e-3)],
}


def last_metric(metrics_path, key):
    val = None
    for line in open(metrics_path):
        rec = json.loads(line)
        if key in rec:
            val = rec[key]
    return val


def train_one(r, data_dir, scratch, results, keep_ckpt):
    L, d, h, micro, accum = SIZES[r["size"]]
    out = Path(scratch) / "runs" / r["name"]
    dest = Path(results) / r["name"]
    if (dest / "DONE").exists():
        print(f"skip {r['name']} (done)", flush=True)
        return json.load(open(dest / "summary.json"))
    shutil.rmtree(out, ignore_errors=True)
    warmup = max(20, round(0.05 * r["steps"]))
    evals = max(1, r["steps"] // 8)
    sets = [f"model.n_layers={L}", f"model.emb_dim={d}", f"model.n_heads={h}", f"model.vocab_size={VOCAB}",
            f"data.data_dir={data_dir}", f"train.out_dir={out}", f"train.seed={r['seed']}",
            f"train.batch_size={micro}", f"train.grad_accum_steps={accum}", f"train.max_steps={r['steps']}",
            f"train.lr={r['lr']}", f"train.warmup_steps={warmup}", "train.min_lr_ratio=0.1",
            f"train.eval_every={evals}", "train.eval_batches=null", "train.train_eval_batches=20",
            "train.hellaswag_every=1000000000", "train.sample_every=1000000000",
            "train.save_every=1000000000", "train.log_every=25", "train.peak_flops=6.8e+13",
            "train.resume=null"]
    cmd = ["torchrun", "--standalone", "--nproc_per_node=4", "train.py",
           "--config", "configs/gpt2_124m_fineweb_a10g.yaml", "--set", *sets]
    print(f"\n=== RUN {r['name']}: {r['steps']} steps, {r['tokens'] / 1e9:.2f}B tokens, lr {r['lr']}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "train.log", "w") as logf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            logf.write(line)
            sys.stdout.write(line)                             # every 25 steps -> CloudWatch metrics
        rc = p.wait()
    hours = (time.time() - t0) / 3600
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.jsonl", "config.yaml", "train.log", "loss.png"):
        if (out / f).exists():
            shutil.copy(out / f, dest / f)
    if rc != 0:
        print(f"!!! {r['name']} failed (exit {rc}); log in {dest}/train.log", flush=True)
        return None
    if keep_ckpt and r["size"] in keep_ckpt:
        shutil.copy(out / "ckpt_last.pt", dest / "ckpt_last.pt")
    m = out / "metrics.jsonl"
    summary = {**r, "val_loss": last_metric(m, "val/loss"), "train_loss": last_metric(m, "train/loss"),
               "hellaswag_acc_norm": last_metric(m, "hellaswag/acc_norm"), "hours": hours}
    json.dump(summary, open(dest / "summary.json", "w"), indent=1)
    (dest / "DONE").touch()
    with open(Path(results) / "results.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"=== {r['name']} done: val loss {summary['val_loss']:.4f}, "
          f"hellaswag {summary['hellaswag_acc_norm']}, {hours:.2f} h", flush=True)
    shutil.rmtree(out, ignore_errors=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, choices=sorted(PHASES))
    p.add_argument("--data_dir", default="/opt/ml/input/data/train")
    p.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp/scratch"))
    p.add_argument("--results", default="/opt/ml/checkpoints")
    p.add_argument("--keep_ckpt", nargs="*", default=["s200", "s300"], help="sizes whose final checkpoint is kept")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    runs = PHASES[a.phase]
    print(f"{'run':32s} {'N_nonemb':>9s} {'N_total':>8s} {'steps':>6s} {'tokens':>7s} {'tok/N':>6s}")
    for r in runs:
        print(f"{r['name']:32s} {r['n_nonemb'] / 1e6:8.1f}M {r['n_total'] / 1e6:7.1f}M {r['steps']:6d} "
              f"{r['tokens'] / 1e9:6.2f}B {r['tokens'] / r['n_nonemb']:6.1f}")
    if a.dry_run:
        return
    for r in runs:
        train_one(r, a.data_dir, a.scratch, a.results, a.keep_ckpt)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
