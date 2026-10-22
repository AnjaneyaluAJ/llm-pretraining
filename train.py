"""Pretrain a GPT- or Llama-style model from scratch on a dataset made by prepare_data.py.

Single GPU:
  python train.py --config configs/gpt2_verdict.yaml
All GPUs on this node:
  torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml
Override any config field:
  torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
      --set train.lr=3e-4 train.epochs=20 data.data_dir=data/mycorpus
Resume the last checkpoint in out_dir:
  ... --set train.resume=auto
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llmtrain.checkpoint import load_checkpoint, save_checkpoint, unwrap
from llmtrain.config import load_config, save_config
from llmtrain.data import eval_loader, load_meta, split_dataset, train_loader
from llmtrain.dist import all_reduce_mean, barrier, cleanup_distributed, setup_distributed
from llmtrain.evaluation import autocast_context, evaluate, lm_loss
from llmtrain.generation import generate_text
from llmtrain.hellaswag import evaluate_hellaswag
from llmtrain.models import build_model, count_params, flops_per_token
from llmtrain.optim import build_optimizer, lr_at
from llmtrain.tokenizer import get_tokenizer
from llmtrain.tracking import MetricsLogger, plot_losses


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                   help="override config fields, e.g. train.lr=3e-4 model.n_layers=6")
    return p.parse_args()


def micro_batches(loader, start_micro: int):
    """Endless stream of (epoch, batch) that continues from a global micro-batch index."""
    per_epoch = len(loader)
    epoch, skip = divmod(start_micro, per_epoch)
    while True:
        loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if i >= skip:
                yield epoch, batch
        epoch, skip = epoch + 1, 0


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    mc, dc, tc = cfg.model, cfg.data, cfg.train
    d = setup_distributed()
    log = print if d.is_main else (lambda *a, **k: None)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(tc.out_dir)
    if d.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    # ---------------------------------------------------------------- data
    meta = load_meta(dc.data_dir)
    if mc.vocab_size < meta["vocab_size"]:
        raise ValueError(f"model.vocab_size={mc.vocab_size} is smaller than the data tokenizer's "
                         f"vocabulary ({meta['vocab_size']}, {meta['tokenizer']})")
    enc = get_tokenizer(meta["tokenizer"])

    train_ds = split_dataset(dc.data_dir, "train", mc.context_length, dc.stride)
    val_ds = split_dataset(dc.data_dir, "val", mc.context_length, mc.context_length)
    if len(train_ds) == 0:
        raise ValueError(f"Train split has {train_ds.n_tokens} tokens; need more than "
                         f"context_length={mc.context_length}")
    train_dl = train_loader(train_ds, tc.batch_size, d.rank, d.world_size, tc.seed, dc.num_workers)
    train_eval_dl = eval_loader(train_ds, tc.batch_size, d.rank, d.world_size)
    val_dl = eval_loader(val_ds, tc.batch_size, d.rank, d.world_size) if len(val_ds) else None
    bytes_per_token = meta.get("bytes_per_token")
    if val_dl is None:
        log(f"warning: val split has {val_ds.n_tokens} tokens, fewer than one window of "
            f"{mc.context_length + 1}; validation is skipped")

    # ---------------------------------------------------------------- model + optimizer
    torch.manual_seed(tc.seed)  # identical init on every rank
    model = build_model(mc).to(d.device)
    optimizer = build_optimizer(model, tc, d.device)
    n_params = count_params(model)

    step, best_val = 0, float("inf")
    resume_path = out_dir / "ckpt_last.pt" if tc.resume == "auto" else (Path(tc.resume) if tc.resume else None)
    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, map_location=d.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt.get("best_val", best_val)
        log(f"Resumed from {resume_path} at step {step}")
        del ckpt
    elif tc.resume not in (None, "auto"):
        raise FileNotFoundError(f"train.resume={tc.resume} does not exist")

    torch.manual_seed(tc.seed + d.rank + step)  # distinct dropout masks per rank
    compiled = torch.compile(model) if tc.compile else model
    # buffers (RoPE tables) are identical on every rank, so skip the per-forward buffer sync
    ddp_model = DDP(compiled, device_ids=[d.local_rank], forward_sync_buffers=False) if d.enabled else compiled

    # ---------------------------------------------------------------- schedule
    ga = tc.grad_accum_steps
    steps_per_epoch = max(1, len(train_dl) // ga)
    total_steps = tc.max_steps if tc.max_steps is not None else tc.epochs * steps_per_epoch
    tokens_per_step = tc.batch_size * mc.context_length * ga * d.world_size
    flops_tok = flops_per_token(mc, n_params)

    log(f"model: {mc.type} {n_params / 1e6:.1f}M params | layers={mc.n_layers} d={mc.emb_dim} "
        f"heads={mc.n_heads} ctx={mc.context_length}")
    log(f"data:  {meta['tokenizer']} | train {train_ds.n_tokens:,} tok ({len(train_ds):,} windows) | "
        f"val {val_ds.n_tokens:,} tok ({len(val_ds):,} windows)")
    log(f"run:   {d.world_size} GPU(s) x batch {tc.batch_size} x accum {ga} = "
        f"{tokens_per_step:,} tok/step | {steps_per_epoch} steps/epoch | {total_steps} steps total | "
        f"{tc.precision}{' + compile' if tc.compile else ''}")

    logger = MetricsLogger(tc.out_dir, tc.tensorboard) if d.is_main else None

    def run_eval() -> dict:
        metrics = {}
        tr = evaluate(compiled, train_eval_dl, d.device, tc.precision, tc.eval_batches, bytes_per_token)
        metrics.update({f"train/{k}": v for k, v in tr.items() if k != "tokens"})
        if val_dl is not None:
            va = evaluate(compiled, val_dl, d.device, tc.precision, tc.eval_batches, bytes_per_token)
            metrics.update({f"val/{k}": v for k, v in va.items() if k != "tokens"})
        return metrics

    def save(name: str, with_optimizer: bool) -> None:
        if d.is_main:
            save_checkpoint(out_dir / name, model, cfg, meta, optimizer if with_optimizer else None,
                            step=step, best_val=best_val, tokens_seen=step * tokens_per_step)
        barrier()

    def sample() -> None:
        if d.is_main:
            text = generate_text(model, enc, tc.sample_prompt, d.device, tc.sample_tokens,
                                 mc.context_length, temperature=tc.sample_temperature,
                                 top_k=tc.sample_top_k, precision=tc.precision)
            log(f"  sample: {text!r}")

    # ---------------------------------------------------------------- train loop
    stream = micro_batches(train_dl, step * ga)
    autocast = autocast_context(d.device, tc.precision)
    ddp_model.train()
    t_last, step_last = time.time(), step
    paused = 0.0  # seconds spent in eval / sampling / checkpointing, excluded from throughput
    last_eval_step = -1
    epoch = 0

    while step < total_steps:
        lr = lr_at(step, tc, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.zeros((), device=d.device)
        for micro in range(ga):
            epoch, (x, y) = next(stream)
            x, y = x.to(d.device, non_blocking=True), y.to(d.device, non_blocking=True)
            # only all-reduce gradients on the last micro-step
            sync = nullcontext() if (not d.enabled or micro == ga - 1) else ddp_model.no_sync()
            with sync:
                with autocast:
                    logits = ddp_model(x)
                loss = lm_loss(logits, y)
                (loss / ga).backward()
            loss_accum += loss.detach() / ga

        grad_norm = None
        if tc.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen = step * tokens_per_step

        if step % tc.log_every == 0 or step == 1 or step == total_steps:
            loss_val = all_reduce_mean(loss_accum).item()
            elapsed = time.time() - t_last - paused
            tok_s = (step - step_last) * tokens_per_step / max(elapsed, 1e-9)
            mfu = flops_tok * tok_s / (tc.peak_flops * d.world_size)
            t_last, step_last, paused = time.time(), step, 0.0
            gn = f" | grad {grad_norm.item():.2f}" if grad_norm is not None else ""
            log(f"Ep {epoch + 1} (Step {step:06d}/{total_steps}) | loss {loss_val:.4f} | lr {lr:.2e}{gn} | "
                f"{tok_s:,.0f} tok/s | MFU {100 * mfu:.1f}%")
            if logger:
                logger.log(step, **{"train/step_loss": loss_val, "lr": lr, "tokens_seen": tokens_seen,
                                    "epoch": epoch + 1, "tok_per_s": tok_s, "mfu": mfu,
                                    **({"grad_norm": grad_norm.item()} if grad_norm is not None else {})})

        t_pause = time.time()
        if step % tc.eval_every == 0 or step == total_steps:
            metrics = run_eval()
            last_eval_step = step
            msg = f"  eval step {step}: train loss {metrics['train/loss']:.3f}"
            if "val/loss" in metrics:
                msg += (f", val loss {metrics['val/loss']:.3f}, val ppl {metrics['val/perplexity']:.2f}, "
                        f"val acc {100 * metrics['val/accuracy']:.1f}%")
            log(msg)
            if logger:
                logger.log(step, tokens_seen=tokens_seen, **metrics)
            if metrics.get("val/loss", float("inf")) < best_val:
                best_val = metrics["val/loss"]
                save("ckpt_best.pt", with_optimizer=False)

        if tc.hellaswag_every and (step % tc.hellaswag_every == 0 or step == total_steps):
            hs = evaluate_hellaswag(model, enc, d.device, tc.precision, d.rank, d.world_size,
                                    tc.hellaswag_limit)
            log(f"  hellaswag step {step}: acc_norm {100 * hs['acc_norm']:.2f}%, "
                f"acc {100 * hs['acc']:.2f}% ({hs['examples']:,} examples)")
            if logger:
                logger.log(step, tokens_seen=tokens_seen, **{"hellaswag/acc": hs["acc"],
                                                             "hellaswag/acc_norm": hs["acc_norm"]})

        if tc.sample_every and (step % tc.sample_every == 0 or step == total_steps):
            sample()

        if tc.save_every and step % tc.save_every == 0 and step != total_steps:
            save("ckpt_last.pt", with_optimizer=True)
        paused += time.time() - t_pause

    # ---------------------------------------------------------------- finish
    if last_eval_step != step:
        run_eval()
    save("ckpt_last.pt", with_optimizer=True)
    if logger:
        logger.close()
        plot_losses(out_dir / "metrics.jsonl", out_dir / "loss.png")
    best = f"best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})" if best_val < float("inf") else "no validation"
    log(f"done: {step} steps, {step * tokens_per_step:,} tokens, {best}")
    log(f"outputs in {out_dir}/: ckpt_last.pt, ckpt_best.pt, metrics.jsonl, loss.png, config.yaml")
    cleanup_distributed()


if __name__ == "__main__":
    main()
