"""Instruction finetuning (book chapter 7) on one GPU or all of them.

Starts from any checkpoint of this repo -- OpenAI GPT-2 imported with import_gpt2.py, or
your own pretraining run -- and trains on Alpaca-style {"instruction", "input", "output"}
records. Afterwards it writes the finetuned model and its responses to the test split.

  python import_gpt2.py --size 355M
  torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-355m.pt \
      --data data/instruct/instruction-data.json --out_dir runs/sft_gpt2_355m

--batch_size is the global batch (default 8, as in the book). On 8 GPUs that is one
example per GPU per step, so the optimisation matches the single-GPU book run.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from llmtrain.checkpoint import load_model, save_checkpoint
from llmtrain.config import TrainConfig
from llmtrain.data import ShardSampler
from llmtrain.dist import all_reduce_mean, all_reduce_sum, barrier, cleanup_distributed, setup_distributed
from llmtrain.evaluation import autocast_context, evaluate, lm_loss
from llmtrain.generation import generate_text
from llmtrain.instruct import InstructionDataset, collate, format_prompt, load_entries, split_entries
from llmtrain.optim import build_optimizer, lr_at
from llmtrain.tokenizer import get_tokenizer
from llmtrain.tracking import MetricsLogger, plot_losses


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="checkpoint to start from")
    p.add_argument("--data", default="data/instruct/instruction-data.json", help="JSON or JSONL records")
    p.add_argument("--val_data", default=None, help="optional separate validation file")
    p.add_argument("--test_data", default=None, help="optional separate test file")
    p.add_argument("--out_dir", default="runs/sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="global batch size (examples per step)")
    p.add_argument("--micro_batch_size", type=int, default=None,
                   help="examples per GPU per forward pass; default batch_size / num_gpus")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--max_length", type=int, default=1024, help="truncate examples to this many tokens")
    p.add_argument("--mask_prompt", action="store_true", help="train on response tokens only")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_batches", type=int, default=5, help="per-GPU batches per periodic eval")
    p.add_argument("--max_new_tokens", type=int, default=256, help="for test-set responses")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    d = setup_distributed()
    log = print if d.is_main else (lambda *a, **k: None)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.batch_size % d.world_size:
        raise ValueError(f"--batch_size {args.batch_size} must be divisible by the GPU count {d.world_size}")
    micro = args.micro_batch_size or args.batch_size // d.world_size
    if args.batch_size % (micro * d.world_size):
        raise ValueError("--batch_size must be a multiple of micro_batch_size x num_gpus")
    ga = args.batch_size // (micro * d.world_size)

    out_dir = Path(args.out_dir)
    if d.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- model
    model, cfg, meta, base_ckpt = load_model(args.base, d.device)
    model.train()
    enc = get_tokenizer(meta["tokenizer"])
    max_length = min(args.max_length, cfg.model.context_length)

    # ---------------------------------------------------------------- data
    entries = load_entries(args.data)
    if args.val_data or args.test_data:
        train_e = entries
        val_e = load_entries(args.val_data) if args.val_data else []
        test_e = load_entries(args.test_data) if args.test_data else []
    else:
        train_e, val_e, test_e = split_entries(entries)
    log(f"data: {len(train_e)} train / {len(val_e)} val / {len(test_e)} test examples from {args.data}")

    coll = partial(collate, pad_id=enc.eot_token, max_length=max_length, mask_prompt=args.mask_prompt)
    train_ds, val_ds = InstructionDataset(train_e, enc), InstructionDataset(val_e, enc)
    sampler = DistributedSampler(train_ds, num_replicas=d.world_size, rank=d.rank, shuffle=True,
                                 seed=args.seed, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=micro, sampler=sampler, collate_fn=coll, drop_last=True)
    train_eval_dl = DataLoader(train_ds, batch_size=micro, collate_fn=coll,
                               sampler=ShardSampler(train_ds, d.rank, d.world_size))
    val_dl = DataLoader(val_ds, batch_size=micro, collate_fn=coll,
                        sampler=ShardSampler(val_ds, d.rank, d.world_size)) if len(val_ds) else None

    # ---------------------------------------------------------------- optimiser (book: AdamW defaults)
    tc = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, beta2=0.999, warmup_steps=args.warmup_steps,
                     lr_schedule=args.lr_schedule, grad_clip=args.grad_clip)
    optimizer = build_optimizer(model, tc, d.device)
    steps_per_epoch = len(train_dl) // ga
    total_steps = args.epochs * steps_per_epoch
    ddp_model = DDP(model, device_ids=[d.local_rank]) if d.enabled else model
    autocast = autocast_context(d.device, args.precision)

    log(f"base: {args.base} ({base_ckpt.get('source', 'step ' + str(base_ckpt.get('step')))}) | "
        f"{cfg.model.type} {cfg.model.n_layers} layers, d={cfg.model.emb_dim}")
    log(f"run:  {d.world_size} GPU(s) x micro-batch {micro} x accum {ga} = batch {args.batch_size} | "
        f"{steps_per_epoch} steps/epoch | {total_steps} steps | lr {args.lr} | "
        f"{'response-only loss' if args.mask_prompt else 'loss on prompt + response'}")

    def run_eval(max_batches):
        m = {f"train/{k}": v for k, v in evaluate(model, train_eval_dl, d.device, args.precision,
                                                   max_batches).items() if k != "tokens"}
        if val_dl is not None:
            m.update({f"val/{k}": v for k, v in evaluate(model, val_dl, d.device, args.precision,
                                                          max_batches).items() if k != "tokens"})
        return m

    def fmt(m):
        s = f"train loss {m['train/loss']:.3f}"
        return s + (f", val loss {m['val/loss']:.3f}" if "val/loss" in m else "")

    logger = MetricsLogger(str(out_dir), tensorboard=True) if d.is_main else None
    log(f"initial: {fmt(run_eval(args.eval_batches))}")

    # ---------------------------------------------------------------- train
    torch.manual_seed(args.seed + d.rank)
    step, tokens_seen, t0 = 0, 0, time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        batches = iter(train_dl)
        for _ in range(steps_per_epoch):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, tc, total_steps)
            loss_accum = torch.zeros((), device=d.device)
            n_tok = torch.zeros((), device=d.device)
            for m_i in range(ga):
                x, y = next(batches)
                x, y = x.to(d.device), y.to(d.device)
                sync = nullcontext() if (not d.enabled or m_i == ga - 1) else ddp_model.no_sync()
                with sync:
                    with autocast:
                        logits = ddp_model(x)
                    loss = lm_loss(logits, y)
                    (loss / ga).backward()
                loss_accum += loss.detach() / ga
                n_tok += x.numel()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += int(all_reduce_sum(n_tok).item())
            loss_val = all_reduce_mean(loss_accum).item()
            if logger:
                logger.log(step, **{"train/step_loss": loss_val, "tokens_seen": tokens_seen,
                                    "lr": optimizer.param_groups[0]["lr"]})

            if step % args.eval_every == 0 or step == total_steps:
                m = run_eval(args.eval_batches)
                log(f"Ep {epoch + 1} (Step {step:06d}/{total_steps}): {fmt(m)}")
                if logger:
                    logger.log(step, tokens_seen=tokens_seen, **m)
        if d.is_main:
            prompt = format_prompt(val_e[0] if val_e else train_e[0])
            text = generate_text(model, enc, prompt, d.device, 35, cfg.model.context_length,
                                 precision=args.precision)
            log(f"  sample after epoch {epoch + 1}: {text[len(prompt):].replace(enc.decode([enc.eot_token]), '').strip()!r}")
    log(f"training took {(time.time() - t0) / 60:.2f} min")

    # ---------------------------------------------------------------- full evaluation + save
    final = run_eval(None)
    log(f"final (full splits): {fmt(final)}" +
        (f", val ppl {final['val/perplexity']:.2f}, val acc {100 * final['val/accuracy']:.1f}%"
         if "val/loss" in final else ""))
    if d.is_main:
        save_checkpoint(out_dir / "model.pt", model, cfg, meta, step=step, tokens_seen=tokens_seen,
                        source=f"sft of {args.base}", sft_args=vars(args), final_metrics=final)
        logger.log(step, tokens_seen=tokens_seen, **{f"final_{k}": v for k, v in final.items()})
        logger.close()
        plot_losses(out_dir / "metrics.jsonl", out_dir / "loss.png")
        log(f"saved {out_dir / 'model.pt'}")

    # ---------------------------------------------------------------- test-set responses (sharded)
    if test_e:
        mine = []
        for i in range(d.rank, len(test_e), d.world_size):
            prompt = format_prompt(test_e[i])
            text = generate_text(model, enc, prompt, d.device, args.max_new_tokens,
                                 cfg.model.context_length, precision=args.precision)
            mine.append((i, text[len(prompt):].replace("<|endoftext|>", "").strip()))
        gathered = [None] * d.world_size
        if d.enabled:
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        if d.is_main:
            responses = dict(r for part in gathered for r in part)
            results = [{**e, "model_response": responses[i]} for i, e in enumerate(test_e)]
            path = out_dir / "test_responses.json"
            path.write_text(json.dumps(results, indent=2))
            for r in results[:3]:
                log(f"\n### Instruction: {r['instruction']}" + (f"\n### Input: {r['input']}" if r.get("input") else "")
                    + f"\n>> expected: {r['output']}\n>> model:    {r['model_response']}")
            log(f"\nsaved {len(results)} test responses to {path}  (score them: python score_responses.py {path})")
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
