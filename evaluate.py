"""Evaluate a checkpoint: loss, perplexity, next-token accuracy and bits-per-byte.

Evaluate on a prepared dataset (defaults to the training run's validation split):
  python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt
  python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --data_dir data/other --split val
Evaluate on raw text files, tokenized on the fly with the checkpoint's tokenizer:
  python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --text_files held_out.txt
Use every GPU (the split is sharded across ranks and the totals are summed):
  torchrun --standalone --nproc_per_node=8 evaluate.py --checkpoint ... --data_dir ...
Add generations for qualitative checks:
  ... --prompt "Every effort moves you" --prompt "The picture was" --max_new_tokens 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llmtrain.checkpoint import load_model
from llmtrain.data import TokenWindowDataset, eval_loader, load_meta, split_dataset
from llmtrain.dist import cleanup_distributed, setup_distributed
from llmtrain.evaluation import evaluate
from llmtrain.generation import generate_text
from llmtrain.tokenizer import encode, get_tokenizer, token_dtype


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data_dir", help="prepared dataset (default: the one the model was trained on)")
    src.add_argument("--text_files", nargs="+", help="raw text files to evaluate on")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--context_length", type=int, default=None, help="default: model context length")
    p.add_argument("--stride", type=int, default=None, help="default: context length (no overlap)")
    p.add_argument("--max_batches", type=int, default=None, help="per-GPU cap on evaluated batches")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--prompt", action="append", default=[], help="repeatable")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--output", default=None, help="results JSON (default: next to the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    d = setup_distributed()
    log = print if d.is_main else (lambda *a, **k: None)

    model, cfg, train_meta, ckpt = load_model(args.checkpoint, d.device)
    enc = get_tokenizer(train_meta["tokenizer"])
    ctx = args.context_length or cfg.model.context_length
    if ctx > cfg.model.context_length:
        raise ValueError(f"--context_length {ctx} exceeds the model's {cfg.model.context_length}")

    if args.text_files:
        texts = [Path(f).read_text(encoding="utf-8", errors="replace") for f in args.text_files]
        ids: list[int] = []
        for text in texts:
            ids += encode(enc, text) + [enc.eot_token]
        tokens = np.asarray(ids, dtype=token_dtype(enc.n_vocab))
        dataset = TokenWindowDataset(tokens, ctx, args.stride)
        bytes_per_token = sum(len(t.encode("utf-8")) for t in texts) / max(1, len(tokens))
        source = ", ".join(args.text_files)
        name = Path(args.text_files[0]).stem
    else:
        data_dir = args.data_dir or cfg.data.data_dir
        meta = load_meta(data_dir)
        if meta["tokenizer"] != train_meta["tokenizer"]:
            raise ValueError(f"{data_dir} was tokenized with {meta['tokenizer']}, but the model "
                             f"was trained with {train_meta['tokenizer']}")
        dataset = split_dataset(data_dir, args.split, ctx, args.stride)
        bytes_per_token = meta.get("bytes_per_token")
        source = f"{data_dir} [{args.split}]"
        name = f"{Path(data_dir).name}_{args.split}"

    if len(dataset) == 0:
        raise ValueError(f"{source}: {dataset.n_tokens} tokens is not enough for one window of {ctx}")

    loader = eval_loader(dataset, args.batch_size, d.rank, d.world_size)
    results = evaluate(model, loader, d.device, args.precision, args.max_batches, bytes_per_token)

    samples = []
    if d.is_main:
        for prompt in args.prompt:
            samples.append({"prompt": prompt, "text": generate_text(
                model, enc, prompt, d.device, args.max_new_tokens, cfg.model.context_length,
                temperature=args.temperature, top_k=args.top_k, precision=args.precision)})

    log(f"checkpoint: {args.checkpoint} (step {ckpt.get('step', '?')}, "
        f"{ckpt.get('tokens_seen', 0):,} tokens seen)")
    log(f"data:       {source} | {dataset.n_tokens:,} tokens, {len(dataset):,} windows of {ctx}")
    log(f"loss        {results['loss']:.4f}")
    log(f"perplexity  {results['perplexity']:.2f}")
    log(f"accuracy    {100 * results['accuracy']:.2f}%  (top-1 next token)")
    if "bits_per_byte" in results:
        log(f"bits/byte   {results['bits_per_byte']:.4f}")
    log(f"tokens      {results['tokens']:,} evaluated on {d.world_size} GPU(s)")
    for s in samples:
        log(f"\n--- prompt: {s['prompt']!r}\n{s['text']}")

    if d.is_main:
        out = Path(args.output) if args.output else Path(args.checkpoint).parent / f"eval_{name}.json"
        with open(out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "source": source, "context_length": ctx,
                       "step": ckpt.get("step"), **results, "samples": samples}, f, indent=2)
        log(f"\nsaved {out}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
