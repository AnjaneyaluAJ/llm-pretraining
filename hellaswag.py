"""Zero-shot HellaSwag accuracy of a checkpoint (10,042 validation examples, random = 25%).

  python hellaswag.py --checkpoint /mnt/nvme/runs/gpt2_124m/ckpt_last.pt
  torchrun --standalone --nproc_per_node=4 hellaswag.py --checkpoint pretrained/gpt2-124m.pt
Compare against OpenAI's GPT-2 by importing it first: python import_gpt2.py --size 124M
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llmtrain.checkpoint import load_model
from llmtrain.dist import cleanup_distributed, setup_distributed
from llmtrain.hellaswag import evaluate_hellaswag
from llmtrain.tokenizer import get_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--limit", type=int, default=None, help="only the first N examples")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--output", default=None, help="results JSON (default: next to the checkpoint)")
    args = p.parse_args()

    d = setup_distributed()
    model, cfg, meta, ckpt = load_model(args.checkpoint, d.device)
    enc = get_tokenizer(meta["tokenizer"])
    t0 = time.time()
    res = evaluate_hellaswag(model, enc, d.device, args.precision, d.rank, d.world_size, args.limit, args.split)

    if d.is_main:
        print(f"checkpoint: {args.checkpoint} (step {ckpt.get('step', '?')}, "
              f"{ckpt.get('tokens_seen', 0):,} tokens seen)")
        print(f"HellaSwag {args.split}: acc {100 * res['acc']:.2f}% | acc_norm {100 * res['acc_norm']:.2f}% "
              f"| {res['examples']:,} examples in {time.time() - t0:.0f}s on {d.world_size} GPU(s)")
        out = Path(args.output) if args.output else Path(args.checkpoint).with_name(
            Path(args.checkpoint).stem + f"_hellaswag_{args.split}.json")
        with open(out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "step": ckpt.get("step"), **res}, f, indent=2)
        print(f"saved {out}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
