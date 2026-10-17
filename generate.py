"""Generate text from a trained checkpoint.

  python generate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --prompt "Every effort moves you"
  python generate.py --checkpoint ... --prompt "Once upon a time" --temperature 0.8 --top_k 50 --num_samples 3
  python generate.py --checkpoint ...            # interactive: type prompts, Ctrl-D to quit
"""

from __future__ import annotations

import argparse

import torch

from llmtrain.checkpoint import load_model
from llmtrain.generation import generate_text
from llmtrain.tokenizer import get_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default=None, help="omit for an interactive prompt")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, cfg, meta, ckpt = load_model(args.checkpoint, device)
    enc = get_tokenizer(meta["tokenizer"])
    print(f"loaded {args.checkpoint} (step {ckpt.get('step', '?')})")

    def run(prompt: str) -> None:
        for i in range(args.num_samples):
            torch.manual_seed(args.seed + i)
            text = generate_text(model, enc, prompt, device, args.max_new_tokens, cfg.model.context_length,
                                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                 precision=args.precision)
            header = f"--- sample {i + 1}" if args.num_samples > 1 else "---"
            print(f"{header}\n{text}\n")

    if args.prompt is not None:
        run(args.prompt)
        return
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        run(prompt)


if __name__ == "__main__":
    main()
