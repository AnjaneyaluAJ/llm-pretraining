"""Download OpenAI's pretrained GPT-2 and save it as a checkpoint for this repo.

The result works with evaluate.py, generate.py and finetune.py like any other checkpoint:
  python import_gpt2.py --size 355M                     # -> pretrained/gpt2-355m.pt
  python import_gpt2.py --size 124M --out pretrained/gpt2-124m.pt
Sizes: 124M, 355M, 774M, 1558M.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llmtrain.checkpoint import save_checkpoint
from llmtrain.config import Config
from llmtrain.generation import generate_text
from llmtrain.gpt2_weights import GPT2_PRESETS, download_gpt2, gpt2_config, load_into_gpt, load_tf_params
from llmtrain.models import build_model, count_params
from llmtrain.tokenizer import get_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="355M", choices=sorted(GPT2_PRESETS))
    p.add_argument("--models_dir", default="pretrained/openai", help="where the TF checkpoints are downloaded")
    p.add_argument("--out", default=None, help="default: pretrained/gpt2-<size>.pt")
    args = p.parse_args()
    out = Path(args.out or f"pretrained/{GPT2_PRESETS[args.size]}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    model_dir = download_gpt2(args.size, args.models_dir)
    print(f"reading TensorFlow checkpoint in {model_dir}")
    params = load_tf_params(model_dir)

    mc = gpt2_config(args.size)
    model = build_model(mc)
    load_into_gpt(model, params)
    print(f"loaded GPT-2 {args.size}: {count_params(model) / 1e6:.1f}M params")

    # sanity check: the book's chapter 5 prompt should give fluent English
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    enc = get_tokenizer("gpt2")
    print("sample:", repr(generate_text(model, enc, "Every effort moves you", device, 25, mc.context_length)))

    meta = {"tokenizer": "gpt2", "vocab_size": enc.n_vocab, "eot_token": enc.eot_token}
    save_checkpoint(out, model, Config(model=mc), meta, step=0, tokens_seen=0,
                    source=f"openai/gpt2-{args.size}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
