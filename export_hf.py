"""Export a GPT-style checkpoint to Hugging Face transformers format (GPT2LMHeadModel).

The result loads with plain transformers, no need for this repo:
  python export_hf.py --checkpoint runs/my_run/ckpt_best.pt --out hf/my-gpt2
  python -c "from transformers import pipeline; print(pipeline('text-generation', 'hf/my-gpt2')('Hello'))"

The export is checked by comparing logits of both models on a sample text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llmtrain.checkpoint import load_model
from llmtrain.tokenizer import get_tokenizer


def to_hf_state_dict(sd: dict, n_layers: int, qkv_bias: bool) -> dict:
    """Book GPTModel names -> GPT2LMHeadModel names. HF stores linear weights as (in, out)."""
    out = {
        "transformer.wte.weight": sd["tok_emb.weight"],
        "transformer.wpe.weight": sd["pos_emb.weight"],
        "transformer.ln_f.weight": sd["final_norm.scale"],
        "transformer.ln_f.bias": sd["final_norm.shift"],
        "lm_head.weight": sd["out_head.weight"],
    }
    for i in range(n_layers):
        b, h = f"trf_blocks.{i}.", f"transformer.h.{i}."
        qkv = [sd[b + f"att.W_{n}.weight"] for n in ("query", "key", "value")]
        out[h + "attn.c_attn.weight"] = torch.cat(qkv, dim=0).t()
        out[h + "attn.c_attn.bias"] = (torch.cat([sd[b + f"att.W_{n}.bias"] for n in ("query", "key", "value")])
                                       if qkv_bias else torch.zeros(3 * qkv[0].shape[0]))
        out[h + "attn.c_proj.weight"] = sd[b + "att.out_proj.weight"].t()
        out[h + "attn.c_proj.bias"] = sd[b + "att.out_proj.bias"]
        out[h + "mlp.c_fc.weight"] = sd[b + "ff.layers.0.weight"].t()
        out[h + "mlp.c_fc.bias"] = sd[b + "ff.layers.0.bias"]
        out[h + "mlp.c_proj.weight"] = sd[b + "ff.layers.2.weight"].t()
        out[h + "mlp.c_proj.bias"] = sd[b + "ff.layers.2.bias"]
        out[h + "ln_1.weight"] = sd[b + "norm1.scale"]
        out[h + "ln_1.bias"] = sd[b + "norm1.shift"]
        out[h + "ln_2.weight"] = sd[b + "norm2.scale"]
        out[h + "ln_2.bias"] = sd[b + "norm2.shift"]
    return {k: v.contiguous() for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    model, cfg, meta, ckpt = load_model(args.checkpoint, torch.device("cpu"))
    mc = cfg.model
    if mc.type != "gpt":
        raise SystemExit("only GPT-2 style models (model.type=gpt) can be exported to GPT2LMHeadModel")
    if meta["tokenizer"] != "gpt2":
        raise SystemExit(f"the model uses the {meta['tokenizer']} tokenizer; GPT2LMHeadModel export needs gpt2")

    hf_cfg = GPT2Config(vocab_size=mc.vocab_size, n_positions=mc.context_length, n_embd=mc.emb_dim,
                        n_layer=mc.n_layers, n_head=mc.n_heads, activation_function="gelu_new",
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5,
                        tie_word_embeddings=mc.tie_embeddings, bos_token_id=meta["eot_token"],
                        eos_token_id=meta["eot_token"])
    hf = GPT2LMHeadModel(hf_cfg)
    missing, unexpected = hf.load_state_dict(to_hf_state_dict(model.state_dict(), mc.n_layers, mc.qkv_bias),
                                             strict=False)
    missing = [k for k in missing if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")]
    if missing or unexpected:
        raise SystemExit(f"weight mapping failed: missing {missing}, unexpected {unexpected}")
    hf.eval()

    enc = get_tokenizer("gpt2")
    ids = torch.tensor([enc.encode("Photosynthesis is the process by which plants convert sunlight into")])
    with torch.no_grad():
        ref, got = model(ids).float(), hf(ids).logits.float()
    diff = (ref - got).abs().max().item()
    print(f"max logit difference vs the original model: {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("exported model does not match the original")

    out = Path(args.out)
    hf.save_pretrained(out)  # safetensors
    GPT2TokenizerFast.from_pretrained("openai-community/gpt2").save_pretrained(out)
    print(f"saved {out} (step {ckpt.get('step')}, {ckpt.get('tokens_seen', 0):,} tokens seen)")


if __name__ == "__main__":
    main()
