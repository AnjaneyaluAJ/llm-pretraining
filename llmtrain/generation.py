"""Autoregressive sampling (greedy, temperature, top-k, top-p). No KV cache: simple and exact."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from llmtrain.evaluation import autocast_context


@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_length: int,
             temperature: float = 0.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
             eos_id: Optional[int] = None, precision: str = "bf16") -> torch.Tensor:
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with autocast_context(idx.device, precision):
            logits = model(idx_cond)
        logits = logits[:, -1, :].float()

        if temperature > 0:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # drop tokens once the cumulative mass before them already exceeds top_p
                drop = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and bool((next_id == eos_id).all()):
            break
    model.train(was_training)
    return idx


def generate_text(model: nn.Module, enc, prompt: str, device: torch.device, max_new_tokens: int,
                  context_length: int, **kwargs) -> str:
    prompt_ids = enc.encode_ordinary(prompt) or [enc.eot_token]  # empty prompt: start a new document
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = generate(model, ids, max_new_tokens, context_length, eos_id=enc.eot_token, **kwargs)
    return enc.decode(out[0].tolist())
