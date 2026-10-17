"""Evaluation: next-token loss, perplexity, accuracy and bits-per-byte over a token split."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmtrain.dist import all_reduce_sum

IGNORE_INDEX = -100  # target value excluded from loss and metrics (padding, masked prompts)


def autocast_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    # loss in fp32 even under bf16 autocast, for a stable log-softmax over the vocab
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction=reduction,
                           ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, precision: str = "bf16",
             max_batches: Optional[int] = None, bytes_per_token: Optional[float] = None) -> dict:
    """Average over every token of the split, summed across ranks.

    Pass the plain (non-DDP) model: ranks may hold different numbers of eval batches.
    """
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)  # nll sum, tokens, correct
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = model(x)
        valid = y != IGNORE_INDEX
        totals[0] += lm_loss(logits, y, reduction="sum").double()
        totals[1] += valid.sum()
        totals[2] += ((logits.argmax(dim=-1) == y) & valid).sum().double()
    all_reduce_sum(totals)
    model.train(was_training)

    nll, n_tokens, correct = totals.tolist()
    if n_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "accuracy": float("nan"), "tokens": 0}
    loss = nll / n_tokens
    out = {
        "loss": loss,
        "perplexity": math.exp(min(loss, 700.0)),
        "accuracy": correct / n_tokens,
        "tokens": int(n_tokens),
    }
    if bytes_per_token:
        # tokenizer-independent: nats/token -> bits/byte
        out["bits_per_byte"] = loss / math.log(2) / bytes_per_token
    return out
