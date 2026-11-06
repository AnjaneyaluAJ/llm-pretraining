"""Causal self-attention kernels, selectable with model.attn_impl.

"sdpa"   PyTorch's fused scaled_dot_product_attention (FlashAttention / memory-efficient
         kernels when the GPU and dtype allow it). The default.
"manual" The textbook version from the reference book: build the full (t x t) score
         matrix, mask the future, softmax, multiply by V. Same math, kept for the
         performance ablation, since it stores the whole score matrix for backward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ATTN_IMPLS = ("sdpa", "manual")


def causal_attention(q, k, v, impl: str = "sdpa", dropout_p: float = 0.0, enable_gqa: bool = False):
    """q: (b, h, t, hd); k, v: (b, h_kv, t, hd). Returns (b, h, t, hd)."""
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p,
                                              enable_gqa=enable_gqa)
    if impl != "manual":
        raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {impl!r}")
    if enable_gqa and k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    t = q.shape[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v
