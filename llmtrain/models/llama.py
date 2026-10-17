"""Llama 3 style decoder (reference: pkg/llms_from_scratch/llama3.py).

Pre-norm RMSNorm, rotary position embeddings (split-halves style, as in the
reference and Hugging Face), grouped-query attention and a SwiGLU feed-forward.
RoPE frequency scaling is omitted: it only matters when extending a pretrained
model's context, not when training from scratch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope_params(head_dim: int, context_length: int, theta_base: float):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    positions = torch.arange(context_length, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]          # (context_length, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)             # (context_length, head_dim)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim)
    seq_len, head_dim = x.shape[2], x.shape[3]
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos, sin = cos[:seq_len][None, None], sin[:seq_len][None, None]
    return (x * cos + rotated * sin).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.emb_dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = d // cfg.n_heads
        self.W_query = nn.Linear(d, d, bias=False)
        self.W_key = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout_p = cfg.drop_rate

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q = self.W_query(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # enable_gqa shares each K/V head across n_heads / n_kv_heads query heads
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_kv_heads != self.n_heads,
            dropout_p=self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg.emb_dim, eps=cfg.norm_eps)
        self.norm2 = nn.RMSNorm(cfg.emb_dim, eps=cfg.norm_eps)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x, cos, sin):
        x = x + self.drop_shortcut(self.att(self.norm1(x), cos, sin))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.emb_dim, eps=cfg.norm_eps)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        cos, sin = compute_rope_params(cfg.emb_dim // cfg.n_heads, cfg.context_length, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        if cfg.init_std is not None:  # None keeps PyTorch default init, as in the book
            self._init_all()
        if cfg.tie_embeddings:
            self.out_head.weight = self.tok_emb.weight

    def _init_all(self):
        self.apply(self._init_weights)
        # GPT-2: scale the residual-stream output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("att.out_proj.weight") or name.endswith("ff.fc3.weight"):
                nn.init.normal_(p, mean=0.0, std=self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context_length {self.cfg.context_length}")
        x = self.drop_emb(self.tok_emb(in_idx))
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        return self.out_head(x)
