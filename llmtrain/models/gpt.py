"""GPT-2 style decoder (chapters 3-4 of the reference book).

Parameter names match the book's GPTModel (tok_emb, pos_emb, trf_blocks.N.att.W_query,
...), so weights loaded with the book's GPT-2 loaders map onto this model. The only
change in the math is that attention uses PyTorch's fused scaled_dot_product_attention
(FlashAttention on A100) instead of an explicit mask + softmax.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.scale, self.shift, self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.emb_dim
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.W_query = nn.Linear(d, d, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(d, d, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(d, d, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(d, d)
        self.dropout_p = cfg.drop_rate

    def forward(self, x):
        b, t, d = x.shape
        # (b, t, d) -> (b, n_heads, t, head_dim)
        q = self.W_query(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        if cfg.init_std is not None:  # None keeps PyTorch default init, as in the book
            self._init_all()
        if cfg.tie_embeddings:
            self.out_head.weight = self.tok_emb.weight

    def _init_all(self):
        self.apply(self._init_weights)
        # GPT-2: scale the residual-stream output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("att.out_proj.weight") or name.endswith("ff.layers.2.weight"):
                nn.init.normal_(p, mean=0.0, std=self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context_length {self.cfg.context_length}")
        pos = torch.arange(seq_len, device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(pos))
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
