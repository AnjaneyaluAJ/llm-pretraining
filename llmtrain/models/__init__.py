"""Model registry: architecture presets and a single build_model() entry point."""

from __future__ import annotations

import dataclasses

import torch.nn as nn

from llmtrain.config import ModelConfig
from llmtrain.models.gpt import GPTModel
from llmtrain.models.llama import LlamaModel

# Presets fill in architecture fields; anything set explicitly in the config wins.
# The Llama presets keep the Llama 3.2 shapes but use whatever tokenizer the data
# was prepared with (vocab_size comes from the config / data, not from Meta's 128k vocab).
PRESETS: dict[str, dict] = {
    "gpt2-124m":  dict(type="gpt", context_length=1024, emb_dim=768,  n_heads=12, n_layers=12),
    "gpt2-355m":  dict(type="gpt", context_length=1024, emb_dim=1024, n_heads=16, n_layers=24),
    "gpt2-774m":  dict(type="gpt", context_length=1024, emb_dim=1280, n_heads=20, n_layers=36),
    "gpt2-1558m": dict(type="gpt", context_length=1024, emb_dim=1600, n_heads=25, n_layers=48),
    "llama-tiny": dict(type="llama", context_length=512, emb_dim=512, n_heads=8, n_kv_heads=4,
                       n_layers=8, drop_rate=0.0, tie_embeddings=True),
    "llama-1b":   dict(type="llama", context_length=2048, emb_dim=2048, n_heads=32, n_kv_heads=8,
                       n_layers=16, hidden_dim=8192, rope_theta=500_000.0, drop_rate=0.0,
                       tie_embeddings=True),
    "llama-3b":   dict(type="llama", context_length=2048, emb_dim=3072, n_heads=24, n_kv_heads=8,
                       n_layers=28, hidden_dim=8192, rope_theta=500_000.0, drop_rate=0.0,
                       tie_embeddings=True),
}

MODEL_TYPES = {"gpt": GPTModel, "llama": LlamaModel}


def resolve_model_config(raw: dict) -> ModelConfig:
    """Merge preset defaults with explicit fields and fill derived Llama sizes."""
    raw = dict(raw)
    values: dict = {}
    preset = raw.get("preset")
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown model preset {preset!r}; choose from {sorted(PRESETS)}")
        values.update(PRESETS[preset])
    values.update(raw)

    names = {f.name for f in dataclasses.fields(ModelConfig)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown ModelConfig field(s): {sorted(unknown)}")
    cfg = ModelConfig(**values)

    if cfg.type not in MODEL_TYPES:
        raise ValueError(f"model.type must be one of {sorted(MODEL_TYPES)}, got {cfg.type!r}")
    if cfg.emb_dim % cfg.n_heads:
        raise ValueError("model.emb_dim must be divisible by model.n_heads")
    if cfg.type == "llama":
        if cfg.n_kv_heads is None:
            cfg.n_kv_heads = cfg.n_heads
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads")
        if cfg.hidden_dim is None:
            # Llama rule: 2/3 of 4*d, rounded up to a multiple of 256
            cfg.hidden_dim = 256 * ((int(8 * cfg.emb_dim / 3) + 255) // 256)
    return cfg


def build_model(cfg: ModelConfig) -> nn.Module:
    return MODEL_TYPES[cfg.type](cfg)


def count_params(model: nn.Module) -> int:
    """Unique parameters (tied embeddings counted once)."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def flops_per_token(cfg: ModelConfig, n_params: int) -> float:
    """Training FLOPs per token: 6N for the weights plus the attention score/value matmuls."""
    return 6 * n_params + 12 * cfg.n_layers * cfg.emb_dim * cfg.context_length
