"""Typed configuration loaded from YAML, with command-line overrides.

A config file has three sections -- model, data, train -- that map onto the
dataclasses below. Any field can be overridden from the command line:

    python train.py --config configs/gpt2_verdict.yaml --set train.lr=3e-4 model.n_layers=6
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    preset: Optional[str] = None        # e.g. "gpt2-124m", "llama-tiny"; fields below override it
    type: str = "gpt"                   # "gpt" (GPT-2 style) or "llama" (RMSNorm, RoPE, GQA, SwiGLU)
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    tie_embeddings: bool = False
    init_std: Optional[float] = 0.02    # null = PyTorch default init (the book; embeddings ~ N(0, 1))
    # llama-only
    n_kv_heads: Optional[int] = None    # defaults to n_heads (plain multi-head attention)
    hidden_dim: Optional[int] = None    # SwiGLU width; defaults to ~8/3 * emb_dim
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5


@dataclass
class DataConfig:
    data_dir: str = "data/verdict"      # output of prepare_data.py: train.bin, val.bin, meta.json
    stride: Optional[int] = None        # window step in tokens; defaults to context_length (no overlap)
    num_workers: int = 2


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    seed: int = 123
    # batch
    batch_size: int = 2                 # sequences per GPU per micro-step
    grad_accum_steps: int = 1
    # length: set exactly one of epochs / max_steps
    epochs: Optional[int] = 10
    max_steps: Optional[int] = None
    # optimizer
    lr: float = 5e-4
    min_lr_ratio: float = 0.1           # cosine decays to lr * min_lr_ratio
    lr_schedule: str = "cosine"         # "cosine" or "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: Optional[float] = 1.0    # null disables clipping
    # performance
    precision: str = "bf16"             # "bf16" (autocast) or "fp32"
    compile: bool = False
    peak_flops: float = 312e12          # per-GPU peak used for MFU; A100 bf16 dense = 312 TFLOP/s
    # evaluation and logging
    log_every: int = 1
    eval_every: int = 5
    eval_batches: Optional[int] = None  # per-GPU batches per eval; null = whole split
    sample_every: int = 50
    sample_prompt: str = "Every effort moves you"
    sample_tokens: int = 50
    sample_temperature: float = 0.0     # 0 = greedy
    sample_top_k: Optional[int] = None
    hellaswag_every: Optional[int] = None   # steps between HellaSwag evals; null disables
    hellaswag_limit: Optional[int] = None   # examples per HellaSwag eval; null = all 10,042
    tensorboard: bool = True
    # checkpoints
    save_every: Optional[int] = None    # steps; final checkpoint is always saved
    resume: Optional[str] = None        # path to a checkpoint, or "auto" for out_dir/ckpt_last.pt


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, values: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**values)


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' strings; values are parsed as YAML (numbers, bools, null)."""
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        raw.setdefault(section, {})[name] = yaml.safe_load(value)
    return raw


def load_config(path: Optional[str] = None, overrides: Optional[list[str]] = None) -> Config:
    raw: dict[str, Any] = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    raw = apply_overrides(raw, overrides or [])
    unknown = set(raw) - {"model", "data", "train"}
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}")

    from llmtrain.models import resolve_model_config  # local import avoids a cycle
    model = resolve_model_config(raw.get("model", {}))
    cfg = Config(model=model,
                 data=_build(DataConfig, raw.get("data", {})),
                 train=_build(TrainConfig, raw.get("train", {})))
    if (cfg.train.epochs is None) == (cfg.train.max_steps is None):
        raise ValueError("Set exactly one of train.epochs or train.max_steps")
    return cfg


def config_from_dict(d: dict) -> Config:
    """Rebuild a Config saved inside a checkpoint."""
    return Config(model=_build(ModelConfig, d["model"]),
                  data=_build(DataConfig, d["data"]),
                  train=_build(TrainConfig, d["train"]))


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
