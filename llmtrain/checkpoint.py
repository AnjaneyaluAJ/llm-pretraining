"""Saving and loading checkpoints (model, optimizer, progress, config, data metadata)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from llmtrain.config import Config, config_from_dict
from llmtrain.models import build_model


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DistributedDataParallel and torch.compile wrappers."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(path: Union[str, Path], model: nn.Module, cfg: Config, meta: dict,
                    optimizer: Optional[torch.optim.Optimizer] = None, **state) -> None:
    """Atomic write: a crash mid-save never leaves a truncated checkpoint behind."""
    ckpt = {
        "model": unwrap(model).state_dict(),
        "config": cfg.to_dict(),
        "data_meta": meta,
        **state,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Union[str, Path], map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(path: Union[str, Path], device: torch.device):
    """Rebuild the model from a checkpoint; returns (model, cfg, data_meta, ckpt)."""
    ckpt = load_checkpoint(path)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt["data_meta"], ckpt
