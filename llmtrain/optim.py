"""AdamW with decoupled weight decay on matrices only, and the learning-rate schedule."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from llmtrain.config import TrainConfig


def build_optimizer(model: nn.Module, tc: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    # weight-decay matrices (linear layers, embeddings); never biases or norm gains
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                             fused=device.type == "cuda")


def lr_at(step: int, tc: TrainConfig, total_steps: int) -> float:
    """Linear warmup, then constant or cosine decay to lr * min_lr_ratio."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if tc.lr_schedule == "constant":
        return tc.lr
    if tc.lr_schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule {tc.lr_schedule!r}")
    progress = (step - tc.warmup_steps) / max(1, total_steps - tc.warmup_steps)
    progress = min(1.0, progress)
    min_lr = tc.lr * tc.min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (tc.lr - min_lr)
