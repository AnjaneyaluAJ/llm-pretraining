"""Metrics: console, metrics.jsonl, optional TensorBoard, and a loss plot at the end."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class MetricsLogger:
    """Only instantiate on rank 0."""

    def __init__(self, out_dir: str, tensorboard: bool = True):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / "metrics.jsonl"
        self.writer = None
        if tensorboard:
            # The event writer does not need TensorFlow, but tensorboard imports it when installed
            # (it is, via requirements.txt), which prints CUDA start-up noise. Block that import.
            sys.modules.setdefault("tensorflow", None)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.out_dir / "tb"))
            except ImportError:
                print("tensorboard not installed; logging to metrics.jsonl only")

    def log(self, step: int, **metrics) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def plot_losses(metrics_path: Path, out_path: Path) -> None:
    """Train/val loss against steps and tokens seen, like the book's plot_losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = [], []
    with open(metrics_path) as f:
        for line in f:
            r = json.loads(line)
            if "train/step_loss" in r:
                train.append((r["step"], r["tokens_seen"], r["train/step_loss"]))
            if "val/loss" in r:
                val.append((r["step"], r["tokens_seen"], r["val/loss"]))
    if not train:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot([s for s, _, _ in train], [v for _, _, v in train], label="Training loss")
    if val:
        ax1.plot([s for s, _, _ in val], [v for _, _, v in val], linestyle="-.", marker="o",
                 markersize=3, label="Validation loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot([t for _, t, _ in train], [v for _, _, v in train], alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
