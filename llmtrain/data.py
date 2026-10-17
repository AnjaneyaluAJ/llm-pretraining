"""Datasets over the flat token files written by prepare_data.py.

Each split is one contiguous array of token ids (train.bin / val.bin). A sample is
a window of context_length tokens plus the next token, split into (input, target)
with target shifted one position -- the same sliding-window scheme as the book's
GPTDatasetV1, but memory-mapped so the data can be far larger than RAM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


def load_meta(data_dir: Union[str, Path]) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run prepare_data.py first")
    with open(path) as f:
        return json.load(f)


class TokenWindowDataset(Dataset):
    def __init__(self, source: Union[str, Path, np.ndarray], context_length: int,
                 stride: Optional[int] = None, dtype: Union[str, np.dtype] = np.uint16):
        self.context_length = context_length
        self.stride = stride or context_length
        if isinstance(source, np.ndarray):
            self._tokens = source
            self.path = None
            n_tokens = len(source)
        else:
            self.path = str(source)
            self.dtype = np.dtype(dtype)
            self._tokens = None  # opened lazily so each DataLoader worker maps its own copy
            n_tokens = os.path.getsize(self.path) // self.dtype.itemsize
        self.n_tokens = n_tokens
        # a window needs context_length + 1 tokens (inputs plus the shifted target)
        if n_tokens > context_length:
            self.n_windows = (n_tokens - context_length - 1) // self.stride + 1
        else:
            self.n_windows = 0

    def _array(self) -> np.ndarray:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        chunk = self._array()[start:start + self.context_length + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        return chunk[:-1], chunk[1:]


class ShardSampler(Sampler):
    """Deterministic rank::world_size split with no padding, for exact evaluation."""

    def __init__(self, dataset: Dataset, rank: int = 0, world_size: int = 1):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def split_dataset(data_dir: Union[str, Path], split: str, context_length: int,
                  stride: Optional[int] = None) -> TokenWindowDataset:
    meta = load_meta(data_dir)
    return TokenWindowDataset(Path(data_dir) / f"{split}.bin", context_length, stride, meta["dtype"])


def train_loader(dataset: TokenWindowDataset, batch_size: int, rank: int, world_size: int,
                 seed: int, num_workers: int) -> DataLoader:
    # DistributedSampler pads so every rank sees the same number of batches (required by DDP)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=num_workers > 0)


def eval_loader(dataset: TokenWindowDataset, batch_size: int, rank: int, world_size: int,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, sampler=ShardSampler(dataset, rank, world_size),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
