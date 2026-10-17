"""Thin wrapper over torch.distributed so the same code runs under torchrun or plain python."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Initialise the process group when launched by torchrun, else run single-process."""
    if "RANK" not in os.environ:
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return DistInfo(device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    return DistInfo(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t
