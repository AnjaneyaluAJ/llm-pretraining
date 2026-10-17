"""Instruction data (book chapter 7): Alpaca prompt format, dataset, and padding/masking collate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from llmtrain.evaluation import IGNORE_INDEX
from llmtrain.tokenizer import encode

RESPONSE_HEADER = "\n\n### Response:\n"


def format_input(entry: dict) -> str:
    """Alpaca-style prompt: instruction plus optional input (without the response header)."""
    text = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):
        text += f"\n\n### Input:\n{entry['input']}"
    return text


def format_prompt(entry: dict) -> str:
    """What the model sees at inference: the prompt up to and including the response header."""
    return format_input(entry) + RESPONSE_HEADER


def load_entries(path: str) -> list[dict]:
    """A JSON list or JSONL file of {"instruction", "input" (optional), "output"} records."""
    text = Path(path).read_text(encoding="utf-8")
    entries = json.loads(text) if text.lstrip().startswith("[") else \
        [json.loads(line) for line in text.splitlines() if line.strip()]
    for i, e in enumerate(entries):
        if "instruction" not in e or "output" not in e:
            raise KeyError(f"{path} record {i} needs 'instruction' and 'output' fields, got {sorted(e)}")
    return entries


def split_entries(entries: list[dict], train_frac: float = 0.85, test_frac: float = 0.10):
    """The book's split, in file order: 85% train, 10% test, the remaining 5% validation."""
    n_train = int(len(entries) * train_frac)
    n_test = int(len(entries) * test_frac)
    return entries[:n_train], entries[n_train + n_test:], entries[n_train:n_train + n_test]


class InstructionDataset(Dataset):
    """Pre-tokenized prompt + response; remembers the prompt length for optional loss masking."""

    def __init__(self, entries: list[dict], enc):
        self.items = []
        for e in entries:
            prompt_ids = encode(enc, format_prompt(e))
            response_ids = encode(enc, e["output"])
            self.items.append((prompt_ids + response_ids, len(prompt_ids)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate(batch, pad_id: int, max_length: int, mask_prompt: bool = False):
    """Pad to the longest sequence in the batch.

    One <|endoftext|> is appended and kept as a target so the model learns to stop; further
    padding targets are IGNORE_INDEX. With mask_prompt, the prompt tokens are not trained on
    either, so the loss covers only the response.
    """
    seq_len = min(max(len(ids) + 1 for ids, _ in batch), max_length + 1)
    inputs, targets = [], []
    for ids, prompt_len in batch:
        ids = (ids + [pad_id])[:seq_len]
        n_real = len(ids)
        padded = torch.tensor(ids + [pad_id] * (seq_len - n_real))
        x, y = padded[:-1].clone(), padded[1:].clone()
        y[n_real - 1:] = IGNORE_INDEX  # padding beyond the single end-of-text target
        if mask_prompt:
            y[:prompt_len - 1] = IGNORE_INDEX
        inputs.append(x)
        targets.append(y)
    return torch.stack(inputs), torch.stack(targets)
