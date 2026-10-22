"""HellaSwag: pick the most plausible of 4 endings for a short context (Zellers et al., 2019).

Each ending is scored by the model's cross-entropy on the ending tokens given the
context, the standard zero-shot method used to compare GPT-2-sized models:
  acc       ending with the lowest total loss
  acc_norm  ending with the lowest mean loss per token (removes the bias toward short endings)
Random guessing scores 25%. Examples are sharded across ranks and the counts summed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmtrain.dist import all_reduce_sum
from llmtrain.evaluation import autocast_context
from llmtrain.tokenizer import encode


def load_examples(split: str = "val") -> list[dict]:
    """The official HellaSwag release, from the Hugging Face hub (needs `datasets`)."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split={"val": "validation"}.get(split, split))
    return [{"ctx": e["ctx"], "endings": e["endings"], "label": e["label"]} for e in ds]


def render(example: dict, enc, device: torch.device):
    """Return (tokens, mask, label): 4 rows of context + ending, mask = 1 on ending tokens."""
    ctx = encode(enc, example["ctx"])
    rows = [ctx + encode(enc, " " + ending) for ending in example["endings"]]
    width = max(len(r) for r in rows)
    tokens = torch.zeros((4, width), dtype=torch.long)
    mask = torch.zeros((4, width), dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.tensor(row)
        mask[i, len(ctx):len(row)] = 1
    return tokens.to(device), mask.to(device), int(example["label"])


@torch.no_grad()
def evaluate_hellaswag(model, enc, device: torch.device, precision: str = "bf16", rank: int = 0,
                       world_size: int = 1, limit: int | None = None, split: str = "val") -> dict:
    examples = load_examples(split)[:limit]
    was_training = model.training
    model.eval()
    counts = torch.zeros(3, device=device)  # correct, correct_norm, total
    for example in examples[rank::world_size]:
        tokens, mask, label = render(example, enc, device)
        tokens, mask = tokens[:, -model.cfg.context_length - 1:], mask[:, -model.cfg.context_length - 1:]
        with autocast_context(device, precision):
            logits = model(tokens[:, :-1])
        losses = F.cross_entropy(logits.float().transpose(1, 2), tokens[:, 1:], reduction="none")
        m = mask[:, 1:]
        total = (losses * m).sum(dim=1)
        counts[0] += int(total.argmin().item() == label)
        counts[1] += int((total / m.sum(dim=1)).argmin().item() == label)
        counts[2] += 1
    all_reduce_sum(counts)
    model.train(was_training)
    correct, correct_norm, n = counts.tolist()
    return {"acc": correct / n, "acc_norm": correct_norm / n, "examples": int(n)}
