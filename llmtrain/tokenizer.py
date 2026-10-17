"""Tokenizer helpers. Any tiktoken encoding works: gpt2, r50k_base, cl100k_base, o200k_base."""

from __future__ import annotations

import numpy as np
import tiktoken

EOT_TEXT = "<|endoftext|>"


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= 2**16 else np.uint32)


def encode(enc: tiktoken.Encoding, text: str) -> list[int]:
    """Encode text, honouring a literal <|endoftext|> as the document separator.

    Other special-token strings are encoded as ordinary text instead of raising.
    """
    allowed = {EOT_TEXT} if EOT_TEXT in enc.special_tokens_set else set()
    return enc.encode(text, allowed_special=allowed, disallowed_special=())
