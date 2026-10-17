"""Tokenize raw text into train.bin / val.bin / meta.json for train.py.

Inputs can be any mix of files, directories (searched recursively) and globs:
  .txt .md .text .txt.utf8      one document per file
  .jsonl .jsonl.gz .json        one document per record, text taken from --text_field
  .csv                          one document per row, text taken from --text_field
  .parquet                      one document per row (needs pyarrow)
or a Hugging Face dataset with --hf_dataset (needs `pip install datasets`).

Documents are joined with <|endoftext|>. Unless --val_input is given, the last
--val_fraction of the token stream becomes the validation split.

Examples:
  python prepare_data.py --input data/raw/the-verdict.txt --output_dir data/verdict
  python prepare_data.py --input /mnt/nvme/books/ --output_dir /mnt/nvme/tok/books \
      --val_fraction 0.005 --num_proc 64
  python prepare_data.py --input corpus.jsonl --text_field content --output_dir data/corpus
  python prepare_data.py --hf_dataset HuggingFaceFW/fineweb-edu --hf_config sample-10BT \
      --output_dir /mnt/nvme/tok/fineweb-edu-10bt --val_fraction 0.001 --num_proc 64
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llmtrain.tokenizer import encode, get_tokenizer, token_dtype

TEXT_SUFFIXES = (".txt", ".md", ".text", ".txt.utf8")
RECORD_SUFFIXES = (".jsonl", ".jsonl.gz", ".json", ".csv", ".parquet")
TEXT_CHUNK_CHARS = 4_000_000  # large text files are tokenized in pieces cut at newlines


# ----------------------------------------------------------------------------- input discovery

def find_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and f.name.lower().endswith(TEXT_SUFFIXES + RECORD_SUFFIXES))
        elif p.is_file():
            files.append(p)
        elif glob.has_magic(item):
            files += sorted(Path(m) for m in glob.glob(item, recursive=True) if Path(m).is_file())
        else:
            raise FileNotFoundError(f"Input not found: {item}")
    if not files:
        raise FileNotFoundError(f"No supported files found in {inputs}")
    return files


# ----------------------------------------------------------------------------- readers
# Each reader yields (text, is_end_of_document) pieces.

def read_text_file(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = ""
        while block := f.read(TEXT_CHUNK_CHARS):
            buf += block
            cut = buf.rfind("\n")
            if cut != -1:
                yield buf[:cut + 1], False
                buf = buf[cut + 1:]
            elif len(buf) > 4 * TEXT_CHUNK_CHARS:  # pathological file with no newlines
                yield buf, False
                buf = ""
        yield buf, True


def _field(record, text_field: str, path: Path):
    if isinstance(record, str):
        return record
    if text_field not in record:
        raise KeyError(f"{path}: record has no field {text_field!r} (fields: {sorted(record)}); "
                       f"set --text_field")
    return record[text_field]


def read_record_file(path: Path, text_field: str):
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield _field(json.loads(line), text_field, path)
    elif name.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        for record in records:
            yield _field(record, text_field, path)
    elif name.endswith(".csv"):
        import pandas as pd
        for chunk in pd.read_csv(path, usecols=[text_field], chunksize=10_000):
            yield from chunk[text_field].astype(str)
    elif name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            sys.exit("Reading .parquet needs pyarrow: uv pip install pyarrow")
        for batch in pq.ParquetFile(path).iter_batches(columns=[text_field], batch_size=10_000):
            yield from batch.column(0).to_pylist()


def iter_pieces(files: list[Path], text_field: str, max_docs: int | None):
    docs = 0
    for path in files:
        if path.name.lower().endswith(TEXT_SUFFIXES):
            yield from read_text_file(path)
            docs += 1
        else:
            for text in read_record_file(path, text_field):
                if text:
                    yield text, True
                    docs += 1
                    if max_docs and docs >= max_docs:
                        return
        if max_docs and docs >= max_docs:
            return


def iter_hf_pieces(name: str, config: str | None, split: str, text_field: str, max_docs: int | None):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("--hf_dataset needs the datasets package: uv pip install datasets")
    ds = load_dataset(name, config, split=split, streaming=True)
    for i, example in enumerate(ds):
        if max_docs and i >= max_docs:
            return
        text = example[text_field]
        if text:
            yield text, True


# ----------------------------------------------------------------------------- tokenization

_ENC = None
_DTYPE = None


def _init_worker(tokenizer_name: str, dtype: str) -> None:
    global _ENC, _DTYPE
    _ENC = get_tokenizer(tokenizer_name)
    _DTYPE = np.dtype(dtype)


def _encode(piece):
    text, end_of_doc = piece
    ids = encode(_ENC, text)
    if end_of_doc:
        ids.append(_ENC.eot_token)
    return np.asarray(ids, dtype=_DTYPE), len(text.encode("utf-8")), end_of_doc


def tokenize_to_file(pieces, out_path: Path, tokenizer: str, dtype: np.dtype, num_proc: int,
                     desc: str) -> dict:
    stats = {"tokens": 0, "documents": 0, "bytes": 0}
    pool = None
    if num_proc > 1:
        pool = Pool(num_proc, initializer=_init_worker, initargs=(tokenizer, dtype.name))
        results = pool.imap(_encode, pieces, chunksize=16)
    else:
        _init_worker(tokenizer, dtype.name)
        results = map(_encode, pieces)
    try:
        with open(out_path, "wb") as f, tqdm(desc=desc, unit=" tok", unit_scale=True) as bar:
            for ids, n_bytes, end_of_doc in results:
                f.write(ids.tobytes())
                stats["tokens"] += len(ids)
                stats["bytes"] += n_bytes
                stats["documents"] += int(end_of_doc)
                bar.update(len(ids))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return stats


def split_tail(train_path: Path, val_path: Path, dtype: np.dtype, n_val: int) -> None:
    """Move the last n_val tokens of train_path into val_path, in place."""
    n_total = os.path.getsize(train_path) // dtype.itemsize
    tokens = np.memmap(train_path, dtype=dtype, mode="r")
    with open(val_path, "wb") as f:
        step = 64 * 2**20
        for start in range(n_total - n_val, n_total, step):
            f.write(np.asarray(tokens[start:min(start + step, n_total)]).tobytes())
    del tokens
    os.truncate(train_path, (n_total - n_val) * dtype.itemsize)


# ----------------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--input", nargs="+", help="files, directories or globs")
    src.add_argument("--hf_dataset", help="Hugging Face dataset name (streamed)")
    src.add_argument("--hf_config", default=None, help="Hugging Face dataset config / subset")
    src.add_argument("--hf_split", default="train")
    src.add_argument("--text_field", default="text", help="text column for jsonl/json/csv/parquet/HF")
    src.add_argument("--max_docs", type=int, default=None, help="stop after this many documents")
    out = p.add_argument_group("output")
    out.add_argument("--output_dir", required=True)
    out.add_argument("--tokenizer", default="gpt2", help="tiktoken encoding name")
    out.add_argument("--val_fraction", type=float, default=0.1,
                     help="fraction of tokens (taken from the end) used for validation")
    out.add_argument("--val_input", nargs="+", default=None,
                     help="separate validation files; overrides --val_fraction")
    out.add_argument("--num_proc", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    out.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if bool(args.input) == bool(args.hf_dataset):
        p.error("give exactly one of --input or --hf_dataset")
    if not 0.0 <= args.val_fraction < 1.0:
        p.error("--val_fraction must be in [0, 1)")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    train_path, val_path, meta_path = out_dir / "train.bin", out_dir / "val.bin", out_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        sys.exit(f"{out_dir} already holds a prepared dataset; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = get_tokenizer(args.tokenizer)
    dtype = token_dtype(enc.n_vocab)
    t0 = time.time()

    if args.hf_dataset:
        sources = [f"hf:{args.hf_dataset}" + (f"/{args.hf_config}" if args.hf_config else "")
                   + f"[{args.hf_split}]"]
        pieces = iter_hf_pieces(args.hf_dataset, args.hf_config, args.hf_split, args.text_field,
                                args.max_docs)
    else:
        files = find_files(args.input)
        sources = [str(f) for f in files]
        print(f"{len(files)} input file(s)")
        pieces = iter_pieces(files, args.text_field, args.max_docs)

    stats = tokenize_to_file(pieces, train_path, args.tokenizer, dtype, args.num_proc, "train")

    if args.val_input:
        val_files = find_files(args.val_input)
        val_stats = tokenize_to_file(iter_pieces(val_files, args.text_field, None), val_path,
                                     args.tokenizer, dtype, args.num_proc, "val")
        n_train, n_val = stats["tokens"], val_stats["tokens"]
        stats = {k: stats[k] + val_stats[k] for k in stats}
    else:
        n_val = int(round(stats["tokens"] * args.val_fraction))
        split_tail(train_path, val_path, dtype, n_val)
        n_train = stats["tokens"] - n_val

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": dtype.name,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "documents": stats["documents"],
        "bytes": stats["bytes"],
        "bytes_per_token": stats["bytes"] / max(1, stats["tokens"]),
        "sources": sources if len(sources) <= 100 else sources[:100] + [f"... {len(sources) - 100} more"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{stats['documents']:,} documents, {stats['bytes'] / 1e6:,.1f} MB of text "
          f"-> {stats['tokens']:,} tokens ({meta['bytes_per_token']:.2f} bytes/token) "
          f"in {time.time() - t0:.1f}s")
    print(f"train: {n_train:,} tokens  {train_path}")
    print(f"val:   {n_val:,} tokens  {val_path}")
    if n_val == 0:
        print("warning: empty validation split -- validation metrics will be skipped")


if __name__ == "__main__":
    main()
