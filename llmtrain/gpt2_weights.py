"""OpenAI GPT-2 weights: download the original TensorFlow checkpoints and load them into GPTModel.

Follows the book's gpt_download.py / load_weights_into_gpt (chapter 5). TensorFlow is only
used to read the checkpoint files, on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

from llmtrain.config import ModelConfig
from llmtrain.models import resolve_model_config

GPT2_PRESETS = {"124M": "gpt2-124m", "355M": "gpt2-355m", "774M": "gpt2-774m", "1558M": "gpt2-1558m"}
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
FILES = ["checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001",
         "model.ckpt.index", "model.ckpt.meta", "vocab.bpe"]


def gpt2_config(size: str) -> ModelConfig:
    """OpenAI GPT-2 architecture: biased QKV, tied input/output embeddings, no dropout for finetuning."""
    return resolve_model_config({"preset": GPT2_PRESETS[size], "qkv_bias": True,
                                 "tie_embeddings": True, "drop_rate": 0.0})


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length", 0))
        if dest.exists() and size and dest.stat().st_size == size:
            return
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        os.replace(tmp, dest)


def download_gpt2(size: str, models_dir: str) -> Path:
    if size not in GPT2_PRESETS:
        raise ValueError(f"GPT-2 size must be one of {sorted(GPT2_PRESETS)}")
    model_dir = Path(models_dir) / size
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        try:
            _download(f"{BASE_URL}/{size}/{name}", model_dir / name)
        except requests.RequestException:
            print(f"primary download failed for {name}; trying backup mirror")
            _download(f"{BACKUP_URL}/{size}/{name}", model_dir / name)
    return model_dir


def load_tf_params(model_dir: Path) -> dict:
    """Read the TF checkpoint into nested dicts of numpy arrays: {"wte", "wpe", "g", "b", "blocks": [...]}."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    settings = json.loads((model_dir / "hparams.json").read_text())
    ckpt = tf.train.latest_checkpoint(str(model_dir))
    params: dict = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt):
        array = np.squeeze(tf.train.load_variable(ckpt, name))
        parts = name.split("/")[1:]  # drop the "model/" prefix
        target = params
        if parts[0].startswith("h"):
            target = params["blocks"][int(parts[0][1:])]
        for key in parts[1:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = array
    return params


@torch.no_grad()
def load_into_gpt(model, params: dict) -> None:
    def put(param: torch.Tensor, array: np.ndarray) -> None:
        if tuple(param.shape) != array.shape:
            raise ValueError(f"shape mismatch: model {tuple(param.shape)} vs checkpoint {array.shape}")
        param.copy_(torch.from_numpy(np.ascontiguousarray(array)))

    put(model.pos_emb.weight, params["wpe"])
    put(model.tok_emb.weight, params["wte"])  # out_head shares this tensor (tied)
    for b, block in enumerate(model.trf_blocks):
        p = params["blocks"][b]
        # OpenAI fuses Q, K, V into one c_attn matrix; TF stores weights as (in, out)
        q_w, k_w, v_w = np.split(p["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(p["attn"]["c_attn"]["b"], 3, axis=-1)
        put(block.att.W_query.weight, q_w.T)
        put(block.att.W_key.weight, k_w.T)
        put(block.att.W_value.weight, v_w.T)
        put(block.att.W_query.bias, q_b)
        put(block.att.W_key.bias, k_b)
        put(block.att.W_value.bias, v_b)
        put(block.att.out_proj.weight, p["attn"]["c_proj"]["w"].T)
        put(block.att.out_proj.bias, p["attn"]["c_proj"]["b"])
        put(block.ff.layers[0].weight, p["mlp"]["c_fc"]["w"].T)
        put(block.ff.layers[0].bias, p["mlp"]["c_fc"]["b"])
        put(block.ff.layers[2].weight, p["mlp"]["c_proj"]["w"].T)
        put(block.ff.layers[2].bias, p["mlp"]["c_proj"]["b"])
        put(block.norm1.scale, p["ln_1"]["g"])
        put(block.norm1.shift, p["ln_1"]["b"])
        put(block.norm2.scale, p["ln_2"]["g"])
        put(block.norm2.shift, p["ln_2"]["b"])
    put(model.final_norm.scale, params["g"])
    put(model.final_norm.shift, params["b"])
    if model.out_head.weight is not model.tok_emb.weight:
        put(model.out_head.weight, params["wte"])
