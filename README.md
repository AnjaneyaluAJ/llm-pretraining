# llm-pretraining

Pretrain GPT-2 and Llama-style language models from scratch, on any text data, from the
console, on one GPU or all of them. Plain `.py` files, no notebooks.

The models and training recipe follow
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (chapters 2-5 and
`pkg/llms_from_scratch/llama3.py`), reorganised into a package and extended with
multi-GPU training (DDP), bf16, checkpoints and resume, evaluation, and a data pipeline
that streams from disk so datasets can be much larger than RAM.

## Layout

```
prepare_data.py        raw text / jsonl / csv / parquet / HF dataset -> train.bin, val.bin, meta.json
train.py               training (python or torchrun), config-driven
evaluate.py            loss, perplexity, accuracy, bits-per-byte (+ sample generations)
generate.py            text generation from a checkpoint (one-shot or interactive)
configs/               YAML run configs
scripts/run_verdict.sh end-to-end test on "The Verdict"
llmtrain/
  config.py            dataclass configs, YAML loading, --set overrides
  models/              gpt.py (GPT-2), llama.py (Llama 3), presets + build_model
  data.py              memory-mapped token windows, distributed samplers
  evaluation.py        distributed loss / perplexity / accuracy
  generation.py        greedy, temperature, top-k, top-p sampling
  optim.py             AdamW param groups, warmup + cosine schedule
  checkpoint.py        save / load / resume
  dist.py              torchrun setup, all-reduce helpers
  tracking.py          metrics.jsonl, TensorBoard, loss plot
  tokenizer.py         tiktoken encodings
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=cu132
source .venv/bin/activate
```

`requirements.txt` is the reference repo's file plus `pyyaml` and `tensorboard`.

## 1. Prepare data

Any mix of files, directories and globs. Documents are separated by `<|endoftext|>`; the
last `--val_fraction` of tokens becomes the validation split (or pass `--val_input`).

```bash
# the book's short story (already in data/raw/)
python prepare_data.py --input data/raw/the-verdict.txt --output_dir data/verdict

# a directory of .txt files
python prepare_data.py --input /mnt/nvme/books/ --output_dir /mnt/nvme/tok/books --val_fraction 0.005 --num_proc 64

# jsonl / csv / parquet with the text in a named field
python prepare_data.py --input corpus.jsonl --text_field content --output_dir data/corpus

# a Hugging Face dataset, streamed (uv pip install datasets)
python prepare_data.py --hf_dataset HuggingFaceFW/fineweb-edu --hf_config sample-10BT \
    --output_dir /mnt/nvme/tok/fineweb-edu-10bt --val_fraction 0.001 --num_proc 64
```

`--tokenizer` takes any tiktoken encoding (`gpt2` default, `cl100k_base`, `o200k_base`).

## 2. Train

```bash
# one GPU
python train.py --config configs/gpt2_verdict.yaml

# all 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml

# change anything from the command line
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
    --set data.data_dir=data/corpus train.out_dir=runs/corpus train.epochs=3 train.lr=3e-4 model.preset=llama-tiny

# resume after an interruption
... --set train.resume=auto
```

Each run writes to `train.out_dir`: `config.yaml`, `metrics.jsonl`, `tb/` (TensorBoard),
`loss.png`, `ckpt_best.pt` (lowest val loss, weights only) and `ckpt_last.pt` (weights +
optimizer, for resuming). Watch live with `tensorboard --logdir runs`.

Key config fields (see `llmtrain/config.py` for all):

| field | meaning |
|---|---|
| `model.preset` | `gpt2-124m/355m/774m/1558m`, `llama-tiny/1b/3b`; other `model.*` fields override it |
| `model.type` | `gpt` or `llama` when not using a preset |
| `model.init_std` | GPT-2 style init (0.02); `null` = PyTorch default init, as in the book |
| `train.batch_size` | sequences **per GPU** per micro-step |
| `train.grad_accum_steps` | tokens/step = GPUs x batch_size x context_length x grad_accum_steps |
| `train.epochs` / `train.max_steps` | set exactly one |
| `train.lr_schedule` | `cosine` (with `warmup_steps`, `min_lr_ratio`) or `constant` |
| `train.compile` | `torch.compile` the model (faster for long runs) |
| `train.eval_every`, `train.eval_batches` | validation frequency, and per-GPU batch cap |

The global batch grows with the number of GPUs, so the same config takes fewer optimizer
steps per epoch on 8 GPUs than on 1.

## 3. Evaluate

```bash
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt                       # run's val split
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --data_dir data/corpus  # another dataset
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --text_files held_out.txt
torchrun --standalone --nproc_per_node=8 evaluate.py --checkpoint ... --data_dir /mnt/nvme/tok/x
... --prompt "Every effort moves you" --max_new_tokens 50                             # add samples
```

Reports cross-entropy loss, perplexity, top-1 next-token accuracy and bits per byte
(comparable across tokenizers), and saves them as JSON next to the checkpoint. The same
metrics are computed on the train and val splits during training.

## 4. Generate

```bash
python generate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --prompt "Every effort moves you"
python generate.py --checkpoint ... --prompt "I had" --temperature 0.8 --top_k 50 --num_samples 3
python generate.py --checkpoint ...    # interactive
```

## Reference results: GPT-2 124M on "The Verdict" (`configs/gpt2_verdict.yaml`)

| | steps | train loss | val loss |
|---|---|---|---|
| book (ch05, 1 GPU) | 90 | ~0.39 | ~6.45 |
| this repo, 1 GPU | 90 | 0.30 | 6.67 |
| this repo, 8 GPUs | 20 | 2.92 | 6.21 |

The story is ~5k tokens, so a 124M model memorises it; this run checks the pipeline, it is
not a meaningful language model. For real pretraining start from
`configs/gpt2_124m_pretrain.yaml`, and keep data and checkpoints on the NVMe drives
(the root disk is only 45 GB).
