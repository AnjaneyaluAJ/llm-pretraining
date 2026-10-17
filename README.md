# llm-pretraining

Train GPT-2 and Llama-style language models **from scratch** on any text, evaluate them,
and **instruction-finetune** pretrained GPT-2, all from the console with plain Python
scripts. Everything runs on one GPU or on every GPU of the node (`torchrun`, DDP).

The models and recipes follow Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (chapters 2-7). The code
is reorganised into a reusable package and extended with:

- multi-GPU training (DistributedDataParallel), bf16 mixed precision, optional `torch.compile`
- a data pipeline for any text format, memory-mapped from disk, so datasets can be larger than RAM
- YAML configs with command-line overrides, checkpoints and automatic resume
- evaluation: loss, perplexity, next-token accuracy and bits-per-byte, during training and standalone
- loading OpenAI's GPT-2 weights, instruction finetuning, and LLM-judge scoring

---

## Contents

1. [Quick start](#quick-start)
2. [Setup](#setup)
3. [Repository layout](#repository-layout)
4. [Workflow overview](#workflow-overview)
5. [Step 1 - Prepare data](#step-1---prepare-data)
6. [Step 2 - Pretrain](#step-2---pretrain)
7. [Step 3 - Evaluate](#step-3---evaluate)
8. [Step 4 - Generate text](#step-4---generate-text)
9. [Step 5 - Instruction finetuning](#step-5---instruction-finetuning)
10. [Step 6 - Score responses with an LLM judge](#step-6---score-responses-with-an-llm-judge)
11. [Configuration reference](#configuration-reference)
12. [Reference results](#reference-results)
13. [Running long jobs](#running-long-jobs)
14. [Troubleshooting](#troubleshooting)
15. [Mapping to the book](#mapping-to-the-book)

---

## Quick start

On the GPU server:

```bash
ssh a100-oci
cd ~/AJ/llm-pretraining
source .venv/bin/activate

# Pretrain GPT-2 124M on "The Verdict" on all GPUs, then evaluate and generate
bash scripts/run_verdict.sh

# Instruction-finetune OpenAI's GPT-2 355M and chat with it
python import_gpt2.py --size 355M
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-355m.pt --out_dir runs/sft_gpt2_355m
python generate.py --checkpoint runs/sft_gpt2_355m/model.pt --instruct --prompt "Convert 5 kilometers to meters."
```

---

## Setup

The repo has its own virtual environment in `.venv` (Python 3.12, PyTorch 2.14 + CUDA 13.2).
To recreate it:

```bash
cd ~/AJ/llm-pretraining
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend=cu132
source .venv/bin/activate
```

`requirements.txt` is the reference repo's file plus `pyyaml` (configs) and `tensorboard`
(training curves). TensorFlow, also from the reference requirements, is used only to read
OpenAI's original GPT-2 checkpoints.

Optional extras, only needed for specific features:

| package | needed for | install |
|---|---|---|
| `datasets` | `prepare_data.py --hf_dataset` | `uv pip install datasets` |
| `pyarrow` | `.parquet` input files | `uv pip install pyarrow` |
| Ollama | `score_responses.py` | see [Step 6](#step-6---score-responses-with-an-llm-judge) |

Check the GPUs:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

---

## Repository layout

```
llm-pretraining/
├── prepare_data.py          raw text / jsonl / csv / parquet / HF dataset -> train.bin, val.bin, meta.json
├── train.py                 pretraining (python or torchrun), config driven
├── evaluate.py              loss, perplexity, accuracy, bits/byte (+ sample generations)
├── generate.py              text generation (one-shot or interactive, base or instruction mode)
├── import_gpt2.py           download OpenAI GPT-2 weights -> checkpoint
├── finetune.py              instruction finetuning, writes test-set responses
├── score_responses.py       LLM-judge scoring of responses (Ollama)
├── configs/
│   ├── gpt2_verdict.yaml        GPT-2 124M on The Verdict (book ch05 settings)
│   ├── llama_tiny_verdict.yaml  small Llama-style model on The Verdict
│   └── gpt2_124m_pretrain.yaml  template for a real ~10B-token run on 8 GPUs
├── scripts/run_verdict.sh   end-to-end test: prepare -> train -> evaluate -> generate
├── data/
│   ├── raw/the-verdict.txt          pretraining test text (book ch02)
│   └── instruct/instruction-data.json  1,100 instruction examples (book ch07)
├── llmtrain/                the library behind the scripts
│   ├── config.py            dataclass configs, YAML loading, --set overrides
│   ├── models/              gpt.py (GPT-2), llama.py (Llama 3), presets, build_model
│   ├── data.py              memory-mapped token windows, distributed samplers
│   ├── evaluation.py        distributed loss / perplexity / accuracy / bits-per-byte
│   ├── generation.py        greedy, temperature, top-k, top-p sampling
│   ├── optim.py             AdamW parameter groups, warmup + cosine schedule
│   ├── checkpoint.py        save / load / resume
│   ├── dist.py              torchrun setup and all-reduce helpers
│   ├── tracking.py          metrics.jsonl, TensorBoard, loss plot
│   ├── tokenizer.py         tiktoken encodings
│   ├── gpt2_weights.py      OpenAI GPT-2 download and weight mapping
│   └── instruct.py          instruction prompt template, dataset, padding/masking
├── pretrained/              downloaded / imported base models (git-ignored)
└── runs/                    training outputs (git-ignored)
```

---

## Workflow overview

```
            ┌──────────────────┐     ┌──────────┐     ┌─────────────┐     ┌─────────────┐
 raw text ─►│ prepare_data.py  ├────►│ train.py ├────►│ evaluate.py │     │ generate.py │
            │ train/val .bin   │     │ ckpt_*.pt│     │ ppl, acc... │     │ samples     │
            └──────────────────┘     └────┬─────┘     └─────────────┘     └─────────────┘
                                          │ (any checkpoint)
 OpenAI GPT-2 ─► import_gpt2.py ──────────┤
                                          ▼
 instruction data ───────────────► finetune.py ─► model.pt + test_responses.json ─► score_responses.py
```

Every checkpoint (from `train.py`, `import_gpt2.py` or `finetune.py`) has the same format,
so `evaluate.py`, `generate.py` and `finetune.py` accept any of them.

---

## Step 1 - Prepare data

`prepare_data.py` tokenizes text once and writes three files that training reads directly
from disk:

| file | content |
|---|---|
| `train.bin` | training tokens, one flat array (`uint16` for GPT-2's 50k vocabulary) |
| `val.bin` | validation tokens |
| `meta.json` | tokenizer, vocabulary size, token counts, bytes per token, sources |

### Supported inputs

Any mix of files, directories (searched recursively) and glob patterns:

| format | how documents are read |
|---|---|
| `.txt` `.md` `.text` `.txt.utf8` | one document per file |
| `.jsonl` `.jsonl.gz` | one document per line, text from `--text_field` |
| `.json` | a list of records (or one record), text from `--text_field` |
| `.csv` | one document per row, text from column `--text_field` |
| `.parquet` | one document per row (needs `pyarrow`) |
| Hugging Face dataset | `--hf_dataset NAME [--hf_config SUBSET]`, streamed (needs `datasets`) |

Documents are separated with `<|endoftext|>`. Unless you pass `--val_input`, the **last**
`--val_fraction` of all tokens becomes the validation split.

### Options

| option | default | meaning |
|---|---|---|
| `--input` | | files, directories or globs |
| `--hf_dataset`, `--hf_config`, `--hf_split` | `--hf_split train` | Hugging Face source instead of `--input` |
| `--text_field` | `text` | field / column holding the text |
| `--max_docs` | all | stop after N documents (useful for quick tests) |
| `--output_dir` | required | where the three files are written |
| `--tokenizer` | `gpt2` | any tiktoken encoding: `gpt2`, `cl100k_base`, `o200k_base` |
| `--val_fraction` | `0.1` | share of tokens held out for validation |
| `--val_input` | | separate validation files (overrides `--val_fraction`) |
| `--num_proc` | half the CPUs, max 32 | tokenizer worker processes |
| `--overwrite` | off | replace an existing prepared dataset |

### Examples

```bash
# The book's short story (used by the verdict configs)
python prepare_data.py --input data/raw/the-verdict.txt --output_dir data/verdict

# A folder of text files, 0.5% held out, 64 tokenizer processes
python prepare_data.py --input /mnt/nvme/books/ --output_dir /mnt/nvme/tok/books \
    --val_fraction 0.005 --num_proc 64

# Several files and a glob (quote globs so the shell does not expand them)
python prepare_data.py --input notes.md "articles/**/*.txt" --output_dir data/mix

# JSONL whose text is in a field called "content"
python prepare_data.py --input corpus.jsonl --text_field content --output_dir data/corpus

# CSV column "body", with a separate validation file
python prepare_data.py --input train.csv --val_input heldout.csv --text_field body --output_dir data/csvdata

# Hugging Face FineWeb-Edu 10B-token sample, streamed straight into token files
uv pip install datasets
python prepare_data.py --hf_dataset HuggingFaceFW/fineweb-edu --hf_config sample-10BT \
    --output_dir /mnt/nvme/tok/fineweb-edu-10bt --val_fraction 0.001 --num_proc 64

# Quick 10k-document test of a big dataset
python prepare_data.py --hf_dataset HuggingFaceFW/fineweb-edu --hf_config sample-10BT \
    --max_docs 10000 --output_dir data/fineweb_tiny

# A different tokenizer (100k vocabulary, stored as uint32)
python prepare_data.py --input data/raw/the-verdict.txt --tokenizer cl100k_base --output_dir data/verdict_cl100k
```

Output for The Verdict:

```
1 documents, 0.0 MB of text -> 5,145 tokens (3.98 bytes/token) in 0.0s
train: 4,631 tokens  data/verdict/train.bin
val:   514 tokens  data/verdict/val.bin
```

> The validation split must hold more than `context_length + 1` tokens, otherwise training
> warns and skips validation.

---

## Step 2 - Pretrain

### Launching

```bash
# One GPU
python train.py --config configs/gpt2_verdict.yaml

# All 8 GPUs on this machine
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml

# A subset of GPUs (here GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config configs/gpt2_verdict.yaml
```

### Overriding settings from the command line

Every config field can be changed with `--set section.field=value` (values parsed as YAML,
so `null`, `true`, numbers work):

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
    --set train.epochs=40 train.lr=3e-4 train.out_dir=runs/verdict_40ep

# Switch architecture without writing a new config
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
    --set model.preset=llama-tiny model.init_std=0.02 train.out_dir=runs/verdict_llama

# A custom-sized model
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
    --set model.n_layers=6 model.emb_dim=384 model.n_heads=6 train.out_dir=runs/verdict_small
```

### Training on your own data

```bash
python prepare_data.py --input /path/to/texts/ --output_dir data/mydata
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml \
    --set data.data_dir=data/mydata train.out_dir=runs/mydata train.epochs=3 \
          model.init_std=0.02 train.grad_clip=1.0 train.beta2=0.95 train.lr_schedule=cosine train.warmup_steps=100
```

(`gpt2_verdict.yaml` uses the book's settings, which suit a tiny text; the extra overrides
above are the standard choices for real pretraining. Or start from `gpt2_124m_pretrain.yaml`.)

### A real pretraining run

`configs/gpt2_124m_pretrain.yaml` is a template for GPT-2 124M on ~10B tokens across
8 GPUs (524,288 tokens per step, cosine schedule, `torch.compile`, auto-resume):

```bash
python prepare_data.py --hf_dataset HuggingFaceFW/fineweb-edu --hf_config sample-10BT \
    --output_dir /mnt/nvme/tok/fineweb-edu-10bt --val_fraction 0.001 --num_proc 64
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_124m_pretrain.yaml
```

Keep data and checkpoints on the NVMe drives: the root disk is only 45 GB, and a GPT-2 124M
checkpoint with optimizer state is ~1.5 GB.

### Batch size and steps

```
tokens per optimizer step = GPUs x train.batch_size x model.context_length x train.grad_accum_steps
steps per epoch            = windows in train split / (GPUs x batch_size x grad_accum_steps)
```

`train.batch_size` is **per GPU**, so the same config takes fewer, larger steps on 8 GPUs
than on 1 (The Verdict: 90 steps on 1 GPU, 20 on 8). Raise `train.epochs` or lower
`train.batch_size` to compensate. To fit a bigger global batch than memory allows, use
`train.grad_accum_steps`.

### What the console shows

```
model: gpt 162.4M params | layers=12 d=768 heads=12 ctx=256
data:  gpt2 | train 4,631 tok (18 windows) | val 514 tok (2 windows)
run:   8 GPU(s) x batch 2 x accum 1 = 4,096 tok/step | 2 steps/epoch | 20 steps total | bf16
Ep 1 (Step 000001/20) | loss 10.9935 | lr 5.00e-04 | 14,981 tok/s | MFU 0.6%
  eval step 5: train loss 7.166, val loss 7.969, val ppl 2891.24, val acc 6.6%
  sample: 'Every effort moves you know, and my a little of the picture to me, ...'
done: 20 steps, 81,920 tokens, best val loss 6.2049 (ppl 495.15)
```

- `loss`: training loss of the step, averaged over GPUs
- `grad`: gradient norm before clipping (shown when `grad_clip` is set)
- `tok/s`, `MFU`: throughput, and model FLOPs utilisation against `train.peak_flops`
  (A100 bf16 = 312 TFLOP/s per GPU)
- `eval`: loss / perplexity / accuracy over the train and val splits (`eval_batches` caps it)
- `sample`: generated text from `train.sample_prompt`

### Outputs (in `train.out_dir`)

| file | content |
|---|---|
| `config.yaml` | the fully resolved config of the run |
| `metrics.jsonl` | one JSON line per log / eval step |
| `tb/` | TensorBoard event files |
| `loss.png` | training and validation loss against steps and tokens |
| `ckpt_best.pt` | weights at the lowest validation loss |
| `ckpt_last.pt` | latest weights + optimizer state (for resuming) |

### Resuming

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml --set train.resume=auto
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_verdict.yaml --set train.resume=runs/x/ckpt_last.pt
```

`auto` resumes from `out_dir/ckpt_last.pt` if it exists, otherwise starts fresh. Set
`train.save_every` so long runs save `ckpt_last.pt` regularly. Training continues from the
exact batch position, learning-rate step and best validation loss.

### Watching training curves

```bash
# on your laptop: forward the port, then open http://localhost:6006
ssh -L 6006:localhost:6006 a100-oci 'cd ~/AJ/llm-pretraining && .venv/bin/tensorboard --logdir runs'

# copy the loss plot to your laptop
scp a100-oci:AJ/llm-pretraining/runs/gpt2_verdict/loss.png ~/Downloads/
```

---

## Step 3 - Evaluate

`evaluate.py` scores any checkpoint on a prepared dataset or on raw text files.

| metric | meaning |
|---|---|
| `loss` | average next-token cross-entropy (nats) |
| `perplexity` | `exp(loss)`: how many tokens the model is "choosing between" on average |
| `accuracy` | how often the most likely next token is the correct one |
| `bits/byte` | loss normalised by text bytes, comparable across tokenizers |

### Options

| option | default | meaning |
|---|---|---|
| `--checkpoint` | required | any checkpoint |
| `--data_dir` | the run's dataset | prepared dataset to evaluate on |
| `--split` | `val` | `train` or `val` |
| `--text_files` | | evaluate on raw text files instead (tokenized on the fly) |
| `--batch_size` | `8` | per GPU |
| `--context_length` | model's | evaluate with shorter windows |
| `--stride` | context length | window step; smaller = overlapping windows |
| `--max_batches` | all | per-GPU cap, for a quick estimate on big data |
| `--prompt` | | repeatable; generations are added to the report |
| `--max_new_tokens`, `--temperature`, `--top_k` | `50`, `0`, none | generation settings for `--prompt` |
| `--output` | `<checkpoint dir>/eval_<name>.json` | results file |

### Examples

```bash
# Validation split of the data the model was trained on
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt

# Same, sharded over all GPUs (identical numbers, faster on large data)
torchrun --standalone --nproc_per_node=8 evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt

# Training split, to measure overfitting
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --split train

# Another prepared dataset (must use the same tokenizer)
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --data_dir data/mydata

# Raw text files
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --text_files held_out.txt more.txt

# Quick estimate on a huge validation set
torchrun --standalone --nproc_per_node=8 evaluate.py --checkpoint /mnt/nvme/runs/gpt2_124m/ckpt_best.pt --max_batches 50

# Compare your model with OpenAI's GPT-2 on the same data
python import_gpt2.py --size 124M
python evaluate.py --checkpoint pretrained/gpt2-124m.pt --data_dir data/verdict

# Add generations to the report
python evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt \
    --prompt "Every effort moves you" --prompt "I had always thought" --max_new_tokens 50
```

Example output:

```
checkpoint: runs/gpt2_verdict/ckpt_best.pt (step 20, 81,920 tokens seen)
data:       data/verdict [val] | 514 tokens, 2 windows of 256
loss        6.2049
perplexity  495.15
accuracy    10.74%  (top-1 next token)
bits/byte   2.2490
tokens      512 evaluated on 8 GPU(s)
saved runs/gpt2_verdict/eval_verdict_val.json
```

---

## Step 4 - Generate text

| option | default | meaning |
|---|---|---|
| `--checkpoint` | required | any checkpoint |
| `--prompt` | interactive | text to continue; omit to type prompts (Ctrl-D quits) |
| `--instruct` | off | wrap the prompt in the instruction template (for finetuned models) |
| `--max_new_tokens` | `100` | length of the continuation |
| `--temperature` | `0` | `0` = greedy (deterministic); higher = more random |
| `--top_k`, `--top_p` | none | sample only from the k most likely tokens / smallest set with mass p |
| `--num_samples`, `--seed` | `1`, `123` | several samples, reproducible |

```bash
# Greedy continuation
python generate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt --prompt "Every effort moves you"

# Three creative samples
python generate.py --checkpoint pretrained/gpt2-355m.pt --prompt "The future of AI is" \
    --temperature 0.8 --top_k 50 --num_samples 3

# Nucleus sampling
python generate.py --checkpoint pretrained/gpt2-355m.pt --prompt "Once upon a time" --temperature 1.0 --top_p 0.9

# Ask an instruction-finetuned model a question
python generate.py --checkpoint runs/sft_gpt2_355m/model.pt --instruct --prompt "Name the author of 'Pride and Prejudice'."

# Interactive session with a finetuned model
python generate.py --checkpoint runs/sft_gpt2_355m/model.pt --instruct
```

---

## Step 5 - Instruction finetuning

Teach a pretrained model to follow instructions (book chapter 7).

### 5a. Get a base model

`import_gpt2.py` downloads OpenAI's GPT-2 and converts it to this repo's checkpoint format:

| size | params | layers / width | download | command |
|---|---|---|---|---|
| small | 124M | 12 / 768 | 0.5 GB | `python import_gpt2.py --size 124M` |
| medium | 355M | 24 / 1024 | 1.4 GB | `python import_gpt2.py --size 355M` |
| large | 774M | 36 / 1280 | 3 GB | `python import_gpt2.py --size 774M` |
| XL | 1558M | 48 / 1600 | 6 GB | `python import_gpt2.py --size 1558M` |

The checkpoint lands in `pretrained/gpt2-<size>.pt`, and the script prints a sample to
confirm the weights loaded. You can also finetune your own pretrained checkpoint
(`runs/.../ckpt_best.pt`).

### 5b. Instruction data format

A JSON list or JSONL file of records with `instruction`, optional `input`, and `output`:

```json
{"instruction": "Rewrite the sentence using a simile.",
 "input": "The car is very fast.",
 "output": "The car is as fast as lightning."}
```

Each record is formatted with the Alpaca template, and the model is trained on the full text:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Rewrite the sentence using a simile.

### Input:
The car is very fast.

### Response:
The car is as fast as lightning.<|endoftext|>
```

`data/instruct/instruction-data.json` (1,100 examples from the book) is split in file
order into 85% train / 10% test / 5% validation, unless you pass `--val_data` / `--test_data`.

### 5c. Finetune

```bash
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-355m.pt \
    --data data/instruct/instruction-data.json --out_dir runs/sft_gpt2_355m
```

| option | default | meaning |
|---|---|---|
| `--base` | required | checkpoint to start from |
| `--data` | `data/instruct/instruction-data.json` | JSON / JSONL records |
| `--val_data`, `--test_data` | split from `--data` | separate files |
| `--out_dir` | `runs/sft` | output directory |
| `--epochs` | `2` | passes over the training set |
| `--batch_size` | `8` | **global** examples per step (book value) |
| `--micro_batch_size` | `batch_size / GPUs` | per-GPU examples per forward pass; gradient accumulation fills the rest |
| `--lr`, `--weight_decay` | `5e-5`, `0.1` | AdamW (book values) |
| `--lr_schedule`, `--warmup_steps` | `constant`, `0` | or `cosine` with warmup |
| `--grad_clip` | none | gradient-norm clipping |
| `--max_length` | `1024` | truncate long examples |
| `--mask_prompt` | off | compute the loss on response tokens only |
| `--eval_every`, `--eval_batches` | `5`, `5` | periodic train/val loss |
| `--max_new_tokens` | `256` | length limit for test-set responses |

Because `--batch_size` is global, an 8-GPU run takes the same optimisation steps as the
book's single-GPU run (one example per GPU per step).

### More finetuning examples

```bash
# Smaller, faster model
python import_gpt2.py --size 124M
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-124m.pt --out_dir runs/sft_gpt2_124m

# Larger model, response-only loss, 3 epochs
python import_gpt2.py --size 774M
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-774m.pt \
    --out_dir runs/sft_gpt2_774m --mask_prompt --epochs 3

# Your own instruction data with separate validation and test files
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-355m.pt \
    --data my_train.jsonl --val_data my_val.jsonl --test_data my_test.jsonl --out_dir runs/sft_mine

# Bigger global batch with a cosine schedule
torchrun --standalone --nproc_per_node=8 finetune.py --base pretrained/gpt2-355m.pt \
    --batch_size 32 --lr 1e-4 --lr_schedule cosine --warmup_steps 10 --out_dir runs/sft_b32

# Finetune a model you pretrained yourself
torchrun --standalone --nproc_per_node=8 finetune.py --base runs/mydata/ckpt_best.pt --out_dir runs/sft_mydata

# Single GPU
python finetune.py --base pretrained/gpt2-124m.pt --out_dir runs/sft_1gpu
```

### What finetuning prints and writes

```
data: 935 train / 55 val / 110 test examples from data/instruct/instruction-data.json
run:  8 GPU(s) x micro-batch 1 x accum 1 = batch 8 | 116 steps/epoch | 232 steps | lr 5e-05
initial: train loss 3.848, val loss 3.756
Ep 1 (Step 000005/232): train loss 1.068, val loss 1.141
...
final (full splits): train loss 0.318, val loss 0.637, val ppl 1.89, val acc 86.1%

### Instruction: Name the author of 'Pride and Prejudice'.
>> expected: Jane Austen.
>> model:    The author of 'Pride and Prejudice' is Jane Austen.
```

| file | content |
|---|---|
| `model.pt` | finetuned weights (works with `generate.py --instruct` and `evaluate.py`) |
| `test_responses.json` | every test example plus the model's `model_response` |
| `metrics.jsonl`, `loss.png`, `tb/` | training curves |

---

## Step 6 - Score responses with an LLM judge

Loss tells you how well the model predicts the reference answers; a judge model tells you
how good its own answers are. `score_responses.py` follows the book: a larger model (Llama 3
via [Ollama](https://ollama.com)) rates each response 0-100 against the reference answer.

```bash
# one-time: install Ollama and pull the judge model (~4.7 GB)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# score one run
python score_responses.py runs/sft_gpt2_355m/test_responses.json

# compare several runs side by side
python score_responses.py runs/sft_gpt2_124m/test_responses.json runs/sft_gpt2_355m/test_responses.json

# a different judge model or server
python score_responses.py runs/sft_gpt2_355m/test_responses.json --judge llama3.1:70b
python score_responses.py runs/sft_gpt2_355m/test_responses.json --url http://other-host:11434/api/chat
```

The scores are saved per example in `test_responses_scored.json`, and one average per
file is printed:

```
judge: llama3
<average>  (<scored>/<total> scored)  runs/sft_gpt2_355m/test_responses.json
```

---

## Configuration reference

A config YAML has three sections. Anything not set uses the defaults in
`llmtrain/config.py`.

### `model`

| field | default | meaning |
|---|---|---|
| `preset` | none | fills the fields below; explicit fields override it |
| `type` | `gpt` | `gpt` (LayerNorm, learned positions, GELU) or `llama` (RMSNorm, RoPE, GQA, SwiGLU) |
| `vocab_size` | `50257` | must be at least the data tokenizer's vocabulary |
| `context_length` | `256` | tokens per training window |
| `emb_dim`, `n_heads`, `n_layers` | `768`, `12`, `12` | width, attention heads, depth |
| `drop_rate` | `0.1` | dropout |
| `qkv_bias` | `false` | bias on Q/K/V projections (true for OpenAI GPT-2) |
| `tie_embeddings` | `false` | share input embedding and output head |
| `init_std` | `0.02` | GPT-2 initialisation; `null` = PyTorch default (as in the book) |
| `n_kv_heads` | `n_heads` | llama: key/value heads for grouped-query attention |
| `hidden_dim` | ~8/3 x `emb_dim` | llama: SwiGLU width |
| `rope_theta`, `norm_eps` | `10000`, `1e-5` | llama: RoPE base, RMSNorm epsilon |

Presets:

| preset | type | params* | layers | emb_dim | heads (kv) | context |
|---|---|---|---|---|---|---|
| `gpt2-124m` | gpt | 124M | 12 | 768 | 12 | 1024 |
| `gpt2-355m` | gpt | 355M | 24 | 1024 | 16 | 1024 |
| `gpt2-774m` | gpt | 774M | 36 | 1280 | 20 | 1024 |
| `gpt2-1558m` | gpt | 1.56B | 48 | 1600 | 25 | 1024 |
| `llama-tiny` | llama | 51M | 8 | 512 | 8 (4) | 512 |
| `llama-1b` | llama | ~1.1B | 16 | 2048 | 32 (8) | 2048 |
| `llama-3b` | llama | ~3B | 28 | 3072 | 24 (8) | 2048 |

\* with tied embeddings and the 50k GPT-2 vocabulary. Without tying, GPT-2 124M has 162M
parameters (the separate output head adds 38.6M). The Llama presets keep Llama 3.2's
shapes but use the data's tokenizer.

### `data`

| field | default | meaning |
|---|---|---|
| `data_dir` | `data/verdict` | output of `prepare_data.py` |
| `stride` | `context_length` | training window step; smaller = overlapping windows |
| `num_workers` | `2` | DataLoader workers per GPU |

### `train`

| field | default | meaning |
|---|---|---|
| `out_dir` | `runs/default` | where outputs go |
| `seed` | `123` | |
| `batch_size` | `2` | sequences per GPU per micro-step |
| `grad_accum_steps` | `1` | micro-steps per optimizer step |
| `epochs` / `max_steps` | `10` / `null` | set exactly one |
| `lr` | `5e-4` | peak learning rate |
| `lr_schedule` | `cosine` | `cosine` or `constant` |
| `warmup_steps` | `0` | linear warmup |
| `min_lr_ratio` | `0.1` | cosine floor = `lr x min_lr_ratio` |
| `weight_decay` | `0.1` | applied to weight matrices only |
| `beta1`, `beta2` | `0.9`, `0.95` | AdamW betas (book: 0.9 / 0.999) |
| `grad_clip` | `1.0` | max gradient norm; `null` disables |
| `precision` | `bf16` | `bf16` autocast or `fp32` |
| `compile` | `false` | `torch.compile` (slower start, faster steps) |
| `peak_flops` | `312e12` | per-GPU peak for the MFU estimate |
| `log_every` | `1` | steps between log lines |
| `eval_every` | `5` | steps between evaluations |
| `eval_batches` | all | per-GPU batch cap per evaluation |
| `sample_every` | `50` | steps between generated samples (`0` disables) |
| `sample_prompt`, `sample_tokens` | `Every effort moves you`, `50` | |
| `sample_temperature`, `sample_top_k` | `0`, none | |
| `tensorboard` | `true` | write TensorBoard logs |
| `save_every` | none | steps between `ckpt_last.pt` saves (final save always happens) |
| `resume` | none | `auto` or a checkpoint path |

### Example config

```yaml
model:
  preset: gpt2-124m
  context_length: 512
  drop_rate: 0.0
  tie_embeddings: true

data:
  data_dir: /mnt/nvme/tok/books
  num_workers: 4

train:
  out_dir: /mnt/nvme/runs/books_124m
  batch_size: 16
  grad_accum_steps: 4
  epochs: null
  max_steps: 5000
  lr: 6.0e-4
  warmup_steps: 200
  compile: true
  log_every: 10
  eval_every: 250
  eval_batches: 20
  sample_every: 500
  save_every: 500
  resume: auto
```

---

## Reference results

Measured on this server (8x A100 80GB).

**Pretraining GPT-2 124M on The Verdict** (`configs/gpt2_verdict.yaml`, 10 epochs):

| | steps | train loss | val loss |
|---|---|---|---|
| book (ch05, 1 GPU) | 90 | ~0.39 | ~6.45 |
| this repo, 1 GPU | 90 | 0.30 | 6.67 |
| this repo, 8 GPUs | 20 | 2.92 | 6.21 |

The story is only ~5k tokens, so the model memorises it (low train loss, high val loss).
That makes it a pipeline check, not a useful language model.

**Instruction finetuning** (book data, 2 epochs, global batch 8, 8 GPUs):

| base model | train loss | val loss | val ppl | val acc | time |
|---|---|---|---|---|---|
| GPT-2 124M (1 epoch) | 0.65 | 0.79 | 2.20 | 83.6% | 6 s |
| GPT-2 355M | 0.32 | 0.64 | 1.89 | 86.1% | 24 s |
| book, GPT-2 355M | ~0.31 | ~0.66 | | | |

Sample answers from GPT-2 355M after finetuning:

| instruction | reference | model |
|---|---|---|
| Rewrite using a simile: *The car is very fast.* | The car is as fast as lightning. | The car is as fast as a cheetah. |
| Name the author of 'Pride and Prejudice'. | Jane Austen. | The author of 'Pride and Prejudice' is Jane Austen. |
| Type of cloud associated with thunderstorms? | cumulonimbus | ...a type of cloud called a cumulus. |

---

## Running long jobs

Use `tmux` so a run survives a dropped SSH connection:

```bash
tmux new -s train                     # start a session
torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_124m_pretrain.yaml
# detach: Ctrl-b then d       reattach later: tmux attach -t train
```

Or run in the background with a log file:

```bash
nohup torchrun --standalone --nproc_per_node=8 train.py --config configs/gpt2_124m_pretrain.yaml \
    > runs/pretrain.log 2>&1 &
tail -f runs/pretrain.log
```

Monitor the GPUs with `watch -n 2 nvidia-smi`. Stop a run with Ctrl-C (or `pkill -f train.py`);
restart it with `--set train.resume=auto`.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `CUDA out of memory` | lower `train.batch_size` (or `--micro_batch_size`) and raise `grad_accum_steps` to keep the global batch |
| `model.vocab_size ... is smaller than the data tokenizer's vocabulary` | the data uses a bigger tokenizer; set `model.vocab_size` to `meta.json`'s `vocab_size` |
| `Train split has N tokens; need more than context_length` | dataset too small; lower `model.context_length` or add data |
| `warning: val split ... validation is skipped` | raise `--val_fraction` when preparing, or lower `context_length` |
| `... already holds a prepared dataset` | add `--overwrite` to `prepare_data.py` |
| `was tokenized with X, but the model was trained with Y` | evaluate on data prepared with the model's tokenizer |
| GPUs busy with another job | check `nvidia-smi`; pick free GPUs with `CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ...` |
| C compiler `_POSIX_C_SOURCE redefined` warnings | harmless: Triton compiling its CUDA helper on the first `compile: true` run; cached afterwards |
| `OMP_NUM_THREADS` warning from torchrun | harmless |
| `No Ollama server at ...` | start Ollama (`ollama serve`) and `ollama pull llama3` |
| loss is `nan` | lower `lr`, set `grad_clip: 1.0`, keep `precision: bf16` |

---

## Mapping to the book

| book chapter | here |
|---|---|
| ch02 - text data, tokenization, sliding windows | `prepare_data.py`, `llmtrain/tokenizer.py`, `llmtrain/data.py` |
| ch03 - attention | `MultiHeadAttention` in `llmtrain/models/gpt.py` (fused `scaled_dot_product_attention`) |
| ch04 - GPT model | `llmtrain/models/gpt.py` (same parameter names as the book) |
| ch05 - pretraining, loss, generation, OpenAI weights | `train.py`, `llmtrain/evaluation.py`, `llmtrain/generation.py`, `import_gpt2.py` |
| ch05 bonus - Llama 3 | `llmtrain/models/llama.py` |
| ch07 - instruction finetuning and evaluation | `finetune.py`, `llmtrain/instruct.py`, `score_responses.py` |
| appendix D - warmup, cosine decay, gradient clipping | `llmtrain/optim.py`, `train.grad_clip` |
