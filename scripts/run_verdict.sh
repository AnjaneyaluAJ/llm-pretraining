#!/usr/bin/env bash
# End-to-end pipeline test on "The Verdict": prepare -> train on all GPUs -> evaluate -> generate.
# Usage: bash scripts/run_verdict.sh [num_gpus]   (default: all visible GPUs)
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=${1:-$(nvidia-smi -L | wc -l)}
PY=.venv/bin/python

[ -f data/verdict/meta.json ] || $PY prepare_data.py --input data/raw/the-verdict.txt --output_dir data/verdict --num_proc 1
.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" train.py --config configs/gpt2_verdict.yaml
.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" evaluate.py --checkpoint runs/gpt2_verdict/ckpt_best.pt
$PY generate.py --checkpoint runs/gpt2_verdict/ckpt_last.pt --prompt "Every effort moves you" --max_new_tokens 50
