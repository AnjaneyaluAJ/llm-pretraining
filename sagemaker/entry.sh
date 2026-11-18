#!/usr/bin/env bash
# Runs inside a SageMaker training job (AWS PyTorch DLC image). Builds the same environment
# as the GPU box (uv venv, torch $TORCH_SPEC + cu132), then runs $JOB_CMD from the repo root.
#   Inputs:  /opt/ml/input/data/<channel>/  (S3 channels, copied before start)
#   Outputs: /opt/ml/checkpoints/           (continuously synced to the job's checkpoint S3 URI)
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== host"; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
nproc; free -g | head -2; df -h / /opt/ml /tmp 2>/dev/null | sort -u

# scratch space: the mount with the most free space
export SCRATCH="$(df --output=avail,target /opt/ml /tmp 2>/dev/null | tail -n +2 | sort -n | tail -1 | awk '{print $2}')/scratch"
mkdir -p "$SCRATCH" /opt/ml/checkpoints
echo "SCRATCH=$SCRATCH"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
TORCH_SPEC="${TORCH_SPEC:-torch==2.15.0}"
uv venv -q --python 3.12 .venv
grep -v -i tensorflow requirements.txt > /tmp/req.txt
uv pip install -q --python .venv/bin/python -r /tmp/req.txt "$TORCH_SPEC" datasets pyarrow pyyaml \
    huggingface_hub hf_transfer awscli --torch-backend=cu132
export PATH="$PWD/.venv/bin:$PATH"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), 'nccl', torch.cuda.nccl.version())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=1
echo "== JOB_CMD: $JOB_CMD"
eval "$JOB_CMD"
echo "== JOB_DONE"
