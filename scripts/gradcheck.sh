#!/usr/bin/env bash
# One-step gradient equivalence of every strategy vs the single-GPU reference.
# Usage: bash scripts/gradcheck.sh [bf16|fp32]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_FP_FUSION=1
PREC=${1:-bf16}
ROOT=/mnt/nvme/runs/gradcheck/$PREC
rm -rf "$ROOT"
B="scripts/bench_parallel.py --precision $PREC --micro_batch 8 --global_batch 32"
one() { name=$1; shift; CUDA_VISIBLE_DEVICES=0 .venv/bin/python $B --check_grads $ROOT/$name "$@" 2>&1 | grep -E "saved|Error" | tail -1 | sed "s/^/$name: /"; }
two() { name=$1; shift; CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 $B --check_grads $ROOT/$name "$@" 2>&1 | grep -E "saved|Error" | tail -1 | sed "s/^/$name: /"; }
one single         --strategy single
one single_micro4  --strategy single --micro_batch 4
one single_ckpt    --strategy single --act_ckpt
two ddp            --strategy ddp
two ddp_tuned      --strategy ddp --ddp_tuned
[ "$PREC" = bf16 ] && two ddp_bf16grads --strategy ddp --ddp_bf16_grads   # bf16 gradient all-reduce only makes sense with bf16
two fsdp           --strategy fsdp
two fsdp_bf16      --strategy fsdp --fsdp_bf16
two fsdp_noreshard --strategy fsdp --fsdp_no_reshard
two tp             --strategy tp
two tp_sp          --strategy tp --tp_sp
two tp_shared      --strategy tp --tp_shared_input
two tp_sp_shared   --strategy tp --tp_sp --tp_shared_input
two pp_1f1b        --strategy pp
[ "$PREC" = bf16 ] && two pp_gpipe --strategy pp --pp_schedule gpipe   # fp32 GPipe does not fit: all microbatch activations live at once
two pp_split7      --strategy pp --pp_split 7
two pp_interleaved --strategy pp --pp_virtual 2
.venv/bin/python scripts/compare_grads.py $ROOT --ref single --json $ROOT/summary.json
