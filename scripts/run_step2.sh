#!/usr/bin/env bash
# Full step-2 measurement campaign: environment, ladder + leave-one-out, parallel strategies,
# memory limits and profiles. Repeats are interleaved (rep 1 of every config, then rep 2, ...)
# so drift in clocks or temperature cannot favour one configuration.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_FP_FUSION=1
R=${R:-/mnt/nvme/runs/step2}
REPS=${REPS:-3}
mkdir -p $R
DATA=/mnt/nvme/tok/fineweb-edu-300k

# ------------------------------------------------------------ environment
{ git rev-parse HEAD; git status --short; .venv/bin/python -c "import torch, triton; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"nccl\", \".\".join(map(str, torch.cuda.nccl.version())), \"triton\", triton.__version__)"
  nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,power.limit --format=csv; uname -r; } > $R/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv -l 1 > $R/gpu_clocks.csv &
SMI=$!
trap "kill $SMI" EXIT

# ------------------------------------------------------------ 1. ladder + leave-one-out (1 GPU, train.py)
COMMON="data.data_dir=$DATA data.num_workers=4 train.max_steps=80 train.epochs=null train.grad_accum_steps=1
  train.warmup_steps=10 train.log_every=5 train.eval_every=100000 train.eval_batches=2 train.sample_every=0
  train.hellaswag_every=null train.save_every=null train.tensorboard=false train.peak_flops=6.8e+13"
FULL="train.precision=bf16 train.tf32=true model.attn_impl=sdpa train.fused_adamw=true train.compile=true train.batch_size=8 model.vocab_size=50304"
LADDER=(
  "L0_fp32_baseline|train.precision=fp32 train.tf32=false model.attn_impl=manual train.fused_adamw=false train.compile=false train.batch_size=8"
  "L1_tf32|train.precision=fp32 train.tf32=true model.attn_impl=manual train.fused_adamw=false train.compile=false train.batch_size=8"
  "L2_bf16|train.precision=bf16 train.tf32=true model.attn_impl=manual train.fused_adamw=false train.compile=false train.batch_size=8"
  "L3_flash_attention|train.precision=bf16 train.tf32=true model.attn_impl=sdpa train.fused_adamw=false train.compile=false train.batch_size=8"
  "L4_fused_adamw|train.precision=bf16 train.tf32=true model.attn_impl=sdpa train.fused_adamw=true train.compile=false train.batch_size=8"
  "L5_torch_compile|train.precision=bf16 train.tf32=true model.attn_impl=sdpa train.fused_adamw=true train.compile=true train.batch_size=8"
  "L6_vocab_50304|$FULL"
  "L7_batch_16|$FULL train.batch_size=16"
  "O_minus_tf32|$FULL train.tf32=false"
  "O_minus_bf16|$FULL train.precision=fp32"
  "O_minus_flash|$FULL model.attn_impl=manual"
  "O_minus_fused_adamw|$FULL train.fused_adamw=false"
  "O_minus_compile|$FULL train.compile=false"
  "O_minus_vocab_pad|$FULL model.vocab_size=50257"
)
for rep in $(seq 1 $REPS); do
  for r in "${LADDER[@]}"; do
    name=${r%%|*}; settings=${r#*|}
    out=$R/ladder/rep$rep/$name
    echo "=== ladder rep$rep $name"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/gpt2_124m_fineweb_a10g.yaml \
      --set $COMMON $settings train.out_dir=$out 2>&1 | grep -E "Step 000080|Error" | tail -1
    rm -f $out/ckpt_*.pt
  done
done

# ------------------------------------------------------------ 2. parallel strategies (bench_parallel.py)
P=$R/parallel.jsonl
B="scripts/bench_parallel.py --steps 80 --skip 30 --out $P"
run1() { CUDA_VISIBLE_DEVICES=0 .venv/bin/python $B "$@" 2>&1 | grep -E "tok/s \||Error" | tail -1; }
run2() { CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 $B "$@" 2>&1 | grep -E "tok/s \||Error" | tail -1; }
run4() { .venv/bin/torchrun --standalone --nproc_per_node=4 $B "$@" 2>&1 | grep -E "tok/s \||Error" | tail -1; }
for rep in $(seq 1 $REPS); do
  echo "=== parallel rep$rep"
  run1 --strategy single --label single
  run2 --strategy ddp --label ddp
  run2 --strategy ddp --ddp_tuned --label ddp_tuned
  run2 --strategy ddp --ddp_bf16_grads --label ddp_bf16grads
  run2 --strategy fsdp --label fsdp
  run2 --strategy fsdp --fsdp_bf16 --label fsdp_bf16
  run2 --strategy fsdp --fsdp_bf16 --fsdp_no_reshard --label fsdp_bf16_noreshard
  run2 --strategy tp --label tp
  run2 --strategy tp --tp_sp --label tp_sp
  run2 --strategy tp --tp_shared_input --label tp_shared
  run2 --strategy tp --tp_sp --tp_shared_input --label tp_sp_shared
  for m in 4 8 16 32; do run2 --strategy pp --microbatches $m --label pp_1f1b_m$m; done
  run2 --strategy pp --microbatches 8 --pp_schedule gpipe --label pp_gpipe_m8
  for k in 7 8 9; do run2 --strategy pp --microbatches 8 --pp_split $k --label pp_1f1b_m8_split$k; done
  run2 --strategy pp --microbatches 8 --pp_virtual 2 --label pp_interleaved_m8
  run1 --strategy single --global_batch 128 --label single_gb128
  run2 --strategy ddp --global_batch 128 --label ddp_gb128
  run2 --strategy ddp --ddp_tuned --global_batch 128 --label ddp_tuned_gb128
  run2 --strategy fsdp --fsdp_bf16 --global_batch 128 --label fsdp_bf16_gb128
  run4 --strategy ddp --global_batch 128 --label ddp4_gb128
  run2 --strategy ddp --ddp_bf16_grads --global_batch 128 --label ddp_bf16grads_gb128
  run4 --strategy ddp --ddp_bf16_grads --global_batch 128 --label ddp4_bf16grads_gb128
  run4 --strategy fsdp --fsdp_bf16 --global_batch 128 --label fsdp4_bf16_gb128
done

# ------------------------------------------------------------ 3. memory limits (one rep, short)
M=$R/memory.jsonl
BM="scripts/bench_parallel.py --steps 12 --skip 4 --out $M"
m1() { CUDA_VISIBLE_DEVICES=0 .venv/bin/python $BM "$@" 2>&1 | grep -E "tok/s \||OutOfMemoryError" | tail -1 | cut -c1-120 | sed "s/^/[$*] /"; }
m2() { CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 $BM "$@" 2>&1 | grep -E "tok/s \||OutOfMemoryError" | tail -1 | cut -c1-120 | sed "s/^/[$*] /"; }
m4() { .venv/bin/torchrun --standalone --nproc_per_node=4 $BM "$@" 2>&1 | grep -E "tok/s \||OutOfMemoryError" | tail -1 | cut -c1-120 | sed "s/^/[$*] /"; }
echo "=== memory limits"
for preset in gpt2-774m gpt2-1558m; do
  for mb in 1 4; do
    m1 --strategy single --preset $preset --micro_batch $mb --label single_${preset}_mb$mb
    m1 --strategy single --preset $preset --micro_batch $mb --act_ckpt --label single_${preset}_mb${mb}_ckpt
    m2 --strategy ddp --preset $preset --micro_batch $mb --label ddp_${preset}_mb$mb
    m2 --strategy ddp --preset $preset --micro_batch $mb --act_ckpt --label ddp_${preset}_mb${mb}_ckpt
    m2 --strategy fsdp --fsdp_bf16 --preset $preset --micro_batch $mb --label fsdp_${preset}_mb$mb
    m2 --strategy fsdp --fsdp_bf16 --preset $preset --micro_batch $mb --act_ckpt --label fsdp_${preset}_mb${mb}_ckpt
    m4 --strategy fsdp --fsdp_bf16 --preset $preset --micro_batch $mb --label fsdp4_${preset}_mb$mb
  done
done

# ------------------------------------------------------------ 4. profiles (one rep)
F=$R/profile.jsonl
BP="scripts/bench_parallel.py --steps 40 --skip 20 --profile $R/traces --out $F"
p1() { CUDA_VISIBLE_DEVICES=0 .venv/bin/python $BP "$@" 2>&1 | grep -E "tok/s \||rank[0-9]:|Error"; }
p2() { CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 $BP "$@" 2>&1 | grep -E "tok/s \||rank[0-9]:|Error"; }
echo "=== profiles"
p1 --strategy single --label single
p2 --strategy ddp --label ddp
p2 --strategy ddp --global_batch 128 --label ddp_gb128
p2 --strategy ddp --ddp_bf16_grads --label ddp_bf16grads
p2 --strategy fsdp --label fsdp
p2 --strategy fsdp --fsdp_bf16 --label fsdp_bf16
p2 --strategy tp --label tp
p2 --strategy tp --tp_sp --label tp_sp
p2 --strategy tp --tp_shared_input --label tp_shared
p2 --strategy tp --tp_sp --tp_shared_input --label tp_sp_shared
p2 --strategy pp --microbatches 8 --label pp_1f1b_m8
p2 --strategy pp --microbatches 8 --pp_split 9 --label pp_1f1b_m8_split9
echo "=== all done"
