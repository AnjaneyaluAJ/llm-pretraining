# Step 3: a small scaling-law study (10M to 200M parameters)

**Status: draft. Phase 3 (the 1e18 FLOP budget) is still running; every section marked _pending_ is
filled in from its results.**

The question: for a fixed amount of training compute, how large should a GPT-2-style model be, and
how many tokens should it see? We answer it the way Chinchilla did (Hoffmann et al., 2022,
"Approach 2"): train several model sizes at the same compute budget, find the size with the lowest
loss, and repeat at larger budgets to see how the best size grows. Then we test the trend by
predicting the next budget before training it.

## Setup

| | |
|---|---|
| Model | GPT-2 architecture from this repo (`llmtrain/models/gpt.py`), context 1024, vocab 50,304, tied embeddings, head dim 64 |
| Data | FineWeb-Edu `sample-10BT`, GPT-2 tokenizer: 9.944B training tokens, 9.95M validation tokens (same split as step 1). No run sees a token twice. |
| Batch | 262,144 tokens per step (4 GPUs x 16 sequences x 1,024 x 4 accumulation steps; 8 x 8 for s200) |
| Optimizer | AdamW (fused), weight decay 0.1, grad clip 1.0, bf16 autocast, `torch.compile` |
| Schedule | linear warmup over 5% of steps, cosine decay to 10% of the peak, **ending exactly at each run's token budget** |
| Metric | validation loss on the full 9.95M-token validation split at the end of each run |
| Hardware | one AWS SageMaker `ml.g5.12xlarge` (4x A10G 24 GB), us-east-1, torch 2.15.0 + CUDA 13.2, NCCL 2.30.7 |

### Model sizes

N is **non-embedding** parameters, 12 x layers x d_model^2 (Kaplan et al., 2020). Total
parameters add the token and position embeddings.

| Name | Layers | d_model | Heads | N (non-emb.) | Total params | Tokens/s (4 GPUs) | MFU |
|---|---|---|---|---|---|---|---|
| s10 | 6 | 384 | 6 | 10.6M | 30.3M | 406k | 31% |
| s20 | 6 | 512 | 8 | 18.9M | 45.2M | 333k | 38% |
| s35 | 8 | 576 | 9 | 31.9M | 61.4M | 258k | 40% |
| s50 | 10 | 640 | 10 | 49.2M | 82.0M | 221k | 46% |
| s85 | 12 | 768 | 12 | 84.9M | 124.4M | 168k | 53% |
| s135 | 14 | 896 | 14 | 134.9M | 180.9M | 92k | 42% |
| s200 | 16 | 1024 | 16 | 201.3M | 253.9M | _pending_ | _pending_ |

MFU is measured against the A10G's measured bf16 matmul peak of 68 TFLOP/s (step 2).

### Compute budgets

Compute is counted as **C = 6 N D** with N non-embedding parameters and D training tokens. Each run's
step count is set so that 6 N D equals its budget: 1e17, 3e17 and 1e18 FLOPs.

## How each run's learning rate was chosen

A scaling-law comparison is only fair if every point is trained near its best learning rate:
a size given a poor learning rate looks worse than it is and moves the minimum. Small models want
higher learning rates than large ones, and short runs want lower ones than long runs.

1. **Sweeps at two reference sizes.** s20 at 1e17 (3,368 steps) over 6e-4 to 4.8e-3, and s85 at
   3e17 (2,246 steps) over 3e-4 to 4.8e-3, each on a factor-2 grid.
2. **Best learning rate per reference.** A parabola in log2(lr) through the lowest point and its two
   neighbours; its vertex is the estimate. If the vertex lies outside the swept range, the code
   refuses to continue (`EdgeError` in `scripts/step3/sweep.py`).
3. **A power law between them,** lr(N) = 3.58e-3 x (N / 18.9M)^-0.451, clipped to [1e-4, 6e-3] and
   rounded to two significant digits. This is a tool for choosing learning rates from two sizes,
   not a measured scaling law.

| Learning rate | 3e-4 | 6e-4 | 1.2e-3 | 2.4e-3 | 4.8e-3 | best (parabola) |
|---|---|---|---|---|---|---|
| s20, 1e17 | | 3.745 | 3.562 | 3.511 | 3.507 | 3.58e-3 |
| s85, 3e17 | 3.818 | 3.604 | 3.449 | 3.431 | 3.592 | 1.82e-3 |

### A correction made during the study

The first version of the rule (phase 2) used sweeps whose lowest loss was at the **top edge** of the
grid for both reference sizes, and took the edge value as the best learning rate. For s85 the loss
was still falling by 0.15 per doubling at the edge, so that rule set the larger models' learning
rates too low. We caught it from the data the next morning, extended the s85 sweep (phase 2c), made
the code refuse unbracketed optima, refit, and checked every earlier point against the new rule. All
points already run were within 1.4x of the corrected learning rate and were kept; the one point
the old rule would have run 2x too low (s135 at 3e17) was run only with the corrected value. Both
rules are in this folder (`lr_rule_p2.json`, `lr_rule_p2c.json`).

## Results

### IsoFLOP curves (validation loss, each size at its best learning rate)

| Budget | s10 | s20 | s35 | s50 | s85 | s135 | s200 | N_opt | D_opt | D/N |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e17 | 3.515 | **3.507** | 3.570 | 3.900 | 4.511 | | | 14.9M | 1.12B | 75 |
| 3e17 | | 3.344 | **3.318** | 3.333 | 3.431 | 3.708 | | 32.7M | 1.53B | 47 |
| 1e18 | | | | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

N_opt is the vertex of a parabola in ln(N) through the lowest point and its two neighbours
(`scripts/step3/analyze.py`). D/N uses non-embedding parameters.

### The prediction, registered before the 1e18 runs

From the 1e17 and 3e17 curves alone (`prediction_before_1e18.json`, commit `567eb35`, pushed before
phase 3 started): **N_opt about 77M, D_opt about 2.17B tokens, L_min about 3.13** (a pure power law in
C; an irreducible-loss floor would make the true minimum somewhat higher).

_Pending: measured 1e18 curve, N_opt, L_min, and how they compare with the prediction._

## Caveats

- **Few budgets.** N_opt(C) is fitted through three budgets spanning one decade of compute. The
  exponent is indicative, not precise.
- **Non-embedding counting.** Our tokens-per-parameter ratios are not comparable with Chinchilla's
  "about 20", which counts total parameters at much larger scale. At these sizes the embeddings are
  21% (s200) to 65% (s10) of all parameters, which is the main reason Kaplan and Chinchilla reported different
  exponents (Pearce and Song, 2024).
- **Short runs.** Points with fewer than about 1,500 steps (s85 at 1e17, s50 at 1e17, s135 at 3e17)
  prefer lower learning rates than the rule gives (s85 at 1e17 did best at 6e-4, the bottom of its
  grid). They sit on the large-model side of each curve, so they shape its right edge more than its
  minimum.
- **s10's learning rate** was capped at 6e-3 by the rule's clip.
- **s200's learning rate** extrapolates the rule 2.4x beyond its largest reference size.
- **One seed per point.** Run-to-run noise was not measured; differences below about 0.01 in loss
  should not be read as real.
- **HellaSwag** stayed at chance (24-27%) for every run, so it is reported in the raw results but
  not used.

## Reproducing

_Pending: SageMaker setup, data-prep job, the four phase commands, analysis._

## Files

| File | What |
|---|---|
| `results_lr.jsonl`, `results_p2.jsonl`, `results_p2c.jsonl` | one summary per run: size, budget, tokens, learning rate, final losses, hours |
| `lr_rule_p2.json`, `lr_rule_p2c.json` | the first (edge-based) and corrected learning-rate rules, with the sweep points used |
| `prediction_before_1e18.json` | the 1e18 prediction, written before any 1e18 run |
