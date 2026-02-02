# Report: Effective-token LR scaling vs packed baseline

**Date:** 2026-02-02
**Author:** Codex
**Status:** Completed

## Objective

Reduce the loss gap between packed and non-packed training by scaling the learning rate with the effective-token ratio (non-padding tokens per update), and evaluate exponent/schedule variants under a fixed token budget.

## Setup

### Environment
- Hardware: NVIDIA RTX PRO 6000 Blackwell (1 GPU used)
- Software: Python 3.13.11, wandb 0.23.1
- Dataset: `patrickamadeus/the_cauldron` (sample_1pct)

### Configuration

```yaml
# Common settings (non-pack runs)
batch_size: 1
gradient_accumulation_steps: 8
max_training_tokens: 500000
pack_sequences: false
lr_schedule_by_tokens: false  # unless noted
lr_mp: 5e-05
lr_vision_backbone: 1e-05
lr_language_backbone: 1e-05
```

## Experiments

### Baseline: packed (reference)

**Run ID:** `ra3h1k7e`

**Results:**
| Metric | Value |
|--------|-------|
| batch_loss | 6.4609 |
| epoch_loss | 7.9333 |
| tokens/consumed | 2,326,651 |

### Non-pack runs (effective-token LR scaling)

MAE is computed vs the packed baseline over `batch_loss` vs `tokens/consumed` (linear interpolation on the packed curve).

| label | run_id | exponent | ema | lr_schedule_by_tokens | MAE vs pack | batch_loss | epoch_loss |
|---|---|---:|---:|---:|---:|---:|---:|
| ratio_exp1.5_step | `4na14fwv` | 1.5 |  | False | 1.540761 | 9.4528 | 8.7321 |
| ratio_exp1.5_step_ema0.9 | `5hrdh9yx` | 1.5 | 0.9 | False | 1.543124 | 9.4596 | 8.7552 |
| exp1_wandb | `shntpy7i` | 1.0 |  | False | 1.563684 | 9.1223 | 8.1320 |
| ratio_exp2_step | `ushl5ypi` | 2.0 |  | False | 1.669953 | 9.7697 | 9.2750 |
| ratio_exp1.5_token | `0fjify3v` | 1.5 |  | True | 1.735130 | 9.9022 | 8.9396 |
| ratio_exp1_step | `he0qvdab` | 0.5 |  | False | 1.787271 | 8.4616 | 7.4751 |
| ratio_exp1_token | `shovgf9c` | 0.5 |  | True | 2.040565 | 9.3441 | 7.4803 |
| no_ratio_no_sched | `cax8po1m` | - |  | False | 2.185831 | 7.1820 | 6.6083 |

## Analysis

### What Worked
- Linear/stronger scaling (exponent 1.0–1.5) narrowed the gap to packed training compared to no scaling.
- Step-based scheduling outperformed token-based scheduling for the same exponent.

### What Failed
- Token-based scheduling (`lr_schedule_by_tokens=True`) consistently worsened MAE vs packed.
- Very strong scaling (exponent 2.0) degraded performance.
- EMA smoothing (beta=0.9) did not improve over the non-EMA run at exponent 1.5.

### Key Insights
1. Best MAE so far is exponent 1.5 with step-based scheduling, closely followed by exponent 1.0 (linear).
2. Non-pack runs remain materially worse than packed under the same token budget (MAE ~1.54–1.56).
3. Token-based scheduling appears to interact poorly with effective-token scaling in this setup.

## Next Steps

- [ ] Consider alternative scaling rules (e.g., clamp ratio, floor LR, piecewise or log scaling) to reduce over-penalization in highly padded batches.
- [ ] Investigate denominator choice (max_seq_len vs per-batch max length) and whether mixed precision of ratio introduces bias.
- [ ] Compare against a baseline that uses packing + same LR schedule to isolate padding-only effects.

## Appendix

Runs and metrics sourced from W&B project `erlandpg/dualtower`.
