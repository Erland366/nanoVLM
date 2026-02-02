---
name: padding-aware-lr-scaling
description: >
  Scale LR by effective-token ratio to reduce padding-induced noise in non-packed training.
  Use when: fine-tuning VLMs without packing and padding dominates token budgets.
metadata:
  short-description: "Effective-token LR scaling for non-pack training"
  tags:
    - lr-scaling
    - padding
  domain: research
  created: 2026-02-02
  author: patrick
---

# Padding-Aware LR Scaling

## General Description

This skill captures how effective-token LR scaling behaved when comparing non-packed
training runs against a packed baseline under a fixed token budget. The goal is to
reduce padding-induced noise by scaling the LR proportionally to the ratio of
non-padding tokens per update.

## When to Apply

Use this knowledge when:
- Fine-tuning VLMs with heavy padding and no sequence packing.
- You need to maintain stability under a small effective-token budget.
- You want to compare non-pack training to a packed reference at equal tokens.

Do NOT use when:
- Packing is available and stable (packing still outperforms non-pack here).

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| MAE vs packed (exp=1.5, step) | 1.5408 | Best MAE among tested non-pack variants |
| MAE vs packed (exp=1.0, step) | 1.5637 | Close second, visually competitive |
| Token-based scheduling | worse | Higher MAE across exponents |
| EMA smoothing (beta=0.9) | no gain | Similar to exp=1.5 step |

## Recommended Practice

Use effective-token LR scaling with a **step-based schedule** and exponent in the
**1.0–1.5** range.

### Step 1: Enable effective-token scaling

```
--effective_token_lr_scale True --effective_token_lr_exponent 1.0
```

### Step 2: Avoid EMA unless needed

EMA smoothing did not improve results at beta=0.9; skip unless you see instability.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Token-based scheduling | worsened MAE | Use step-based scheduling here |
| Exponent = 2.0 | over-penalized LR | Avoid strong scaling |
| EMA beta=0.9 | no improvement | EMA not helpful in this setup |

## Configuration

```yaml
# Non-pack LR scaling (recommended)
effective_token_lr_scale: true
effective_token_lr_exponent: 1.0  # try 1.5 if needed
```

## References

- Related reports: `training_reports/effective-token-lr-scaling-2026-02-02.md`
- Related skills: `torch-compile-dynamic-shapes-nanovlm`
