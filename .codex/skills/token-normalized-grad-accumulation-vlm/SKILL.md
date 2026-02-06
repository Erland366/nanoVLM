---
name: token-normalized-grad-accumulation-vlm
description: >
  Normalize gradient accumulation by valid target tokens instead of per-microbatch mean loss.
  Use when: variable-length VLM batches make accumulation unstable or noisy.
metadata:
  short-description: "Token-count normalized accumulation for stable VLM training"
  tags:
    - optimization
    - gradient-accumulation
    - loss-scaling
    - vlm
    - stability
  domain: research
  created: 2026-02-06
  author: codex
---

# Token-Normalized Gradient Accumulation (VLM)

## General Description

This skill captures a training-stability fix for variable-length VLM batches under gradient accumulation.
Instead of averaging CE loss per microbatch, accumulate CE with `reduction="sum"` and normalize gradients
once per optimizer step by total valid target tokens. This avoids overweighting short-target microbatches.

## When to Apply

Use this knowledge when:
- `labels` contain variable counts of valid targets (`!= -100`) across microbatches.
- Training loss is noisy or converges worse than a baseline despite similar hyperparameters.
- You use gradient accumulation with heterogeneous sequence/answer lengths.

Do NOT use when:
- All microbatches have near-identical valid target counts and training is already stable.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Stability after patch | No NaN seen in tested runs | Finite losses through launched 500-step test run |
| Behavioral parity | Matched DualTowerVLM accumulation strategy | Uses token-count normalization at optimizer step |
| Primary symptom addressed | Noisy/biased accumulation updates | Removes short-target microbatch overweighting |

## Recommended Practice

- In forward, request CE as a **sum** and return valid token count:
  - `loss_reduction="sum"`
  - `return_loss_count=True`
- Accumulate:
  - `accumulated_loss_sum += loss`
  - `accumulated_loss_tokens += loss_token_count`
- At optimizer step:
  - all-reduce both sums across ranks (if DDP)
  - scale grads by `world_size / total_valid_tokens`
  - then clip grad norm and apply optimizer step
- For logging, use per-token batch loss:
  - `batch_loss = loss.item() / loss_token_count`

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Per-microbatch mean CE under accumulation | Variable target counts bias update weighting | Normalize by total valid tokens across the whole accumulation window |
| Missing zero-token guard | Empty-target microbatch can create invalid scaling | Raise on `loss_token_count == 0` and on `total_valid_tokens == 0` |
| Scaling only by accumulation steps | Ignores token-count variability | Accumulation factor alone is insufficient for variable-length supervision |

## Configuration

```yaml
# Forward API expectation
VisionLanguageModel.forward:
  loss_reduction: "sum"
  return_loss_count: true

# Optimizer-step normalization
grad_scale: world_size / total_valid_tokens
```

## References

- Related log entry: `references/experiment-log.md`
- Related troubleshooting: `references/troubleshooting.md`
- Code: `train.py`, `models/vision_language_model.py`
- Related skill: `padding-aware-lr-scaling`
