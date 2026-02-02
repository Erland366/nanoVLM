---
name: nanovlm-checkpointing-determinism
description: >
  Deterministic resume for nanoVLM training with full-state checkpoints.
  Use when: you need exact step-for-step replay after resuming.
metadata:
  short-description: "Deterministic checkpoint/resume recipe"
  tags:
    - checkpointing
    - determinism
    - training
  domain: research
  created: 2026-02-02
  author: codex
---

# nanoVLM Checkpointing Determinism

## General Description

This skill captures the working recipe for deterministic resume in nanoVLM
training. It relies on saving model + optimizer + RNG + dataloader progress,
then restoring them strictly to reproduce step-by-step metrics.

## When to Apply

Use this knowledge when:
- You need to resume training on preemptible hardware.
- You must compare baseline vs resumed runs step-by-step.
- You need deterministic restart for debugging or ablation comparisons.

Do NOT use when:
- Training uses streaming datasets (`stream_dataset=True`) and exact determinism is required.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Step 50–99 match | Exact | GPU baseline vs resume match for `batch_loss` and `grad_norm` |
| GPU compatibility | Requires cu128 | Blackwell (sm_120) needs CUDA 12.8+ PyTorch |

## Recommended Practice

### Step 1: Enable deterministic map-style data

- Set `stream_dataset=False`.
- Keep `max_training_steps` identical between baseline and resume runs.

### Step 2: Save full-state checkpoints

Use a short interval for validation testing, then scale up:

```bash
python train.py \
  --checkpoint_every_n_steps 50 \
  --checkpoint_dir checkpoints \
  --checkpoint_format torch
```

### Step 3: Resume from a saved step

```bash
python train.py \
  --resume_from_checkpoint checkpoints/<run_name>/step_50 \
  --max_training_steps 100
```

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Divergent curves after resume | `max_training_steps` changed LR schedule | Keep schedule length identical between runs |
| Non-deterministic resume | Streaming dataset + shuffle | Use map-style datasets for determinism |
| CUDA kernel image error | PyTorch build lacks sm_120 | Install CUDA 12.8+ PyTorch (e.g., 2.10.0+cu128) |

## Configuration

```yaml
checkpoint_every_n_steps: 50
checkpoint_dir: checkpoints
checkpoint_format: torch
stream_dataset: false
max_training_steps: 100
```

## References

- Related reports: `references/experiment-log.md`
