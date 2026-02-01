---
name: torch-compile-dynamic-shapes-nanovlm
description: >
  Reduce torch.compile recompiles caused by variable batch size / seq length in nanoVLM-style training loops.
  Use when: TORCH_LOGS=recompiles shows size-mismatch guards on input_ids/attention_mask.
metadata:
  short-description: "Mark (B,T) dynamic to prevent recompiles"
  tags:
    - torch-compile
    - dynamic-shapes
    - mark_dynamic
    - recompiles
    - nanovlm
  domain: research
  created: 2026-01-30
  author: codex
---

# torch.compile Dynamic Shapes (nanoVLM)

## General Description

In nanoVLM-style training, the collator can drop samples or produce variable-length batches (variable `B` and/or `T`).
With `torch.compile`, that typically triggers recompilations because graphs are guarded on tensor shapes.

This skill captures an opt-in approach to reduce these recompiles by marking only the `(B, T)` dims dynamic while
keeping hidden dims static.

## When to Apply

Use this when:
- `TORCH_LOGS="recompiles"` shows recompiles due to batch size or sequence length changing.
- You want stable training throughput with `torch.compile` even if collation drops samples.

Do NOT use when:
- You are benchmarking a fixed shape (e.g. a stable train-step benchmark) and want maximum specialization.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Recompiles (varying B/T) | Reduced | Depends on remaining dynamic sources like image tile count |

## Recommended Practice

### Step 1: Enable dynamic `(B,T)` marking

In this repo:
- Set `TrainConfig.compile_dynamic_shapes=True`, or
- Run: `python train.py --compile True --compile_dynamic_shapes True`

This marks dynamic dims for `input_ids`, `labels`, and `attention_mask` using:
- `torch._dynamo.maybe_mark_dynamic(t, 0)` (batch)
- `torch._dynamo.maybe_mark_dynamic(t, 1)` (seq)

### Step 2: Verify recompiles are gone

Run a short training slice with:
```
TORCH_LOGS="recompiles,guards" python train.py --compile True --compile_dynamic_shapes True ...
```
and confirm there are no recurring “Recompiling” messages after warmup.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Still recompiles | Another dynamic axis changed (e.g. image tile count) | Bucket/pad or tensorize images |
| `mark_dynamic` errors | Dim specialized to a constant | Prefer `maybe_mark_dynamic` |

## References

- Related reports: `training_reports/2026-01-30_torch-compile_dynamic-shapes_and_benchmark.md`
- Related skills: `torch-compile-dynamic-shapes`, `torch-compile-dynamic-metadata-propagation`
