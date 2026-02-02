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
    - regional-compile
  domain: research
  created: 2026-01-30
  author: codex
---

# torch.compile Dynamic Shapes (nanoVLM)

## General Description

In nanoVLM-style training, the collator can drop samples or produce variable-length batches (variable `B` and/or `T`).
With `torch.compile`, that typically triggers recompilations because graphs are guarded on tensor shapes.

This skill captures an opt-in approach to reduce these recompiles by marking only the `(B, T)` dims dynamic while
keeping hidden dims static, and by favoring **regional compile** (vision/decoder/MP) to reduce compile scope and
avoid graph breaks from non-tensor helpers.

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

<u>Important: all optimization changes must land in `train.py` (single source of truth). The benchmark (`eval/benchmark_train_step.py`) does not accept optimization flags and only measures the current `train.py` setup.</u>

### Step 1: Use regional compile (per block)

Compile only the repeated blocks (plus MP) to reduce compile time and avoid tracing Python/list-heavy glue code:

```python
for idx, block in enumerate(model.vision_encoder.blocks):
    model.vision_encoder.blocks[idx] = torch.compile(block, mode="reduce-overhead")
for idx, block in enumerate(model.decoder.blocks):
    model.decoder.blocks[idx] = torch.compile(block, mode="reduce-overhead")
model.MP = torch.compile(model.MP, mode="reduce-overhead")
```

### Step 2: Enable dynamic `(B,T)` marking (train.py default)

In this repo, when `TrainConfig.compile=True`, `train.py` **always** applies `torch._dynamo.maybe_mark_dynamic`
on the `(B,T)` dims of `input_ids`, `labels`, and `attention_mask`. There is no separate flag for this;
set `TrainConfig.compile=True` in `models/config.py` to enable compile + dynamic marking.

### Step 3: Verify recompiles are gone

Run a short training slice with:
```
TORCH_LOGS="recompiles,guards" python train.py ...
```
and confirm there are no recurring “Recompiling” messages after warmup (with `TrainConfig.compile=True`).

### Step 4: Keep MoMH block-mask construction outside compiled regions

When MoMH is enabled, building the block mask inside a compiled decoder graph causes a graph break. In this repo,
the VLM wrapper now builds the prefill block mask and passes it to the decoder. If you call `model.decoder(...)`
directly, you can still trigger a graph break unless you supply `prefill_block_mask`.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Still recompiles | Another dynamic axis changed (e.g. image tile count) | Bucket/pad or tensorize images |
| `mark_dynamic` errors | Dim specialized to a constant | Prefer `maybe_mark_dynamic` |
| Graph break in MoMH prefill | Block-mask creation inside compiled decoder | Build mask in VLM wrapper and pass in |
| Long compile times | Full-model compile or dynamic=True everywhere | Use regional compile and limit dynamic dims |
| flex_attention block mask recompiles | Guard on batch size in create_block_mask | Keep B fixed per compile run or cache/precompute mask |
| Flex decoding guard recompiles | guard on seq length in flex decoding path | Avoid crossing guard thresholds within one compile session |

## References

- Related reports: `training_reports/2026-01-30_torch-compile_dynamic-shapes_and_benchmark.md`
- Related skills: `torch-compile-dynamic-shapes`, `torch-compile-dynamic-metadata-propagation`
- Code: `train.py` (regional compile), `models/vision_language_model.py` (MoMH block-mask build)
