# Torch.compile: dynamic shapes (mark_dynamic) + benchmark learnings

Date: 2026-01-30

## Context

We want stable throughput improvements from `torch.compile` while training on data that can produce:
- variable batch sizes (e.g. collator dropping too-long samples)
- variable sequence lengths (packing / padding changes)
- variable image tile counts

These variances can trigger `torch.compile` recompilations (guard failures on shapes).

## What we changed

### 1) Add `maybe_mark_dynamic` to reduce recompiles on variable `(B, T)`

- Added `TrainConfig.compile_dynamic_shapes` and CLI `--compile_dynamic_shapes`
- When enabled, we call `torch._dynamo.maybe_mark_dynamic` on dims 0/1 for:
  - `input_ids`
  - `labels`
  - `attention_mask`

Files:
- `models/config.py` (`TrainConfig.compile_dynamic_shapes`)
- `train.py` (marks dynamic dims in train and eval loops)

### 2) Benchmarks as the source of truth (non-adaptive)

- Use `eval/benchmark_train_step.py` as a fixed harness to compare changes.
- Under GPU contention, we created `train_step_contended_2048.jsonl` to record results.

## Results summary

Note: results under GPU contention are valid for “shared GPU throughput” but not directly comparable to clean baselines.

`seq_len=2048`, `batch_size=4`, HF mode:
- Vanilla eager: 3751.95 tok/s
- Vanilla compile: 4517.01 tok/s
- MoMH eager: 2897.16 tok/s
- MoMH compile: 6521.86 tok/s

## Open issues / next steps

1) Tile-count variance still likely recompiles (vision encoder input length changes).
   - Consider bucketing/padding number of tiles per batch, or tensorizing images in collator.
2) Verify recompiles reduction on real training loop with:
   - `TORCH_LOGS="recompiles,guards" python train.py --compile True --compile_dynamic_shapes True ...`

