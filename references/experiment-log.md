# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

<!-- New entries go above this line -->

## 2026-01-30 — Torch.compile dynamic shapes + train-step benchmark

**Type:** Retrospective
**General description:** Reduce `torch.compile` recompiles from variable batch/sequence shapes while tracking throughput via a stable benchmark.

### Details

Added an opt-in `TrainConfig.compile_dynamic_shapes` switch that uses `torch._dynamo.maybe_mark_dynamic` on `(B,T)` dims for `input_ids`, `labels`, and `attention_mask` in the training and validation loops. This targets the common training failure mode where the collator drops too-long samples and produces variable batch sizes / sequence lengths, leading to recompilations.

Also recorded a set of “contended GPU” benchmark numbers at `seq_len=2048` to keep a reference point for shared-GPU throughput comparisons.

### Key Points

- Prefer `maybe_mark_dynamic` over `mark_dynamic` to avoid `ConstraintViolationError` when dims sometimes specialize to constants.
- Dynamic `(B,T)` reduces recompiles, but tile-count variance (vision input length) can still trigger recompiles.

### Links

- Report: `training_reports/2026-01-30_torch-compile_dynamic-shapes_and_benchmark.md`
