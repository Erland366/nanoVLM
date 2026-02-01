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

## 2026-02-02 — Retrospective: activation checkpointing + compile tradeoffs

**Type:** Retrospective  
**General description:** Summarized activation checkpointing benchmarks and compile/recompile behavior under shape sweeps.

### Details

- Benchmarked manual vs selective activation checkpointing (no compile) and selective activation checkpointing under `torch.compile` in synthetic mode at `batch_size=1`, `seq_len=1024`.
- Selective AC under `torch.compile` requires `allow_cache_entry_mutation=True` to avoid cached‑tensor mutation errors; activation checkpointing now auto-selects selective when compile is enabled.
- Shape sweeps with `(B,T)=(1,512)->(2,1024)` and `(4,128)->(8,64)` still triggered recompiles in `flex_attention.create_block_mask` and block forwards due to batch-size guards; no graph breaks observed.

### Links

- Report: `training_reports/activation-checkpointing-benchmark-2026-02-02.md`
- Report: `training_reports/activation-checkpointing-compile-benchmark-2026-02-02.md`
- Report: `training_reports/compile-reduce-overhead-2026-02-02.md`

## 2026-02-02 — Activation checkpointing benchmark with torch.compile (default mode)

**Type:** Retrospective  
**General description:** Benchmarked no-AC vs selective AC under torch.compile with static shapes.

### Details

- Ran `eval/benchmark_train_step.py` in synthetic mode at batch_size=1, seq_len=1024 (warmup_steps=2, steps=5), `--compile-mode default`.
- Selective AC reduced peak VRAM by ~8.2% with ~17.4% throughput loss vs no-AC.
- Selective AC under torch.compile requires `allow_cache_entry_mutation=True` to avoid cached-tensor mutation errors.
- Activation checkpointing now auto-selects **selective** when compile is enabled; otherwise it uses manual checkpointing.
- Results saved to `benchmark_results/train_step_compile.jsonl`.

### Links

- Report: `training_reports/activation-checkpointing-compile-benchmark-2026-02-02.md`

## 2026-02-02 — Activation checkpointing benchmark (manual vs selective)

**Type:** Retrospective  
**General description:** Benchmarked no activation checkpointing vs manual vs selective (matmul/attention) with static shapes.

### Details

- Ran `eval/benchmark_train_step.py` in synthetic mode at batch_size=1, seq_len=1024 (warmup_steps=2, steps=5).
- Manual AC reduced peak VRAM by ~16.6% with ~19.2% throughput loss vs no-AC.
- Selective AC reduced peak VRAM by ~12.4% with ~33.3% throughput loss vs no-AC.
- Results saved to `benchmark_results/train_step.jsonl`.

### Links

- Report: `training_reports/activation-checkpointing-benchmark-2026-02-02.md`

## 2026-02-01 — Benchmark reflects train.py (no optimization flags)

**Type:** Observation  
**General description:** Removed benchmark-specific optimization flags so results reflect the current training setup.

### Details

- `eval/benchmark_train_step.py` now reads `TrainConfig.compile` from `models/config.py` and no longer accepts compile/mark-dynamic flags.
- Dynamic `(B,T)` marking is always applied when compile is enabled in `train.py` (no separate toggle).
- Documentation and the dynamic-shapes skill updated to match the “no knobs in benchmark” workflow.

## 2026-02-02 — Regional compile (per-block) + reduce-overhead benchmark

**Type:** Retrospective  
**General description:** Benchmarked per-block regional compile with `mode="reduce-overhead"` to cut compile latency while retaining MoMH throughput gains.

### Details

- Per-block regional compile (vision + decoder blocks + MP) kept enabled.
- `mode="reduce-overhead"` reduced compile_time_ms from ~30.8s to ~15.2s.
- MoMH throughput improved vs default compile and no-compile baselines at `seq_len=1024`.

### Links

- Report: `training_reports/compile-reduce-overhead-2026-02-02.md`

## 2026-02-01 — Regional torch.compile + MoMH block-mask graph break fix

**Type:** Retrospective
**General description:** Reduced compile-scope to regional submodules and removed a torch.compile graph break caused by MoMH block-mask construction inside the decoder.

### Details

- Switched `train.py`, `eval/benchmark_train_step.py`, and `eval/measure_vram.py` to **regional compile** (vision encoder, decoder, MP) instead of compiling the full VLM module.
- Added `compile_time_ms` to the training-step benchmark output (first step that triggers compilation).
- Fixed a graph break in `models/language_model.py` by moving MoMH prefill block-mask construction into the VLM wrapper (`models/vision_language_model.py`) and passing it into the decoder.

### Key Points

- Regional compile cuts compile scope and makes it easier to control graph breaks.
- MoMH block-mask creation inside a compiled decoder causes graph breaks because it was wrapped in `torch.compiler.disable`; constructing it outside the compiled region removes that break.
- Benchmark now reports `compile_time_ms` for faster comparison of compile latency changes.

### Links

- Benchmark: `eval/benchmark_train_step.py`
- VLM wrapper: `models/vision_language_model.py`
- Decoder: `models/language_model.py`

## 2026-01-30 — Torch.compile dynamic shapes + train-step benchmark

**Type:** Retrospective
**General description:** Reduce `torch.compile` recompiles from variable batch/sequence shapes while tracking throughput via a stable benchmark.

### Details

Added an opt-in `TrainConfig.compile_dynamic_shapes` switch that uses `torch._dynamo.maybe_mark_dynamic` on `(B,T)` dims for `input_ids`, `labels`, and `attention_mask` in the training and validation loops (removed 2026-02-01; dynamic marking is now always on when compile is enabled). This targets the common training failure mode where the collator drops too-long samples and produces variable batch sizes / sequence lengths, leading to recompilations.

Also recorded a set of “contended GPU” benchmark numbers at `seq_len=2048` to keep a reference point for shared-GPU throughput comparisons.

### Key Points

- Prefer `maybe_mark_dynamic` over `mark_dynamic` to avoid `ConstraintViolationError` when dims sometimes specialize to constants.
- Dynamic `(B,T)` reduces recompiles, but tile-count variance (vision input length) can still trigger recompiles.

### Links

- Report: `training_reports/2026-01-30_torch-compile_dynamic-shapes_and_benchmark.md`
