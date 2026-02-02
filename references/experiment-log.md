# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-02-02 — Retrospective: effective-token LR scaling vs packed baseline

**Type:** Retrospective  
**General description:** Compared effective-token LR scaling variants against packed training to reduce padding-induced noise.

**Details:** Evaluated non-pack runs with effective-token LR scaling exponents (0.5, 1.0, 1.5, 2.0), step vs token scheduling, and EMA smoothing over a 500k-token budget. Best MAE vs packed came from exponent 1.5 step schedule (MAE ~1.54), with exponent 1.0 close behind (MAE ~1.56). Token-based scheduling and EMA did not improve results, and exponent 2.0 degraded performance. Non-pack still lags packed under the same token budget.

**Links:**  
- Report: `training_reports/effective-token-lr-scaling-2026-02-02.md`

## 2026-02-02 — Retrospective: activation checkpointing + compile tradeoffs

**Type:** Retrospective  
**General description:** Summarized activation checkpointing benchmarks and compile/recompile behavior under shape sweeps.

### Details

- Benchmarked manual vs selective activation checkpointing (no compile) and selective activation checkpointing under `torch.compile` in synthetic mode at `batch_size=1`, `seq_len=1024`.
- Selective AC under `torch.compile` requires `allow_cache_entry_mutation=True` to avoid cached-tensor mutation errors; activation checkpointing now auto-selects selective when compile is enabled.
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

### Links

- Benchmark: `eval/benchmark_train_step.py`
- VLM wrapper: `models/vision_language_model.py`
- Decoder: `models/language_model.py`

## 2026-02-01 — Benchmark reflects train.py (no optimization flags)

**Type:** Observation  
**General description:** Removed benchmark-specific optimization flags so results reflect the current training setup.

### Details

- `eval/benchmark_train_step.py` now reads `TrainConfig.compile` from `models/config.py` and no longer accepts compile/mark-dynamic flags.
- Dynamic `(B,T)` marking is always applied when compile is enabled in `train.py` (no separate toggle).

## 2026-02-01 — Token-based stopping

**Type:** Observation  
**General description:** Added an optional stop condition based on effective token count.

**Details:** Added `TrainConfig.max_training_tokens` and `--max_training_tokens` to stop training once the global effective-token count reaches the requested budget.

## 2026-02-01 — Effective-token LR scaling

**Type:** Observation  
**General description:** Added optional LR scaling based on effective (non-padding) tokens per update step.

**Details:** Introduced `TrainConfig.effective_token_lr_scale` and `effective_token_lr_exponent`, applied the scale after the scheduler for all LR groups, and logged `effective_tokens`, `effective_token_ratio`, and `effective_token_lr_scale` each update step.

## 2026-01-30 — Torch.compile dynamic shapes + train-step benchmark

**Type:** Retrospective  
**General description:** Reduce `torch.compile` recompiles from variable batch/sequence shapes while tracking throughput via a stable benchmark.

### Details

Added an opt-in `TrainConfig.compile_dynamic_shapes` switch that uses `torch._dynamo.maybe_mark_dynamic` on `(B,T)` dims for `input_ids`, `labels`, and `attention_mask` in the training and validation loops (removed 2026-02-01; dynamic marking is now always on when compile is enabled). This targets the common training failure mode where the collator drops too-long samples and produces variable batch sizes / sequence lengths, leading to recompilations.

### Links

- Report: `training_reports/2026-01-30_torch-compile_dynamic-shapes_and_benchmark.md`

## 2025-01-14 | Retrospective | MoMH Flex Attention Inference Fix

**General description**: Fixed MoMH attention to work during inference decode phase, achieving 4000x+ speedup.

**Problem**: Model produced garbage output during generation because MoMH attention was only applied during prefill, not decode. The model was trained with specialized head attention patterns (V-heads, T-heads, VT-heads) but fell back to vanilla SDPA during decode.

**Solution**:
1. Implemented `score_mod` with position offset for decode phase (vs `BlockMask` for prefill)
2. Used captured tensors to avoid recompilation when updating position values
3. Added `flex_attention_compiled_dynamic` with `dynamic=True` for decode to handle variable KV lengths

**Key Results**:
| Metric | Before | After |
|--------|--------|-------|
| Decode time/iter | 1036ms | 0.25ms |
| Recompilations | Every step | None |
| Test coverage | 0 | 21 tests |

**Files**: See `training_reports/momh_flex_attention_retrospective.md` for full details.

<!-- New entries go above this line -->
