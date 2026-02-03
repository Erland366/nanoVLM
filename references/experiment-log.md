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

## 2026-02-02 — MoMH uptraining plan (gradual mask ramp)

**Type:** Plan  
**General description:** Uptrain pretrained VLM with MoMH masks using a gradual head-split ramp to avoid optimization shock.

### Details

- **Precondition:** set `VLMConfig.vlm_load_backbone_weights=True` and point `VLMConfig.vlm_checkpoint_path` to the pretrained VLM (uptraining requires loading weights).
- **Phase A (warm start, 10–20% budget):** `momh_enabled=False`, low LR (~0.3× baseline). Optional: freeze vision encoder.
- **Phase B (ramp, 40–50% budget):** enable MoMH and ramp head splits to target in 2–3 stages:
  - B1: `pct_v=0.0`, `pct_t=0.0` (all VT)
  - B2: `pct_v=0.1`, `pct_t=0.15`
  - B3: `pct_v=0.2`, `pct_t=0.3` (target)
- **Phase C (hold, 30–40% budget):** keep target splits; use ~0.5× baseline LR (or 0.3× if unstable).
- **Monitoring:** watch loss spikes at mask switches, grad norms, and (optional) head usage/entropy.

## 2026-02-02 — Retrospective: effective-token LR scaling vs packed baseline

**Type:** Retrospective  
**General description:** Compared effective-token LR scaling variants against packed training to reduce padding-induced noise.

**Details:** Evaluated non-pack runs with effective-token LR scaling exponents (0.5, 1.0, 1.5, 2.0), step vs token scheduling, and EMA smoothing over a 500k-token budget. Best MAE vs packed came from exponent 1.5 step schedule (MAE ~1.54), with exponent 1.0 close behind (MAE ~1.56). Token-based scheduling and EMA did not improve results, and exponent 2.0 degraded performance. Non-pack still lags packed under the same token budget.

**Links:**  
- Report: `training_reports/effective-token-lr-scaling-2026-02-02.md`

## 2026-02-02 — Retrospective: MoMH packing document masking + benchmark

**Type:** Retrospective  
**General description:** Added per-token `document_ids` so MoMH supports packed sequences without cross-sample attention, and benchmarked overhead on GPU.

**Details:** Implemented document-aware MoMH masking by requiring `document_ids[q]==document_ids[kv]` for all head types (V/T/VT). This prevents cross-document leakage when `pack_sequences=True`. Benchmarked prefill-style `BlockMask` build and `flex_attention_compiled` on GPU at `B=1,H=15,T=2048,D=64` (fp16). BlockMask build cost increased slightly in the single-doc case (~+0.09 ms), while packed doc masking increased sparsity and significantly reduced attention kernel time. Also upgraded PyTorch to a cu128 build to support Blackwell (sm_120) after hitting “no kernel image is available” errors with older builds.

**Links:**  
- Report: `training_reports/momh-packing-document-masking-benchmark-2026-02-02.md`

## 2026-02-01 — Effective-token LR scaling

**Type:** Observation  
**General description:** Added optional LR scaling based on effective (non-padding) tokens per update step.

**Details:** Introduced `TrainConfig.effective_token_lr_scale` and `effective_token_lr_exponent`, applied the
scale after the scheduler for all LR groups, and logged `effective_tokens`, `effective_token_ratio`, and
`effective_token_lr_scale` each update step.

## 2026-02-01 — Token-based stopping

**Type:** Observation  
**General description:** Added an optional stop condition based on effective token count.

**Details:** Added `TrainConfig.max_training_tokens` and `--max_training_tokens` to stop training once the
global effective-token count reaches the requested budget.

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
