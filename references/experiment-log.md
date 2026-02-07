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

## 2026-02-06 — Observation: batch_loss now logs optimizer-step loss

**Type:** Observation  
**General description:** Re-mapped default loss logging so `batch_loss` uses the optimizer-step token-normalized loss and moved last-microbatch loss to `micro_loss`.

### Details

- `batch_loss` now logs `step_update_loss` (token-normalized loss across the full accumulation window).
- Added `micro_loss` metric that logs the previous behavior (last micro-batch loss at update step).
- Kept `update_loss` as a compatibility alias of the same optimizer-step loss value.
- Updated README metric descriptions accordingly.

## 2026-02-06 — Observation: removed EMA loss logging from W&B

**Type:** Observation  
**General description:** Removed EMA-smoothed loss logging to keep charts fully raw and unbiased.

### Details

- Removed `update_loss_ema` metric definition and logging from `train.py`.
- Removed EMA-related config fields from `TrainConfig`:
  - `log_update_loss_ema`
  - `update_loss_ema_beta`
- Removed matching CLI flags from `train.py`.
- Kept `update_loss` logging (token-normalized optimizer-step loss) and existing `batch_loss` logging.

## 2026-02-06 — Observation: log optimizer-step loss and EMA in W&B

**Type:** Observation  
**General description:** Added less noisy loss metrics that match optimizer-step behavior under gradient accumulation.

### Details

- Added `update_loss` logging in `train.py` as token-normalized optimizer-step loss (`total_loss_sum / total_loss_tokens`) computed at each update step.
- Added optional `update_loss_ema` logging (default enabled) with new config fields:
  - `TrainConfig.log_update_loss_ema` (default `True`)
  - `TrainConfig.update_loss_ema_beta` (default `0.98`, validated in `[0,1)`).
- Kept existing `batch_loss` logging unchanged for backward compatibility.
- Updated W&B metric definitions so `update_loss` and `update_loss_ema` follow the same configured x-axis step metric.

## 2026-02-06 — Observation: set default GA to 2 for bs4 parity test

**Type:** Observation  
**General description:** Adjusted default accumulation so `batch_size=4` matches prior effective batch size 8 for fair loss-convergence comparison.

### Details

- Changed `TrainConfig.gradient_accumulation_steps` default from `8` to `2` while keeping `batch_size=4`.
- This restores effective global batch to `4 x 2 x 1 = 8`, matching prior single-GPU runs with `bs=1, ga=8`.
- Updated README default-config note to reflect `batch_size=4`, `grad accum=2`.

## 2026-02-06 — Observation: default train batch config switched to bs4/ga8

**Type:** Observation  
**General description:** Updated repo defaults to test loss behavior with larger per-step microbatch while keeping gradient accumulation at 8.

### Details

- Changed `TrainConfig.batch_size` default from `1` to `4`.
- Kept `TrainConfig.gradient_accumulation_steps` at `8`.
- Updated README default-config note to match current defaults.

## 2026-02-06 — Observation: W&B effective-token x-axis restored

**Type:** Observation  
**General description:** Restored stable token-based charting by logging cumulative `effective_tokens` on every main training log event.

### Details

- `train.py` now uses cumulative `effective_tokens` as the default W&B step metric.
- `tokens/consumed` remains available as the default step metric when `TrainConfig.wandb_xaxis_tokens=True`.
- Per-update effective tokens were renamed to `effective_tokens_step` so the cumulative axis metric remains monotonic and usable as x-axis.
- Applied to validation, training stats, batch loss, and epoch logs.

## 2026-02-06 — Observation: compile stability fix for cudagraph pool errors

**Type:** Observation  
**General description:** Stabilized `compile=True` training loop for CUDA by aligning compile mode controls and cudagraph boundaries.

### Details

- Added `TrainConfig.compile_mode` (`default|reduce-overhead|max-autotune`) and CLI `--compile_mode` to control regional `torch.compile` mode directly in `train.py`.
- Added per-microstep `torch.compiler.cudagraph_mark_step_begin()` when `compile=True` on CUDA to avoid cudagraph pool lifetime/accounting failures during training loops with recompute/checkpointing.
- Made `activation_memory_budget` wiring robust across PyTorch builds by checking `torch._dynamo.config` first and falling back to `torch._functorch.config`.

## 2026-02-06 — Retrospective: NaN debug + token-normalized accumulation fix

**Type:** Retrospective  
**General description:** Traced unstable/poor pretraining loss to activation-checkpointing closure correctness and gradient-accumulation loss normalization mismatch.

### Details

- **NaN root cause identified:** activation-checkpointed block loops in ViT/LM used late-bound loop closures, which can recompute the wrong block in backward.
- **Fix applied:** bind each checkpoint closure to the current loop block (`_block=block`) in:
  - `models/vision_transformer.py`
  - `models/language_model.py`
- **Verification:** with `activation_checkpointing=False`, runs stayed finite; with the buggy AC path, first optimizer updates produced non-finite vision grads.
- **Convergence discrepancy vs baseline:** compared `flex_attention` to `DualTowerVLM` and found a training-path mismatch:
  - baseline uses token-count-normalized accumulation (`loss_reduction="sum"` + valid-token count + grad rescale by total valid tokens),
  - current branch used per-microbatch mean CE accumulation.
- **Fix applied:** switched `flex_attention` training to token-normalized accumulation in `train.py`, and extended `VisionLanguageModel.forward(...)` to support `loss_reduction` and `return_loss_count`.
- **Runs executed:**
  - `xw54sjrv` (long no-compile MoMH run) reached ~2927 steps before manual stop.
  - `3yn8ei95` (patched token-normalized run, 500-step target) reached ~278 steps with finite loss before manual stop.

### Links

- W&B runs: `xw54sjrv`, `3yn8ei95`
- Troubleshooting updates: `references/troubleshooting.md`

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
