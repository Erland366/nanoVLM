# Report: Activation checkpointing benchmark with torch.compile (default mode)

**Date:** 2026-02-02  
**Author:** codex  
**Status:** Completed

## Objective

Benchmark **no activation checkpointing** vs **selective activation checkpointing** under `torch.compile` with static shapes.

## Setup

### Environment
- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)
- Software: torch 2.10.0+cu128
- Mode: synthetic (train-step benchmark)
- Device: CUDA
- Dtype: bf16

### Configuration

```yaml
benchmark:
  mode: synthetic
  momh: true
  batch_size: 1
  seq_len: 1024
  steps: 5
  warmup_steps: 2
  dtype: bf16
compile:
  enabled: true
  mode: default
activation_checkpointing:
  enabled: true
  behavior: selective when compile is enabled
  policy: matmul_attention
  allow_cache_entry_mutation: true
```

Output: `benchmark_results/train_step_compile.jsonl`

## Results

| Case | step_time_ms_mean | tokens_per_second | peak_vram_mb | training_vram_mb | compile_time_ms |
| --- | --- | --- | --- | --- | --- |
| No AC | 43.93 | 23096.26 | 1228.66 | 643.69 | 11541.89 |
| Selective AC | 53.19 | 19086.47 | 1127.40 | 544.31 | 2696.20 |

## Observations

- Selective AC reduced **peak VRAM by ~8.2%** vs no-AC, with a **~17.4% throughput drop**.
- Selective AC under `torch.compile` required `allow_cache_entry_mutation=True` to avoid cached-tensor mutation errors.
- Compile time for the first step was lower on the selective AC run (likely due to compilation caching effects across runs).

## Notes

- These runs used `--compile-mode default` to avoid cudagraph errors seen in `reduce-overhead` mode.
- The JSONL entries include timestamps from the system clock (2026-02-02T03:20:xx) and full metadata.
