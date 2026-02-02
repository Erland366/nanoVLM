# Report: Activation checkpointing benchmark (manual vs selective)

**Date:** 2026-02-02  
**Author:** codex  
**Status:** Completed

## Objective

Compare **no activation checkpointing**, **manual activation checkpointing**, and **selective activation checkpointing** (matmul/attention policy) with a static batch size and sequence length.

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
  enabled: false
activation_checkpointing:
  manual: torch.utils.checkpoint on each block
  selective: torch.utils.checkpoint + matmul/attention policy
```

Output: `benchmark_results/train_step.jsonl`

## Results

| Case | step_time_ms_mean | tokens_per_second | peak_vram_mb | training_vram_mb |
| --- | --- | --- | --- | --- |
| No AC | 52.11 | 19395.32 | 1290.05 | 708.82 |
| Manual AC | 64.75 | 15674.09 | 1075.32 | 489.92 |
| Selective AC | 78.52 | 12942.06 | 1130.14 | 543.55 |

## Observations

- Manual AC reduced **peak VRAM by ~16.6%** vs no-AC, with a **~19.2% throughput drop**.
- Selective AC reduced **peak VRAM by ~12.4%** vs no-AC, but throughput dropped **~33.3%**.
- In this run, manual AC saved more VRAM **and** outperformed selective AC in throughput.

## Notes

- The JSONL entries include timestamps from the system clock (2026-02-02T02:56:xx) and full metadata.
