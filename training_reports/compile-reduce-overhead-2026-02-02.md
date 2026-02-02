# Report: Regional compile (per-block) + reduce-overhead benchmark

**Date:** 2026-02-02  
**Author:** codex  
**Status:** Completed

## Objective

Reduce torch.compile **cold-start latency** while keeping MoMH throughput gains, using **per-block regional compile** and `mode="reduce-overhead"`.

## Setup

### Environment
- Hardware: CUDA GPU (device: `CUDA_VISIBLE_DEVICES=1`, model not recorded)
- Software: PyTorch with `torch.compile` enabled (bf16 autocast)
- Dataset: synthetic batches via `eval/benchmark_train_step.py`

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
  regional_compile: per-block (vision encoder blocks, decoder blocks, MP)
  mode: reduce-overhead (when enabled)
```

## Experiments

### Run 1: MoMH + regional compile (reduce-overhead)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| compile | true |
| compile mode | reduce-overhead |
| momh | true |
| batch_size | 1 |
| seq_len | 1024 |

**Results:**
| Metric | Value |
|--------|-------|
| compile_time_ms | 15227.31 |
| step_time_ms_mean | 23.23 |
| tokens_per_second | 43776.10 |
| peak_vram_mb | 1051.66 |
| training_vram_mb | 542.71 |

**Observations:**
- CUDAGraphs warning about pending backward appeared but did not affect results.

### Run 2: MoMH + regional compile (default mode)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| compile | true |
| compile mode | default |
| momh | true |
| batch_size | 1 |
| seq_len | 1024 |

**Results:**
| Metric | Value |
|--------|-------|
| compile_time_ms | 30805.65 |
| step_time_ms_mean | 32.86 |
| tokens_per_second | 30868.90 |
| peak_vram_mb | 1230.28 |
| training_vram_mb | 645.19 |

### Run 3: MoMH + no compile

**Parameters:**
| Parameter | Value |
|-----------|-------|
| compile | false |
| momh | true |
| batch_size | 1 |
| seq_len | 1024 |

**Results:**
| Metric | Value |
|--------|-------|
| step_time_ms_mean | 37.60 |
| tokens_per_second | 26746.12 |
| peak_vram_mb | 1290.05 |
| training_vram_mb | 708.82 |

## Analysis

### What Worked
- `mode="reduce-overhead"` **halved compile time** vs default compile (~30.8s → ~15.2s).
- Throughput improved substantially vs default compile and no-compile baselines.

### What Failed
- Compile latency is still non-trivial (~15s), so first-step overhead remains significant.

### Key Insights
1. Per-block regional compile + reduce-overhead is the best tradeoff so far for compile latency and throughput.
2. MoMH + no-compile remains slower, so compile still provides meaningful gains even at small scale.

## Next Steps

- [ ] Consider compiling only decoder blocks to reduce compile latency further.
- [ ] Add `torch.compiler.cudagraph_mark_step_begin()` in the benchmark loop if CUDAGraphs warnings are distracting.

