---
name: torch-compile-debug
description: Debug torch.compile recompilations and graph breaks
metadata:
  short-description: Debug torch.compile issues
---

# Torch Compile Debugging

## Goal

Identify and fix `torch.compile` recompilations and graph breaks that cause performance degradation.

## Quick Diagnosis

### Enable Logging

```bash
# See recompilations
TORCH_LOGS="recompiles" python script.py

# See graph breaks
TORCH_LOGS="graph_breaks" python script.py

# See both
TORCH_LOGS="recompiles,graph_breaks" python script.py
```

### Interpret Recompilation Logs

```
[__recompiles] Recompiling function flex_attention
    triggered by the following guard failure(s):
    - tensor 'key' size mismatch at index 2. expected 100, actual 101
```

This means: tensor shape changed, causing recompilation.

## Common Causes & Fixes

### 1. Shape Changes (Most Common)

**Symptom**: `tensor 'X' size mismatch at index N`

**Cause**: Tensor shapes change between calls with `dynamic=False`

**Fix**: Use `dynamic=True` for variable shapes
```python
# Before (recompiles for each shape)
fn_compiled = torch.compile(fn, dynamic=False)

# After (handles variable shapes)
fn_compiled = torch.compile(fn, dynamic=True)
```

### 2. Python Value Changes

**Symptom**: Recompilation when Python int/float changes

**Cause**: Python scalars are treated as constants

**Fix**: Wrap in tensor (captured tensor values can change)
```python
# Before (recompiles when offset changes)
def score_mod(score, b, h, q_idx, kv_idx):
    actual_q_idx = q_idx + python_offset  # BAD

# After (no recompilation)
offset_tensor = torch.tensor(0, device="cuda")
def score_mod(score, b, h, q_idx, kv_idx):
    actual_q_idx = q_idx + offset_tensor  # GOOD

# Update value (not shape) before each call
offset_tensor.fill_(new_value)
```

### 3. Data-Dependent Control Flow

**Symptom**: `Graph break: data-dependent`

**Cause**: Branching on tensor values
```python
if tensor.sum() > 0:  # BAD - data dependent
    ...
```

**Fix**: Use `torch.where` or restructure logic
```python
result = torch.where(tensor.sum() > 0, option_a, option_b)
```

### 4. Print/Logging Inside Compiled Function

**Symptom**: Graph break at print statement

**Cause**: `print()` is not compilable

**Fix**: Remove prints or use `torch._dynamo.config.suppress_errors`

## Performance Comparison Pattern

```python
import time
import torch

def benchmark(fn, *args, warmup=3, iterations=10):
    torch._dynamo.reset()

    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations

# Compare dynamic=False vs dynamic=True
fn_static = torch.compile(fn, dynamic=False)
fn_dynamic = torch.compile(fn, dynamic=True)

time_static = benchmark(fn_static, *args)
time_dynamic = benchmark(fn_dynamic, *args)
print(f"Static: {time_static*1000:.2f}ms, Dynamic: {time_dynamic*1000:.2f}ms")
```

## Dynamo Configuration

```python
import torch._dynamo

# Increase cache for many unique shapes (default 8)
torch._dynamo.config.cache_size_limit = 1000

# Reset compiled cache (useful for benchmarking)
torch._dynamo.reset()

# Suppress errors (use cautiously)
torch._dynamo.config.suppress_errors = True
```

## Decision Tree

```
Is the tensor shape fixed?
├── Yes → Use dynamic=False (best performance)
└── No → Does shape change frequently?
    ├── Yes → Use dynamic=True
    └── No (few unique shapes) → Consider:
        ├── Pad to fixed shapes, or
        └── Bucket sizes and pre-compile each
```

## Reference

- `benchmark/benchmark_recompilation.py` - Example benchmark
- `tests/test_momh_recompilation.py` - Diagnostic tests
- `training_reports/momh_flex_attention_retrospective.md` - Case study
