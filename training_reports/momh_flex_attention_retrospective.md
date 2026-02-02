# MoMH Flex Attention: Inference Fix & Recompilation Optimization

**Date**: 2025-01-14
**Status**: Completed
**Impact**: Critical fix for inference, 4000x+ speedup for decode phase

---

## Executive Summary

Fixed MoMH (Mixture of Modality Heads) attention to work correctly during inference (decode phase) and optimized torch.compile settings to eliminate recompilation overhead. The model was producing garbage output during generation because MoMH attention patterns were only applied during prefill, not decode.

---

## Problem Statement

### Initial Symptoms
- Model produced garbage output during inference: `" B1130001331366000\n13366000133330030333311111330333"`
- Expected: Proper letter answers (A, B, C, D) for multiple choice questions

### Root Cause Analysis
1. **Train/Inference Mismatch**: MoMH attention (using `flex_attention` with `BlockMask`) was only applied during prefill phase
2. **Decode Phase Fallback**: During decode, `content_starts=None` was passed, causing standard SDPA attention to be used
3. **Incompatible Attention Patterns**: Model was trained with MoMH head specialization:
   - V-heads (20%): Vision-only attention
   - T-heads (30%): Text-only causal attention
   - VT-heads (50%): Cross-modal attention
4. Using vanilla attention during decode broke the learned attention patterns

---

## Solution: Phase 1 - Correct MoMH Decode

### Technical Approach
The key insight from `flex_attention` documentation:
- **Prefill**: Use `BlockMask` for efficient sparse attention (fixed shape)
- **Decode**: Use `score_mod` with position offset (variable KV length)

During decode:
- Query tensor has shape `[B, H, 1, D]` (single new token)
- KV cache has shape `[B, H, T_kv, D]` where T_kv grows each step
- `q_idx` in score_mod is always 0, but actual position needs offset

### Implementation

**New function in `models/momh_attention.py`:**
```python
def generate_momh_score_mod_with_offset(
    n_q_heads: int,
    S_V: int,
    content_starts: torch.Tensor,  # Captured tensor [B]
    position_offset: torch.Tensor,  # Captured scalar tensor
    pct_v: float = 0.4,
    pct_t: float = 0.4
):
    """
    Uses CAPTURED TENSORS so value changes don't trigger recompilation.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        actual_q_idx = q_idx + position_offset  # Apply offset
        # ... MoMH masking logic using actual_q_idx
        return torch.where(valid_mask, score, -inf)
    return score_mod
```

**Key insight**: Captured tensor VALUES can change without triggering recompilation. Only SHAPES trigger recompilation.

**Changes to `models/language_model.py`:**
- Added `position_offset` parameter to forward methods
- Created `_get_momh_decode_score_mod()` for lazy initialization
- Added decode path using `score_mod` instead of `BlockMask`

**Changes to `models/vision_language_model.py`:**
- Updated `generate()` to pass `content_starts` and `position_offset` during decode

### Verification
Created comprehensive tests in `tests/`:
- `test_momh_attention.py`: Prefill/decode finite output tests
- `test_momh_correctness.py`: Comparison against manual vanilla implementation

All 21 tests pass, confirming correctness.

---

## Solution: Phase 2 - Recompilation Optimization

### Problem Discovery
Running with `TORCH_LOGS="recompiles"` revealed:
```
tensor 'key' size mismatch at index 2. expected 100, actual 101
tensor 'key' size mismatch at index 2. expected 101, actual 102
...
```

**Every decode step with growing KV cache triggered recompilation!**

### Root Cause
`flex_attention_compiled = torch.compile(flex_attention, dynamic=False)`

With `dynamic=False`:
- Tensor shapes are treated as constants
- Each unique shape requires a new compiled kernel
- KV cache grows by 1 each decode step = new compilation each step

### Solution
Added separate compiled function for decode:
```python
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)  # Prefill
flex_attention_compiled_dynamic = torch.compile(flex_attention, dynamic=True)  # Decode
```

With `dynamic=True`:
- Shapes are treated as symbolic
- Single compiled kernel handles all KV lengths
- No recompilation during decode

### Performance Impact

| Phase | Mode | Time per Iteration | Recompilations |
|-------|------|-------------------|----------------|
| Decode | `dynamic=False` | 1036.76ms | Every step |
| Decode | `dynamic=True` | 0.25ms | None |
| **Speedup** | | **4215x** | |

| Phase | Mode | Time per Iteration |
|-------|------|-------------------|
| Prefill (same shape) | `dynamic=False` | 0.17ms |
| Prefill (first run) | `dynamic=False` | 2437ms (compile) |

### Recommendation
- **Prefill**: Use `dynamic=False` (fixed shapes, optimal performance)
- **Decode**: Use `dynamic=True` (variable KV length, avoid recompilation)

---

## Files Modified

| File | Changes |
|------|---------|
| `models/momh_attention.py` | Added `generate_momh_score_mod_with_offset()`, `flex_attention_compiled_dynamic` |
| `models/language_model.py` | Added `position_offset` parameter, `_get_momh_decode_score_mod()`, decode path with MoMH |
| `models/vision_language_model.py` | Updated `generate()` to pass MoMH params during decode |

## Files Created

| File | Purpose |
|------|---------|
| `tests/test_momh_attention.py` | Pytest tests for prefill/decode finite output |
| `tests/test_momh_correctness.py` | Correctness comparison against manual implementation |
| `tests/test_momh_recompilation.py` | Diagnostic tests for recompilation detection |
| `benchmark/benchmark_recompilation.py` | Performance benchmark for recompilation impact |

---

## Lessons Learned

### 1. Captured Tensors in flex_attention
**Value changes don't trigger recompilation, only shape changes do.**
- Create buffer tensors once with correct shape
- Update values in-place before each forward call
- Score_mod closure captures the tensor reference, not the value

### 2. dynamic=True vs dynamic=False
- `dynamic=False`: Best for fixed shapes (prefill with padded sequences)
- `dynamic=True`: Required for variable shapes (decode with growing KV)
- Can use both in same codebase for different phases

### 3. Debugging Recompilation
```bash
TORCH_LOGS="recompiles,graph_breaks" python script.py
```
Shows exactly what triggers recompilation and why.

### 4. flex_attention Inference Pattern
```
Prefill:  BlockMask + dynamic=False  (efficient sparse attention)
Decode:   score_mod + dynamic=True   (position offset, no recompile)
```

---

## Future Considerations

1. **FlexDecoding with PagedAttention**: Could further optimize memory for long sequences
2. **Bucket KV Lengths**: Pre-compile for specific lengths (100, 200, 300...) to get `dynamic=False` performance
3. **Graph Breaks**: Monitor for data-dependent operations that break torch.compile graphs

---

## References

- PyTorch flex_attention documentation
- `compiled_resources/nanoVLM/programming_model_recompilation.md`
- `compiled_resources/nanoVLM/programming_model_common_graph_breaks.md`
- attention-gym repository examples
