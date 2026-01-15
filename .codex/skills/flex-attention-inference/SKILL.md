---
name: flex-attention-inference
description: Patterns for using PyTorch flex_attention in inference (prefill vs decode phases)
metadata:
  short-description: flex_attention inference patterns
---

# Flex Attention Inference Patterns

## Goal

Guide correct implementation of `flex_attention` for transformer inference, handling both prefill and decode phases correctly.

## Key Concepts

### Prefill vs Decode

| Phase | Query Shape | KV Shape | Recommended Approach |
|-------|-------------|----------|---------------------|
| Prefill | `[B, H, T, D]` | `[B, H, T, D]` | `BlockMask` + `dynamic=False` |
| Decode | `[B, H, 1, D]` | `[B, H, T_kv, D]` | `score_mod` + `dynamic=True` |

### Why Different Approaches?

**Prefill**:
- Fixed sequence length (padded to max)
- `BlockMask` provides efficient sparse attention
- `dynamic=False` optimal for fixed shapes

**Decode**:
- KV cache grows by 1 each step
- `BlockMask` would need recreation each step
- `score_mod` with captured tensors avoids recompilation
- `dynamic=True` handles variable KV lengths

## Implementation Pattern

### 1. Compile Both Versions

```python
from torch.nn.attention.flex_attention import flex_attention

# Prefill: fixed shapes, best performance
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

# Decode: variable KV length, avoid recompilation
flex_attention_compiled_dynamic = torch.compile(flex_attention, dynamic=True)
```

### 2. Prefill with BlockMask

```python
def prefill_attention(q, k, v, mask_mod, seq_len):
    block_mask = create_block_mask(
        mask_mod, B=q.size(0), H=q.size(1),
        Q_LEN=seq_len, KV_LEN=seq_len, device=q.device
    )
    return flex_attention_compiled(q, k, v, block_mask=block_mask)
```

### 3. Decode with score_mod and Captured Tensors

```python
# Create buffers ONCE (captured by closure)
position_offset = torch.tensor(0, dtype=torch.int64, device="cuda")

def create_score_mod(position_offset):
    def score_mod(score, b, h, q_idx, kv_idx):
        # q_idx is 0 during decode, add offset for actual position
        actual_q_idx = q_idx + position_offset
        # ... masking logic using actual_q_idx ...
        return torch.where(valid, score, -inf)
    return score_mod

score_mod = create_score_mod(position_offset)

# During decode loop:
for step in range(max_tokens):
    position_offset.fill_(current_position)  # Update VALUE, not shape
    out = flex_attention_compiled_dynamic(q, k, v, score_mod=score_mod)
```

## Critical Rules

1. **Captured tensor VALUES can change** without triggering recompilation
2. **Captured tensor SHAPES cannot change** - will trigger recompilation
3. **Use `dynamic=True` for decode** - KV length grows each step
4. **Use `dynamic=False` for prefill** - fixed shapes, better performance
5. **Cast output dtype** - flex_attention may return float32:
   ```python
   y = flex_attention_compiled(q, k, v, ...)
   y = y.to(q.dtype)  # Cast back to input dtype
   ```

## Debugging

```bash
# Check for recompilations
TORCH_LOGS="recompiles" python script.py

# Check for graph breaks
TORCH_LOGS="graph_breaks" python script.py
```

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using `dynamic=False` for decode | ~1s per token, recompile logs | Use `dynamic=True` |
| Not passing position offset | Wrong attention pattern during decode | Add `position_offset` to score_mod |
| Creating new score_mod each step | Recompilation each step | Create once, update captured tensor values |
| Forgetting dtype cast | dtype mismatch errors | Add `y = y.to(x.dtype)` |

## Reference

- PyTorch flex_attention documentation
- `training_reports/momh_flex_attention_retrospective.md`
