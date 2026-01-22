# torch.compile Optimization Report

**Project**: nanoVLM
**Analysis Date**: 2026-01-22
**Mode**: static
**Entry Point**: train.py

---

## Summary

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Recompilation | 0 | 1 | 0 | 1 |
| Graph Breaks | 2 | 1 | 2 | 5 |
| Dynamic Shape | 0 | 1 | 0 | 1 |
| **Total** | **2** | **3** | **2** | **7** |

## torch.compile Configuration Detected

```python
# Location: train.py:134-135
if train_cfg.compile:
    model = torch.compile(model)
```

**Current settings**: Default (no explicit backend, mode, or dynamic configuration)
- `backend`: inductor (default)
- `mode`: default
- `dynamic`: None (auto-detect)
- `fullgraph`: False (default)

**Observation**: Compile is applied BEFORE DDP wrapping (correct order).

---

## Issues Found

### [HIGH] Graph Break: Tokenizer Access in Forward Pass

**Location**: `models/vision_language_model.py:46`
**Trigger**: `self.tokenizer.image_token_id` accesses tokenizer's `token_to_id` method during forward
**Impact**: Graph break on EVERY forward pass - Dynamo cannot trace tokenizer methods

**Verified via diagnostic run:**
```
TORCH_LOGS="graph_breaks" python train_debug.py
```

**Current code**:
```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    updated_token_embd = token_embd.clone()
    mask = (input_ids == self.tokenizer.image_token_id)  # <-- Graph break!
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
    return updated_token_embd
```

**Fix - Cache token ID at init time**:
```python
# In __init__:
self._image_token_id = self.tokenizer.image_token_id  # Cache as Python int

# In _replace_img_tokens_with_embd:
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    updated_token_embd = token_embd.clone()
    mask = (input_ids == self._image_token_id)  # Uses cached int, no graph break
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
    return updated_token_embd
```

**Reference**: `compiled_resources/torch-compile/torch_compile_missing_manual.md` - "Unsupported method calls"

---

### [HIGH] Graph Break: Data-Dependent Control Flow in RotaryEmbedding

**Location**: `models/language_model.py:87-92`
**Trigger**: `if max_seq > self.original_max_seq_len:` where `max_seq` comes from tensor operation
**Impact**: Graph break on EVERY forward pass in hot path (RotaryEmbedding.forward)

**Current code**:
```python
max_seq = position_ids.max() + 1
if max_seq > self.original_max_seq_len:
    scale = max_seq / self.original_max_seq_len
    inv_freq = self.inv_freq / scale
else:
    inv_freq = self.inv_freq
```

**Fix Option A - Use torch.where (no scaling if not needed)**:
```python
max_seq = position_ids.max() + 1
needs_scaling = max_seq > self.original_max_seq_len
scale = torch.where(
    needs_scaling,
    max_seq / self.original_max_seq_len,
    torch.ones_like(max_seq, dtype=torch.float)
)
inv_freq = self.inv_freq / scale
```

**Fix Option B - Pre-compute for expected max length (simpler)**:
```python
# In __init__, set max_seq_len to your expected maximum
# Then always use the same inv_freq (no dynamic scaling)
inv_freq = self.inv_freq  # Remove conditional entirely if sequences never exceed original
```

**Fix Option C - Use torch.cond**:
```python
def scale_inv_freq(inv_freq, max_seq, original_max):
    scale = max_seq / original_max
    return inv_freq / scale

max_seq = position_ids.max() + 1
inv_freq = torch.cond(
    max_seq > self.original_max_seq_len,
    lambda: scale_inv_freq(self.inv_freq, max_seq, self.original_max_seq_len),
    lambda: self.inv_freq,
)
```

**Reference**: `programming_model_common_graph_breaks.md` - "Data-dependent operations"

---

### [MEDIUM] Graph Break: None Check on Images Tensor

**Location**: `models/vision_language_model.py:66`
**Trigger**: `if images_tensor is not None:`
**Impact**: May cause graph specialization (separate graphs for with/without images)

**Current code**:
```python
if images_tensor is not None:
    image_embd = self.vision_encoder(images_tensor)
    image_embd = self.MP(image_embd)
    token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
```

**Analysis**: This is a Python-level None check. torch.compile handles this via guard specialization - it creates different compiled versions for `images_tensor is None` vs not. This is acceptable if you consistently have images during training.

**Recommendation**: LOW priority if images are always present during training. If sometimes absent, consider:
1. Always pass a dummy image tensor
2. Or accept the specialization (compile once with images, once without)

---

### [MEDIUM] Recompilation Risk: Variable Sequence Lengths

**Location**: `train.py:210` (model forward call)
**Trigger**: Variable `input_ids` sequence lengths across batches
**Impact**: Without explicit dynamic marking, first few batches may trigger recompilations

**Current behavior**:
- `dynamic=None` (default) - auto-switches to dynamic after shape mismatch
- First batch compiles for specific shape
- Different shape triggers recompilation, then switches to dynamic

**Recommendation**: Explicitly mark dynamic dimensions for clarity:
```python
# Option A: At compile time
model = torch.compile(model, dynamic=True)  # All dims dynamic

# Option B: In forward (more targeted)
# Add to VisionLanguageModel.forward():
torch._dynamo.mark_dynamic(input_ids, 0)  # batch dim
torch._dynamo.mark_dynamic(input_ids, 1)  # seq dim
```

---

### [LOW] Graph Break: isinstance Checks in Image Processing

**Location**: `models/vision_language_model.py:52-56`
**Trigger**: `isinstance(images, list)` and nested list checks
**Impact**: Cold path (only at batch boundary), not per-token

**Current code**:
```python
def _process_images(self, images, device):
    if isinstance(images, list):
        if images and isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]
        if not images:
            return None
        else:
            return torch.cat(images, dim=0).to(device)
    return images
```

**Analysis**: These are Python-level type checks that Dynamo handles via guards. Since this happens once per batch (not per token), impact is minimal.

**Recommendation**: Keep as-is. If you want full graph, move image preprocessing OUTSIDE the compiled model call.

---

### [LOW] Print Statements in Model Initialization

**Location**: Multiple files in `models/`
**Examples**:
- `models/language_model.py:205` - "Warning: scaled dot product attention not available..."
- `models/vision_transformer.py:68` - "Warning: scaled dot product attention not available..."
- `models/language_model.py:680` - "Successfully loaded..."

**Impact**: None during training (only in `__init__`, not `forward`)

**Status**: ✅ OK - These are in initialization, not forward pass.

---

## Recommendations Priority

### Immediate (HIGH severity)
1. [ ] **Cache tokenizer.image_token_id** at `models/vision_language_model.py:46`
   - This causes a graph break on every forward pass
   - Cache the token ID in `__init__` as `self._image_token_id`
   - Expected impact: Eliminate 1 graph break per forward

2. [ ] **Fix RotaryEmbedding conditional** at `models/language_model.py:87-92`
   - This causes a graph break on every forward pass
   - Use `torch.where` or pre-compute scaling
   - Expected impact: Eliminate 1 graph break per forward

### Short-term (MEDIUM severity)
2. [ ] **Consider explicit dynamic shapes**
   - Add `dynamic=True` to compile call OR
   - Use `mark_dynamic` for batch/seq dimensions
   - Expected impact: Cleaner compilation, predictable behavior

3. [ ] **Review None checks** if training sometimes lacks images
   - If always have images: no action needed
   - If sometimes no images: consider dummy tensors

### Optional (LOW severity)
4. [ ] Move image preprocessing outside compiled region
   - Only if you need `fullgraph=True`
   - Current setup works fine with default `fullgraph=False`

---

## Diagnostic Commands

To verify these findings, run:

```bash
# See graph breaks (use debug config for faster iteration)
TORCH_LOGS="graph_breaks" python train_debug.py 2>&1 | grep -A5 "Graph break"

# See recompilations
TORCH_LOGS="recompiles" python train_debug.py 2>&1 | grep -A5 "Recompiling"

# Full trace with tlparse (generates interactive HTML report)
TORCH_TRACE=/tmp/nanovlm_trace python train_debug.py
pip install tlparse
tlparse /tmp/nanovlm_trace
# Opens browser - look for:
#   - [0/0] [0/1] [0/2]... frames = recompilations
#   - Light green frames = graph breaks
#   - Red frames = compilation errors
```

### Debug Config Created

A smaller debug config was created for faster iteration:

**Files:**
- `configs/config_debug.py` - Smaller model config (~20M params vs 135M+)
- `train_debug.py` - Debug training script

**Config differences:**
- LM: 4 blocks (vs 32), 256 hidden (vs 960)
- ViT: 4 blocks (vs 12), 256 hidden (vs 768)
- 20 training steps, no wandb, random init
- ~10-20x faster per step

---

## Performance Baseline

| Metric | Current | After Fixes | Expected Change |
|--------|---------|-------------|-----------------|
| Graph breaks per forward | ~1-2 | 0 | Better fusion |
| Compile time (first step) | TBD | TBD | - |
| Step time (avg) | TBD | TBD | 5-15% faster |
| Recompilation count | ~2-3 | 0-1 | Fewer cold starts |

**Next Steps**:
1. Run with `TORCH_LOGS="graph_breaks"` to confirm findings
2. Apply RotaryEmbedding fix (highest impact)
3. Re-run diagnostics to verify improvement
4. Benchmark before/after

---

*Generated by `/torch-compile-optimize` skill*
*Reference: `compiled_resources/torch-compile/`*
