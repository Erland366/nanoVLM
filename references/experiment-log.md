# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

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

---

<!-- New entries go above this line -->
