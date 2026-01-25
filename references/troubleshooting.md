# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Tokenizer graph break | Graph break on every forward pass, `TORCH_LOGS="graph_breaks"` shows tokenizer access | Calling `self.tokenizer.method()` or accessing tokenizer attributes in forward | Cache the value in `__init__`: `self._image_token_id = self.tokenizer.image_token_id` |
| Data-dependent control flow | Graph break at `if/else` statement, log shows "data-dependent" | Using `if tensor_val > threshold:` where tensor_val comes from tensor operation | Replace with `torch.where(condition, true_val, false_val)` |
| Python list recompilation | Recompile on batch size change, guard shows `len(images) == N` | Python list length is specialized by dynamo, `dynamic=True` only affects tensor shapes | Pre-process lists to tensors OUTSIDE compiled path, or compile only tensor-only `_forward_core` |
| Shape mismatch after graph break | `RuntimeError: shape mismatch` in boolean indexing | `@torch.compiler.disable` function receives wrong tensor shapes | Ensure disabled function inputs match expected shapes from surrounding compiled code |
| Batch/seq recompiles in LM blocks | `TORCH_LOGS=recompiles` shows guard failures on `x` size at batch or seq dims | Compiled LM blocks specialize on first-seen shapes | Mark batch and sequence dims as dynamic in `LanguageModel.forward` when `_compile_dynamic` is enabled (avoid calling mark_dynamic inside compiled blocks) |
| vectorized_gather_kernel assert in `train_debug.py --large-scale` | `CUDA error: device-side assert triggered` with `vectorized_gather_kernel index out of bounds` | Synthetic `input_ids` accidentally contains `image_token_id` outside intended image spans, increasing placeholder count beyond image embeddings | Sanitize synthetic `input_ids` after `torch.randint` (replace accidental `image_token_id`); see `_avoid_token_id_collisions` in `train_debug.py` |
