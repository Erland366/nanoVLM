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
| flex_attention dtype mismatch | `RuntimeError: dtype mismatch (float vs c10::Half)` | flex_attention returns float32 even with float16 inputs | Add `y = y.to(x.dtype)` after flex_attention call |
| flex_attention recompilation | Slow decode (~1s per token), `TORCH_LOGS=recompiles` shows size mismatch | `dynamic=False` recompiles for each unique KV length | Use `torch.compile(flex_attention, dynamic=True)` for decode |
| MoMH garbage output | Model produces nonsense during generation | MoMH attention not applied during decode phase | Pass `content_starts` and `position_offset` to decode, use `score_mod` |
| Input shape mismatch | `ValueError: not enough values to unpack` | Passing 2D token IDs when model expects 3D embeddings (`lm_use_tokens=False`) | Use embeddings `[B, T, D]` not token IDs `[B, T]` |
| BlockMask batch size | Errors when batch size changes between calls | BlockMask created with fixed batch size | Recreate BlockMask when batch size changes, or resize content_starts buffer |
| MoMH prefill graph break | `TORCH_LOGS=graph_breaks` shows `_build_momh_block_mask_prefill` | Block-mask creation inside compiled decoder (`torch.compiler.disable`) | Build block mask in VLM wrapper and pass `prefill_block_mask` into decoder |
| Blackwell CUDA kernel image | `CUDA error: no kernel image is available for execution on the device` | PyTorch build lacks sm_120 support | Install a CUDA 12.8+ PyTorch build (e.g., torch 2.10.0+cu128) |
| Checkpoint load missing state | `No trainer state found` or missing trainer fields on resume | PyTorch 2.6+ defaults `torch.load` to `weights_only=True` | Load trainer/checkpoint state with `weights_only=False` |
| `pack_sequences` has no effect | Loss/throughput differs from expected packed-vs-unpacked behavior | `ConstantLengthDataset` was always created with default `pack_sequences=True` | Pass `pack_sequences=train_cfg.pack_sequences` when constructing `ConstantLengthDataset` |

## flex_attention Debugging

### Check for recompilations
```bash
TORCH_LOGS="recompiles" python your_script.py
```

### Check for graph breaks
```bash
TORCH_LOGS="graph_breaks" python your_script.py
```

### Common recompilation triggers
- Tensor shape changes (use `dynamic=True` or pad to fixed shapes)
- Python control flow based on tensor values
- Printing/logging tensor values inside compiled functions
