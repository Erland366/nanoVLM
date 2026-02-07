---
name: activation-checkpointing-compile-tradeoffs-nanovlm
description: >
  Choose activation checkpointing mode (manual vs selective) for nanoVLM and understand compile tradeoffs.
  Use when: deciding between compile+AC vs no-compile AC for memory vs throughput.
metadata:
  short-description: "AC tradeoffs under compile/no-compile"
  tags:
    - activation-checkpointing
    - torch-compile
    - nanovlm
    - performance
    - memory
  domain: research
  created: 2026-02-02
  author: codex
---

# Activation Checkpointing Tradeoffs (nanoVLM)

## General Description

This skill captures observed tradeoffs between manual activation checkpointing and selective activation
checkpointing in nanoVLM, with and without torch.compile. It documents the stable selective policy
(`attention_only`), the compile-specific guard needed for selective AC, and known failure modes with
`reduce-overhead`.

## When to Apply

Use this knowledge when:
- You need to reduce VRAM and are deciding between manual AC and selective AC.
- You are using torch.compile and want to understand the throughput impact of AC.

Do NOT use when:
- You are only benchmarking fixed shapes and want maximum throughput (prefer no AC).

## Results Summary

| Scenario | Tokens/s | Peak VRAM (MB) | Notes |
|----------|----------|----------------|-------|
| No AC (compile default) | 8283 | 9047 | B=1, T=1024, synthetic (2026-02-06 rerun) |
| Selective AC (compile default, attention_only) | 7305 | 9020 | Stable; ~11.8% throughput drop vs no-AC (same rerun) |
| Manual AC (no compile) | 5300 | 9031 | Stable fallback; slower than compile paths (same rerun) |

## Recommended Practice

- If `compile=True` and `activation_checkpointing=True`, use selective AC with
  policy `attention_only` and enable `allow_cache_entry_mutation=True`.
- If `compile=False` and you need VRAM savings, manual AC outperformed selective AC at B=1, T=1024.
- Prefer `compile-mode=default` when AC is enabled; `reduce-overhead` triggered cudagraph issues in practice.
- Keep batch size fixed within a compile session to minimize recompiles in flex_attention block mask.
- In checkpointed loops, bind the current block in the closure (e.g., `def _run_block(x_in, _block=block): ...`)
  to avoid late-bound recomputation bugs in backward.
- For AGENTS small debug configs (`vit_img_size=128`), set `mp_image_token_length=4` to match reduced
  image token geometry (otherwise placeholder/token mismatch errors).

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Selective AC under compile without allow_mutation | Cached tensor mutation error | Must set allow_cache_entry_mutation=True |
| Selective AC policy `matmul_attention` under compile default | Non-finite grads/NaN on first optimizer step | Use `attention_only` policy (default) |
| AC + reduce-overhead | CUDAGraph-related errors | Use compile-mode default for AC runs |
| Selective AC (no compile) slower than manual | Policy overhead outweighs benefits | Use manual AC when not compiling |
| Checkpoint closure captures loop variable (`block`) late | Backward recompute can run the wrong block, causing non-finite grads/NaNs | Bind `block` via default arg in closure (`_block=block`) in LM/ViT loops |
| AGENTS small debug dims with default `mp_image_token_length=64` | `image placeholder count mismatch` | Set `mp_image_token_length=4` when `vit_img_size=128` |

## Configuration

```yaml
# Enable compile + selective AC (auto)
TrainConfig:
  compile: true
VLMConfig:
  activation_checkpointing: true
  # Default selective policy is now attention_only.

# AGENTS small debug profile consistency requirement
VLMConfig:
  vit_img_size: 128
  vit_patch_size: 16
  mp_image_token_length: 4
```

## References

- Related reports: `training_reports/activation-checkpointing-benchmark-2026-02-02.md`,
  `training_reports/activation-checkpointing-compile-benchmark-2026-02-02.md`,
  `training_reports/compile-reduce-overhead-2026-02-02.md`
- Related benchmark artifacts: `benchmark_results/activation_checkpointing_triage_2026-02-06_cuda.jsonl`,
  `benchmark_results/activation_checkpointing_nan_sweep_2026-02-06.json`,
  `benchmark_results/activation_checkpointing_nan_sweep_2026-02-06_agents_small.json`
- Code: `train.py`, `models/language_model.py`, `models/vision_transformer.py`, `models/activation_checkpointing.py`
