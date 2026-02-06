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
checkpointing in nanoVLM, with and without torch.compile. It also documents the compile-specific guard
needed for selective AC and known failure modes with reduce-overhead.

## When to Apply

Use this knowledge when:
- You need to reduce VRAM and are deciding between manual AC and selective AC.
- You are using torch.compile and want to understand the throughput impact of AC.

Do NOT use when:
- You are only benchmarking fixed shapes and want maximum throughput (prefer no AC).

## Results Summary

| Scenario | Tokens/s | Peak VRAM (MB) | Notes |
|----------|----------|----------------|-------|
| No AC (compile default) | 23096 | 1229 | B=1, T=1024, synthetic |
| Selective AC (compile default) | 19086 | 1127 | ~8.2% VRAM reduction, ~17.4% throughput loss |
| Manual AC (no compile) | 15674 | 1075 | ~16.6% VRAM reduction vs no-AC |

## Recommended Practice

- If `compile=True` and `activation_checkpointing=True`, use selective AC and enable
  `allow_cache_entry_mutation=True` (required to avoid cached-tensor mutation errors).
- If `compile=False` and you need VRAM savings, manual AC outperformed selective AC at B=1, T=1024.
- Prefer `compile-mode=default` when AC is enabled; `reduce-overhead` triggered cudagraph issues in practice.
- Keep batch size fixed within a compile session to minimize recompiles in flex_attention block mask.
- In checkpointed loops, bind the current block in the closure (e.g., `def _run_block(x_in, _block=block): ...`)
  to avoid late-bound recomputation bugs in backward.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Selective AC under compile without allow_mutation | Cached tensor mutation error | Must set allow_cache_entry_mutation=True |
| AC + reduce-overhead | CUDAGraph-related errors | Use compile-mode default for AC runs |
| Selective AC (no compile) slower than manual | Policy overhead outweighs benefits | Use manual AC when not compiling |
| Checkpoint closure captures loop variable (`block`) late | Backward recompute can run the wrong block, causing non-finite grads/NaNs | Bind `block` via default arg in closure (`_block=block`) in LM/ViT loops |

## Configuration

```yaml
# Enable compile + selective AC (auto)
TrainConfig:
  compile: true
VLMConfig:
  activation_checkpointing: true
```

## References

- Related reports: `training_reports/activation-checkpointing-benchmark-2026-02-02.md`,
  `training_reports/activation-checkpointing-compile-benchmark-2026-02-02.md`,
  `training_reports/compile-reduce-overhead-2026-02-02.md`
- Code: `train.py`, `models/language_model.py`, `models/vision_transformer.py`, `models/activation_checkpointing.py`
