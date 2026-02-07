---
name: selective-ac-policy-stability-nanovlm
description: >
  Stabilize selective activation checkpointing in nanoVLM under torch.compile.
  Use when: compile+AC yields non-finite grads, step-1 NaNs, or policy-dependent instability.
metadata:
  short-description: "Selective AC policy stability under compile"
  tags:
    - activation-checkpointing
    - torch-compile
    - nanovlm
    - stability
    - debugging
  domain: research
  created: 2026-02-06
  author: codex
---

# Selective AC Policy Stability (nanoVLM)

## General Description

This skill captures the selective activation-checkpointing instability observed under
`torch.compile` in nanoVLM and the validated policy fix. In this project, a selective
policy that saves generic matmul-family ops (`matmul_attention`) can produce non-finite
gradients, while an attention-only selective policy remains stable in tested runs.

## When to Apply

Use this knowledge when:
- `compile=True` and `activation_checkpointing=True`.
- `grad_norm=nan` appears at step 1 or gradients become non-finite in compiled runs.
- You are deciding which selective AC policy to use for compile-time training.

Do NOT use when:
- You are running no-compile training only (manual AC is the primary fallback path).

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| First non-finite step (`matmul_attention`, compile default) | 1 | `grad_norm` became NaN immediately |
| First non-finite step (`attention_only`, compile default) | None in tested window | Stable for 10-step and 20-step sweeps |
| `reduce-overhead` + selective AC | Runtime failure | CUDAGraph/storage errors still unresolved |

## Recommended Practice

### Step 1: Keep selective AC on compile path, but use attention-only saves

- Use policy `attention_only` for selective AC.
- Keep `allow_cache_entry_mutation=True` when selective AC is enabled under compile.
- Keep `compile_mode=default` for selective AC runs until reduce-overhead issues are resolved.

### Step 2: Use a minimal NaN sweep for quick verification

Run short synthetic checks after policy or compile changes:
- `compile default + AC off`
- `compile default + AC selective`
- `no compile + AC manual`
- `compile reduce-overhead + AC selective` (expected to expose known runtime issue)

Record first non-finite step (`loss`/`grad_norm`) and runtime errors.

### Step 3: If using AGENTS small debug config, keep token geometry consistent

With `vit_img_size=128` and `vit_patch_size=16`, set:
- `mp_image_token_length=4`

to avoid image placeholder mismatch errors during synthetic multimodal batches.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Selective policy `matmul_attention` under compile default | Non-finite grads from step 1 | Use `attention_only` selective policy |
| Selective AC with `allow_cache_entry_mutation=False` | Cached tensor mutation error | Keep `allow_cache_entry_mutation=True` |
| `compile_mode=reduce-overhead` + selective AC | CUDAGraph/storage runtime errors | Use `compile_mode=default` for selective AC |
| AGENTS small debug dims with default `mp_image_token_length=64` | Placeholder/token count mismatch | Set `mp_image_token_length=4` |

## Configuration

```yaml
# Stable compile + selective AC path
TrainConfig:
  compile: true
  compile_mode: default

VLMConfig:
  activation_checkpointing: true
  # default selective policy: attention_only

# AGENTS small debug profile consistency
VLMConfig:
  vit_hidden_dim: 256
  vit_inter_dim: 1024
  vit_patch_size: 16
  vit_img_size: 128
  vit_n_heads: 4
  vit_n_blocks: 4
  lm_hidden_dim: 384
  lm_inter_dim: 1024
  lm_n_heads: 6
  lm_n_kv_heads: 2
  lm_n_blocks: 8
  lm_max_length: 1024
  lm_tie_weights: true
  mp_image_token_length: 4
```

## References

- Related reports: `training_reports/activation-checkpointing-compile-benchmark-2026-02-02.md`,
  `training_reports/compile-reduce-overhead-2026-02-02.md`
- Related artifacts: `benchmark_results/activation_checkpointing_triage_2026-02-06_cuda.jsonl`,
  `benchmark_results/activation_checkpointing_nan_sweep_2026-02-06.json`,
  `benchmark_results/activation_checkpointing_nan_sweep_2026-02-06_agents_small.json`
- Related skills: `activation-checkpointing-compile-tradeoffs-nanovlm`
