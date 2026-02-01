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
| (Template) | Describe the error message or behavior | Root cause analysis | Step-by-step fix |
| MoMH prefill graph break | `TORCH_LOGS=graph_breaks` shows `_build_momh_block_mask_prefill` | Block-mask creation inside compiled decoder (`torch.compiler.disable`) | Build block mask in VLM wrapper and pass `prefill_block_mask` into decoder |
