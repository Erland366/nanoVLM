---
name: torch-compile-policy
description: >
  Project defaults for torch.compile in nanoVLM. Use when changing compile
  strategy, debug scripts, or benchmarking defaults.
metadata:
  short-description: "nanoVLM torch.compile defaults"
  tags:
    - torch-compile
    - policy
    - nanoVLM
  created: 2026-01-25
---

# nanoVLM torch.compile Policy

## Defaults
- Prefer **regional compilation** (compile repeated LM/ViT blocks) as the default path.
- Keep “full-model / full-core compile” as an explicit opt-in path only.
- Treat “compile latency comparisons” (regional vs full-core) as **non-default** benchmarks; run only when requested.

## Rationale
- Regional compilation is consistently faster to compile and is the primary workflow we optimize for in this repo.
- Full-core compile is useful for research/debugging, but is not the default because it is slower to compile and adds iteration friction.

## Where This Applies
- `train.py`: default compile path should remain regional unless explicitly changed.
- `train_debug.py`: default diagnostic suite should not include compile-latency comparison; keep it opt-in.

## Notes
- This policy is about the compilation unit/strategy, not dynamic shape policy. Dynamic shape handling should remain explicit and scoped (batch/seq only) where applicable.
