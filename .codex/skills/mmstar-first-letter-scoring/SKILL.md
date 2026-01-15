---
name: mmstar-first-letter-scoring
description: >
  MMStar exact-match scoring uses only the first character of the model output.
  Use when interpreting MMStar results or debugging suspiciously high chance-level scores.
metadata:
  short-description: "MMStar scoring gotcha: first-letter match"
  tags:
    - evaluation
    - mmstar
  domain: research
  created: 2026-01-14
  author: codex
---

# MMStar First-Letter Scoring

## General Description

MMStar scoring in lmms-eval uses a strict `exact_match()` that only inspects the first character of the model output. This can inflate scores when outputs are gibberish but start with A-D. This skill captures the observed evaluation behavior and why the vanilla results can look misleading.

## When to Apply

Use this knowledge when:
- MMStar scores look too high for untrained or step-0 checkpoints.
- Outputs are long strings or gibberish but still show non-zero accuracy.
- Checkpoint comparisons show unexpected regressions.

Do NOT use when:
- The task scorer explicitly parses answers beyond the first character.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| MMStar average (step 0) | ~0.23-0.24 | Chance-level from first-letter scoring, not real reasoning |

## Observations

- Sample outputs for step 0 were gibberish strings that still scored because the first character matched the gold option.
- The current eval pipeline does not enforce constrained, single-letter decoding.

## Open Questions

- Should we enforce letter-only decoding in the eval pipeline to avoid inflated baselines?
- Do we want to post-process outputs before scoring?

## References

- Related reports: `references/experiment-log.md`
- Troubleshooting: `references/troubleshooting.md`
- Scoring logic: `/home/coder/dotfiles/compiled_resources/nanoVLM/lmms-eval/lmms_eval/tasks/mmstar/utils.py`
