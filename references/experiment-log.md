# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

<!-- New entries go above this line -->
## 2026-01-14
- **Type**: Retrospective
- **General description**: Summarize vanilla nanoVLM MMStar evaluations and scoring behavior.
- **Details**: Evaluated vanilla nanoVLM checkpoints (steps 0/5k/9.5k/11.5k) on MMStar with batch size 32; step 0 scored ~0.23-0.24, indicating chance-level performance. Inspecting sample outputs showed gibberish strings that still scored because MMStar's exact-match scoring uses the first character of the model output; random text starting with A-D can inflate scores and distort checkpoint comparisons. Constrained generation for single-letter outputs is not implemented/used in the current eval pipeline.
