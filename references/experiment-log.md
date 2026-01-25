# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-01-23 — Retrospective: torch.compile Optimization

**Type**: Retrospective
**General description**: Optimized nanoVLM for torch.compile, eliminating 2 HIGH-severity graph breaks and achieving 345x speedup after warmup.

### What we tried
- Static analysis with `TORCH_LOGS="graph_breaks"` to identify issues
- Cached tokenizer `image_token_id` at init time
- Replaced `if/else` with `torch.where` in RotaryEmbedding
- Tested `dynamic=True` for varying batch sizes
- Split `forward` into `_forward_core` (tensor-only) for clean compilation

### Key findings
- Tokenizer access in forward = graph break every step (cache at init)
- Data-dependent control flow = graph break (use `torch.where`)
- Python lists cause recompilation, not graph breaks (`dynamic=True` doesn't help)
- Compiling tensor-only `_forward_core` is best for varying batches

### Performance
- Speedup: 345x after compile warmup (3.90s → 0.011s per step)
- Graph breaks: 2 → 0 in hot path (1 intentional remains)

### Artifacts
- `torch_compile_optimization_report.md` - Full analysis
- `train_debug.py` - Debug script with `--compile-core`, `--dynamic`, `--vary-batch`
- `.codex/skills/torch-compile-vlm-patterns/SKILL.md` - Result skill

---

## 2026-01-24 — Retrospective: torch.compile latency + regional compilation

**Type**: Retrospective
**General description**: Focused on compile latency and dynamic-shape hygiene for VLM training/debug.

### What we tried
- Switched debug compiles to `fullgraph=True` and `mode="reduce-overhead"`
- Replaced `dynamic=True` with `mark_dynamic`/`maybe_mark_dynamic` on batch dims
- Added `mark_unbacked` guards when available to avoid 0/1 batch specialization
- Benchmarked regional compilation (LM + ViT blocks) vs full `_forward_core`
- Preferred `TORCHDYNAMO_DISABLE=1` for non-compile baselines
- Made regional compilation the default compile path for training
- Tried dynamic marking around regional blocks to reduce recompiles

### Key findings
- `mark_unbacked` is unavailable in the current PyTorch build, so batch=1 still triggers a recompile guard
- Regional compilation cut first-step compile latency from 27.80s to 6.17s in debug config
- `maybe_mark_dynamic` avoids constraint violations when a dimension becomes static
- Calling `maybe_mark_dynamic` inside compiled blocks is forbidden; dynamic markers must run in non-compiled frames
- Regional blocks still specialize on batch size when batches vary; use `dynamic=True` for blocks or keep batch size fixed

### Artifacts
- `train_debug.py` - Added `--benchmark-compile` harness
- `models/vision_language_model.py` - `maybe_mark_dynamic` and guarded `mark_unbacked`
- `torch_compile_optimization_report.md` - Updated settings and benchmark results
- Global skill: `/home/coder/dotfiles/skills/torch-compile-vlm-compile-latency/SKILL.md`

---

## 2026-01-25 — Retrospective: Global torch-compile Domain Retrospective

**Type**: Retrospective
**General description**: Conducted domain-wide retrospective on 3 days of torch.compile optimization, created new skill documenting PyTorch limitations.

### What worked
- Graph break fixes: cache at init, torch.where, cumsum+where
- Regional compilation: 4.5x faster than full model compile
- Comprehensive debug script with opt-out flags
- Proper baseline comparison methodology

### What failed / limitations discovered
- `mark_unbacked` unavailable in stable PyTorch (batch 1/2 specialization)
- Large sequence variations (4x) trigger sympy bugs
- `mark_dynamic` inside compiled blocks is forbidden
- TORCH_TRACE must be set before script runs
- Regional compile doesn't inherit dynamic marking

### New skill created
- `~/dotfiles/skills/torch-compile-known-limitations/SKILL.md`
- Documents 7 PyTorch-side limitations with workarounds

---

## 2026-01-25 — Retrospective: Comprehensive Debug Script Defaults

**Type**: Retrospective
**General description**: Refactored train_debug.py to run all diagnostics by default, adding vary-sequence test and making VRAM measurement always-on.

### What we tried
- Made all diagnostics run by default (opt-out instead of opt-in)
- Added vary-sequence test to verify `mark_dynamic` on sequence dimension
- Added graph break check using `fullgraph=True` (non-failing diagnostic)
- Added baseline comparison (eager vs compiled) with speed AND VRAM
- Added opt-out flags: `--quick`, `--skip-baseline`, `--skip-graph-check`, `--skip-compile-bench`

### Key findings
- Developers forget to enable diagnostic flags → make diagnostics default
- VRAM measurement has no overhead → always include it
- Sequence length variance is as important as batch variance for dynamic marking
- `fullgraph=True` catches graph breaks early (use as non-failing check)
- Regional compilation = 4.5x faster compile time than full model

### Default diagnostic suite (runs with `python train_debug.py`):
1. Graph break check (fullgraph=True)
2. Baseline comparison (eager vs compiled: speed + VRAM)
3. Vary batch test (mark_dynamic on dim 0)
4. Vary sequence test (mark_dynamic on dim 1)
5. Compile latency comparison (regional vs core)

### Artifacts
- `train_debug.py` - Complete rewrite with comprehensive defaults
- `torch-compile-debug-script/SKILL.md` - Updated with new structure
- `torch-compile-optimize/SKILL.md` - Updated output format

---

## 2026-01-25 — Enhancement: Multi-Config Ablation and Large-Scale Benchmarks

**Type**: Observation
**General description**: Added support for testing multiple inductor configs at once and large-scale model benchmarks for more accurate measurements.

### What we added

- **Multi-config support** (`--ablate-config "config1;config2"`):
  - Parse `"epilogue_fusion;max_autotune"` to test multiple configs
  - Support explicit values: `"epilogue_fusion=True;max_autotune=False"`
  - Tests each config individually if no value specified (True/False)

- **Large-scale benchmarks** (`--large-scale`):
  - Works with ANY diagnostic mode, not just config ablation
  - Increases model dimensions: 768d LM, 12 blocks; 512d ViT, 8 blocks
  - Uses larger batch size (min 8) for better GPU utilization
  - Runs more steps (min 20) for stable measurements
  - Total ~30M params vs debug ~5M params

### Usage examples

```bash
# Large-scale comprehensive diagnostics
python train_debug.py --large-scale

# Multiple configs with large-scale
python train_debug.py --ablate-config "epilogue_fusion;max_autotune" --large-scale

# Large-scale baseline only
python train_debug.py --skip-graph-check --skip-compile-bench --quick --large-scale
```

### Artifacts
- `train_debug.py` - Added `_create_large_scale_config()`, `_parse_config_spec()`, `--large-scale` flag
- `torch-compile-debug-script/SKILL.md` - Updated config ablation section

---

<!-- New entries go above this line -->
