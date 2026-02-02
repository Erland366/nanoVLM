# Experiment: MoMH packing document masking benchmark

- **Date**: 2026-02-02
- **Author**: Codex-assisted
- **Goal**: Add document masking to MoMH so packed sequences (multiple documents per sequence) do not leak attention across packed samples, and quantify any runtime overhead.
- **General description**: Introduced per-token `document_ids` and enforced same-document attention in MoMH, then benchmarked the prefill-style `BlockMask` build and `flex_attention` kernel time on GPU.
- **Models**: `models/momh_attention.py` (flex_attention + BlockMask), nanoVLM decoder attention path
- **Datasets**: synthetic (microbench)

---

## 1. Setup

### 1.1 Model & task

- Task: measure overhead of adding `document_ids` constraint to MoMH masking.
- MoMH pattern: V-heads (vision↔vision), T-heads (text↔text causal), VT-heads (vision + causal text), all restricted to `same_doc`.
- Prefill-style benchmark uses `BlockMask` (`create_momh_block_mask_from_modality`) and `flex_attention_compiled` on GPU.

### 1.2 Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)
- PyTorch: upgraded to `torch==2.10.0+cu128` to support sm_120 (previous build did not support sm_120 and failed with “no kernel image is available”).

### 1.3 Benchmark shapes

- `B=1`, `H=15`, `T=2048`, `D=64`, dtype `fp16`
- `is_vision`: a few scattered “vision token” spans (simulating `<|image|>` placeholders)
- `attention_mask`: all-ones
- `document_ids` variants:
  - `None`: legacy behavior (no doc masking)
  - `zeros`: single-doc (non-packed)
  - `packed`: 4 equal-length docs (increased sparsity)

---

## 2. Results

Measured with CUDA event timing (per-call median, repeated loops; includes kernel time only).

### 2.1 `create_momh_block_mask_from_modality` (BlockMask build)

| document_ids | median (ms) | notes |
|-------------|-------------|------|
| None | ~0.850 | baseline |
| zeros | ~0.937 | +~0.09 ms vs baseline (same sparsity, extra constraint check) |
| packed | ~0.891 | slightly above baseline |

### 2.2 `flex_attention_compiled(q,k,v, block_mask=...)` (attention kernel)

| document_ids | median (ms) | notes |
|-------------|-------------|------|
| None | ~0.258 | baseline |
| zeros | ~0.272 | +~0.01 ms vs baseline (effectively negligible) |
| packed | ~0.093 | much faster due to higher sparsity from doc masking |

---

## 3. Analysis

- Adding `document_ids` is not “free”: it adds a small cost to BlockMask construction, and a very small cost to the attention call when sparsity is unchanged (single-doc).
- When packing is enabled, `document_ids` increases sparsity (blocks cross-document attention), which can more than repay the extra masking logic by reducing actual attention compute.
- Net effect in packed mode is typically a speedup for the attention kernel; overhead is dominated by BlockMask build, which is already relatively expensive regardless of doc masking.

---

## 4. Lessons learned → candidate skills

- Candidate skill: “MoMH packing: enforce document masking via `document_ids`”
- Candidate skill: “Blackwell (sm_120) PyTorch install: fix ‘no kernel image’ CUDA errors by upgrading to cu128 builds”

