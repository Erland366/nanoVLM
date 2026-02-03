#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.momh_attention import create_momh_block_mask_from_modality, flex_attention_compiled


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    max_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    timestamp: str
    torch_version: str
    cuda_version: str | None
    device: str
    device_name: str | None
    device_capability: tuple[int, int] | None
    arch_list: list[str] | None
    dtype: str
    batch_size: int
    heads: int
    seq_len: int
    head_dim: int
    pct_v: float
    pct_t: float
    num_docs: int
    blockmask_doc_none: TimingStats
    blockmask_doc_zeros: TimingStats
    blockmask_doc_packed: TimingStats
    attn_doc_none: TimingStats
    attn_doc_zeros: TimingStats
    attn_doc_packed: TimingStats
    first_call_blockmask_ms: float
    first_call_attn_ms: float


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark MoMH document masking overhead using FlexAttention.\n\n"
            "Measures:\n"
            "  (1) BlockMask construction time via create_momh_block_mask_from_modality\n"
            "  (2) Attention kernel time via flex_attention_compiled(q,k,v, block_mask=...)\n\n"
            "Compares 3 document_id modes:\n"
            "  - None: no document masking (legacy)\n"
            "  - zeros: single-document (non-packed)\n"
            "  - packed: multiple documents in one sequence (increased sparsity)\n"
        )
    )
    p.add_argument("--device", default="cuda", help="Device to run on (cuda recommended).")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--heads", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--pct-v", type=float, default=0.2, help="Fraction of heads for V->V.")
    p.add_argument("--pct-t", type=float, default=0.3, help="Fraction of heads for T->T.")
    p.add_argument("--num-docs", type=int, default=4, help="Number of packed documents for doc=packed.")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (excluded from timing).")
    p.add_argument("--iters", type=int, default=200, help="Iterations per timing repeat.")
    p.add_argument("--repeats", type=int, default=15, help="Number of repeats for timing stability.")
    p.add_argument(
        "--out-jsonl",
        default="benchmark_results/momh_doc_mask.jsonl",
        help="Path to append a JSONL benchmark record.",
    )
    p.add_argument("--no-write", action="store_true", help="Do not write JSONL output.")
    return p.parse_args(argv)


def _dtype_from_str(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _event_bench(fn, *, warmup: int, iters: int, repeats: int) -> TimingStats:
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()

    per_iter_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end)) / max(iters, 1)
        per_iter_ms.append(ms)

    return TimingStats(
        median_ms=float(statistics.median(per_iter_ms)),
        mean_ms=float(statistics.mean(per_iter_ms)),
        max_ms=float(max(per_iter_ms)),
    )


def _make_is_vision(*, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    # Scattered “vision tokens” to mimic image placeholder placement.
    is_vision = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
    spans = [(0, 64), (256, 320), (1024, 1088)]
    for lo, hi in spans:
        if lo >= seq_len:
            continue
        is_vision[:, lo : min(hi, seq_len)] = True
    return is_vision


def _make_document_ids(
    *, mode: str, batch_size: int, seq_len: int, num_docs: int, device: torch.device
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "zeros":
        return torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
    if mode == "packed":
        if num_docs <= 0:
            raise ValueError("--num-docs must be > 0 for packed mode.")
        doc = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
        seg = seq_len // num_docs
        if seg == 0:
            doc.fill_(0)
            return doc
        for i in range(num_docs):
            start = i * seg
            end = (i + 1) * seg
            doc[:, start:end] = i
        if seg * num_docs < seq_len:
            doc[:, seg * num_docs :] = num_docs - 1
        return doc
    raise ValueError(f"Unknown document_ids mode: {mode}")


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for CUDA. Use --device=cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dtype = _dtype_from_str(args.dtype)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = int(args.batch_size)
    H = int(args.heads)
    T = int(args.seq_len)
    D = int(args.head_dim)
    pct_v = float(args.pct_v)
    pct_t = float(args.pct_t)
    num_docs = int(args.num_docs)

    attention_mask = torch.ones((B, T), device=device, dtype=torch.bool)
    is_vision = _make_is_vision(batch_size=B, seq_len=T, device=device)

    q = torch.randn((B, H, T, D), device=device, dtype=dtype)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype)

    def build_blockmask(doc_ids: torch.Tensor | None):
        return create_momh_block_mask_from_modality(
            n_q_heads=H,
            q_len=T,
            kv_len=T,
            is_vision=is_vision,
            attention_mask=attention_mask,
            document_ids=doc_ids,
            pct_v=pct_v,
            pct_t=pct_t,
            device=str(device),
        )

    # Measure compilation / first-call overhead once (informational).
    doc_zeros = _make_document_ids(
        mode="zeros", batch_size=B, seq_len=T, num_docs=num_docs, device=device
    )
    t0 = time.perf_counter()
    bm_first = build_blockmask(doc_zeros)
    torch.cuda.synchronize()
    first_call_blockmask_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = flex_attention_compiled(q, k, v, block_mask=bm_first)
        torch.cuda.synchronize()
        first_call_attn_ms = (time.perf_counter() - t0) * 1000.0

    # Prepare document-id variants and blockmasks (built once for attention timing).
    doc_none = _make_document_ids(
        mode="none", batch_size=B, seq_len=T, num_docs=num_docs, device=device
    )
    doc_zeros = _make_document_ids(
        mode="zeros", batch_size=B, seq_len=T, num_docs=num_docs, device=device
    )
    doc_packed = _make_document_ids(
        mode="packed", batch_size=B, seq_len=T, num_docs=num_docs, device=device
    )

    # BlockMask build timing (per-call).
    blockmask_doc_none = _event_bench(
        lambda: build_blockmask(doc_none),
        warmup=int(args.warmup),
        iters=max(int(args.iters) // 4, 1),
        repeats=int(args.repeats),
    )
    blockmask_doc_zeros = _event_bench(
        lambda: build_blockmask(doc_zeros),
        warmup=int(args.warmup),
        iters=max(int(args.iters) // 4, 1),
        repeats=int(args.repeats),
    )
    blockmask_doc_packed = _event_bench(
        lambda: build_blockmask(doc_packed),
        warmup=int(args.warmup),
        iters=max(int(args.iters) // 4, 1),
        repeats=int(args.repeats),
    )

    bm_none = build_blockmask(doc_none)
    bm_zeros = build_blockmask(doc_zeros)
    bm_packed = build_blockmask(doc_packed)

    @torch.no_grad()
    def attn(bm):
        flex_attention_compiled(q, k, v, block_mask=bm)

    attn_doc_none = _event_bench(
        lambda: attn(bm_none),
        warmup=int(args.warmup),
        iters=int(args.iters),
        repeats=int(args.repeats),
    )
    attn_doc_zeros = _event_bench(
        lambda: attn(bm_zeros),
        warmup=int(args.warmup),
        iters=int(args.iters),
        repeats=int(args.repeats),
    )
    attn_doc_packed = _event_bench(
        lambda: attn(bm_packed),
        warmup=int(args.warmup),
        iters=int(args.iters),
        repeats=int(args.repeats),
    )

    result = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
        device=str(device),
        device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        device_capability=torch.cuda.get_device_capability(0) if device.type == "cuda" else None,
        arch_list=list(torch.cuda.get_arch_list()) if device.type == "cuda" else None,
        dtype=str(dtype).replace("torch.", ""),
        batch_size=B,
        heads=H,
        seq_len=T,
        head_dim=D,
        pct_v=pct_v,
        pct_t=pct_t,
        num_docs=num_docs,
        blockmask_doc_none=blockmask_doc_none,
        blockmask_doc_zeros=blockmask_doc_zeros,
        blockmask_doc_packed=blockmask_doc_packed,
        attn_doc_none=attn_doc_none,
        attn_doc_zeros=attn_doc_zeros,
        attn_doc_packed=attn_doc_packed,
        first_call_blockmask_ms=float(first_call_blockmask_ms),
        first_call_attn_ms=float(first_call_attn_ms),
    )

    payload = asdict(result)
    print(json.dumps(payload, indent=2))

    if not args.no_write:
        out_path = _ensure_parent(args.out_jsonl)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
