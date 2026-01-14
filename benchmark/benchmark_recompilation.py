"""
Benchmark script for MoMH flex_attention recompilation behavior.

Compares performance between:
- dynamic=False: Recompiles for each unique KV length (bad for decode)
- dynamic=True: Uses symbolic shapes, no recompilation (good for decode)

Usage:
    cd /home/coder/edd/nanoVLM_root/nanoVLM_momh
    source .venv/bin/activate

    # Run benchmark
    python benchmark/benchmark_recompilation.py

    # Run with recompilation logging
    TORCH_LOGS="recompiles" python benchmark/benchmark_recompilation.py
"""

import sys
import time
import argparse
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.momh_attention import (
    flex_attention_compiled,
    flex_attention_compiled_dynamic,
    generate_momh_score_mod_with_offset,
)
from models.config import VLMConfig


def benchmark_decode_recompilation(
    num_iterations: int = 50,
    initial_kv_len: int = 100,
    batch_size: int = 1,
    warmup: int = 3,
    verbose: bool = True
):
    """
    Benchmark decode phase with growing KV cache.

    Args:
        num_iterations: Number of decode steps to simulate
        initial_kv_len: Starting KV cache length (simulating prefill length)
        batch_size: Batch size
        warmup: Number of warmup iterations (not timed)
        verbose: Print detailed output

    Returns:
        dict with benchmark results
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This benchmark requires CUDA.")

    config = VLMConfig()
    n_heads = config.lm_n_heads
    head_dim = config.lm_hidden_dim // config.lm_n_heads
    S_V = config.mp_image_token_length

    results = {}

    # Create captured buffers for score_mod
    content_starts_buffer = torch.tensor([10] * batch_size, dtype=torch.int64, device="cuda")
    position_offset_buffer = torch.tensor(0, dtype=torch.int64, device="cuda")

    score_mod = generate_momh_score_mod_with_offset(
        n_heads, S_V, content_starts_buffer, position_offset_buffer,
        config.momh_head_pct_vision, config.momh_head_pct_text
    )

    kv_lengths = list(range(initial_kv_len, initial_kv_len + num_iterations))

    if verbose:
        print("=" * 70)
        print("MoMH Decode Recompilation Benchmark")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  n_heads: {n_heads}")
        print(f"  head_dim: {head_dim}")
        print(f"  batch_size: {batch_size}")
        print(f"  num_iterations: {num_iterations}")
        print(f"  initial_kv_len: {initial_kv_len}")
        print(f"  warmup iterations: {warmup}")
        print()

    # Benchmark dynamic=False (OLD behavior)
    if verbose:
        print("-" * 70)
        print("Test 1: dynamic=False (triggers recompilation per KV length)")
        print("-" * 70)

    torch._dynamo.reset()
    torch.cuda.synchronize()

    # Warmup
    for i, kv_len in enumerate(kv_lengths[:warmup]):
        position_offset_buffer.fill_(kv_len)
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = flex_attention_compiled(q, k, v, score_mod=score_mod)
    torch.cuda.synchronize()

    # Timed run (limited to avoid excessive time)
    test_iterations_static = min(num_iterations - warmup, 20)  # Cap at 20 to avoid long wait
    start = time.perf_counter()
    for i, kv_len in enumerate(kv_lengths[warmup:warmup + test_iterations_static]):
        position_offset_buffer.fill_(kv_len)
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = flex_attention_compiled(q, k, v, score_mod=score_mod)
    torch.cuda.synchronize()
    elapsed_static = time.perf_counter() - start

    results["dynamic_false"] = {
        "total_time": elapsed_static,
        "iterations": test_iterations_static,
        "time_per_iter": elapsed_static / test_iterations_static,
        "note": "Each unique KV length triggers recompilation"
    }

    if verbose:
        print(f"  Iterations: {test_iterations_static}")
        print(f"  Total time: {elapsed_static:.3f}s")
        print(f"  Time per iteration: {elapsed_static / test_iterations_static * 1000:.2f}ms")
        print()

    # Benchmark dynamic=True (NEW behavior)
    if verbose:
        print("-" * 70)
        print("Test 2: dynamic=True (no recompilation for different KV lengths)")
        print("-" * 70)

    torch._dynamo.reset()
    torch.cuda.synchronize()

    # Warmup
    for i, kv_len in enumerate(kv_lengths[:warmup]):
        position_offset_buffer.fill_(kv_len)
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = flex_attention_compiled_dynamic(q, k, v, score_mod=score_mod)
    torch.cuda.synchronize()

    # Timed run (full iterations)
    test_iterations_dynamic = num_iterations - warmup
    start = time.perf_counter()
    for i, kv_len in enumerate(kv_lengths[warmup:]):
        position_offset_buffer.fill_(kv_len)
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = flex_attention_compiled_dynamic(q, k, v, score_mod=score_mod)
    torch.cuda.synchronize()
    elapsed_dynamic = time.perf_counter() - start

    results["dynamic_true"] = {
        "total_time": elapsed_dynamic,
        "iterations": test_iterations_dynamic,
        "time_per_iter": elapsed_dynamic / test_iterations_dynamic,
        "note": "Symbolic shapes avoid recompilation"
    }

    if verbose:
        print(f"  Iterations: {test_iterations_dynamic}")
        print(f"  Total time: {elapsed_dynamic:.3f}s")
        print(f"  Time per iteration: {elapsed_dynamic / test_iterations_dynamic * 1000:.2f}ms")
        print()

    # Calculate speedup
    speedup = (elapsed_static / test_iterations_static) / (elapsed_dynamic / test_iterations_dynamic)
    results["speedup"] = speedup

    if verbose:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  dynamic=False: {elapsed_static / test_iterations_static * 1000:.2f}ms per iteration")
        print(f"  dynamic=True:  {elapsed_dynamic / test_iterations_dynamic * 1000:.2f}ms per iteration")
        print(f"  Speedup: {speedup:.1f}x faster with dynamic=True")
        print()
        print("Recommendation: Use dynamic=True for decode phase (growing KV cache)")
        print("               Use dynamic=False for prefill phase (fixed shapes)")

    return results


def benchmark_prefill_consistency(
    seq_len: int = 256,
    num_runs: int = 10,
    batch_size: int = 1,
    verbose: bool = True
):
    """
    Benchmark prefill with consistent shapes (no recompilation expected).

    Args:
        seq_len: Fixed sequence length
        num_runs: Number of runs
        batch_size: Batch size
        verbose: Print detailed output

    Returns:
        dict with benchmark results
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This benchmark requires CUDA.")

    from models.momh_attention import create_momh_block_mask

    config = VLMConfig()
    n_heads = config.lm_n_heads
    head_dim = config.lm_hidden_dim // config.lm_n_heads
    S_V = config.mp_image_token_length

    if verbose:
        print("=" * 70)
        print("MoMH Prefill Consistency Benchmark (Same Shape)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  seq_len: {seq_len}")
        print(f"  batch_size: {batch_size}")
        print(f"  num_runs: {num_runs}")
        print()

    torch._dynamo.reset()

    times = []
    for i in range(num_runs):
        content_starts = torch.tensor([10] * batch_size, device="cuda")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        block_mask = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = flex_attention_compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if verbose and i < 3:
            print(f"  Run {i+1}: {elapsed * 1000:.2f}ms")

    avg_time = sum(times[1:]) / len(times[1:])  # Exclude first run (compilation)
    first_run = times[0]

    if verbose:
        print()
        print(f"  First run (includes compile): {first_run * 1000:.2f}ms")
        print(f"  Average (runs 2-{num_runs}): {avg_time * 1000:.2f}ms")
        print(f"  Compile overhead: {(first_run - avg_time) * 1000:.2f}ms")

    return {
        "first_run": first_run,
        "avg_time": avg_time,
        "compile_overhead": first_run - avg_time,
        "all_times": times
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoMH flex_attention recompilation")
    parser.add_argument("--decode-iterations", type=int, default=50,
                        help="Number of decode iterations to benchmark")
    parser.add_argument("--initial-kv-len", type=int, default=100,
                        help="Initial KV cache length for decode benchmark")
    parser.add_argument("--prefill-seq-len", type=int, default=256,
                        help="Sequence length for prefill benchmark")
    parser.add_argument("--prefill-runs", type=int, default=10,
                        help="Number of prefill runs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip decode benchmark")
    parser.add_argument("--skip-prefill", action="store_true",
                        help="Skip prefill benchmark")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    print()
    print("MoMH Flex Attention Recompilation Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    if not args.skip_decode:
        benchmark_decode_recompilation(
            num_iterations=args.decode_iterations,
            initial_kv_len=args.initial_kv_len,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        print()

    if not args.skip_prefill:
        benchmark_prefill_consistency(
            seq_len=args.prefill_seq_len,
            num_runs=args.prefill_runs,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
