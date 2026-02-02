"""
Benchmark token efficiency sync methods.

Run with: torchrun --nproc_per_node=2 benchmark/token_sync_benchmark.py

Tests three approaches:
1. Local-only: Just sum local lists (no communication)
2. Sync all-reduce: Blocking all-reduce
3. Async all-reduce: Non-blocking all-reduce with wait
"""

import os
import time
import torch
import torch.distributed as dist
from statistics import mean, stdev


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def benchmark_local_only(accumulated_effective: list, accumulated_total: list, n_iters: int = 1000):
    """Benchmark local-only summing (no communication)."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()

        local_effective = sum(accumulated_effective)
        local_total = sum(accumulated_total)
        efficiency = local_effective / local_total if local_total > 0 else 1.0

        times.append(time.perf_counter() - start)

    return times, local_effective, local_total, efficiency


def benchmark_sync_allreduce(accumulated_effective: list, accumulated_total: list, device, n_iters: int = 1000):
    """Benchmark synchronous all-reduce."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()

        local_effective = sum(accumulated_effective)
        local_total = sum(accumulated_total)
        token_tensor = torch.tensor([local_effective, local_total],
                                    device=device,
                                    dtype=torch.float64)
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, async_op=False)
        global_effective = token_tensor[0].item()
        global_total = token_tensor[1].item()
        efficiency = global_effective / global_total if global_total > 0 else 1.0

        times.append(time.perf_counter() - start)

    return times, global_effective, global_total, efficiency


def benchmark_async_allreduce(accumulated_effective: list, accumulated_total: list, device, n_iters: int = 1000):
    """Benchmark async all-reduce with immediate wait."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()

        local_effective = sum(accumulated_effective)
        local_total = sum(accumulated_total)
        token_tensor = torch.tensor([local_effective, local_total],
                                    device=device,
                                    dtype=torch.float64)
        work = dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        global_effective = token_tensor[0].item()
        global_total = token_tensor[1].item()
        efficiency = global_effective / global_total if global_total > 0 else 1.0

        times.append(time.perf_counter() - start)

    return times, global_effective, global_total, efficiency


def benchmark_async_with_work(accumulated_effective: list, accumulated_total: list, device, n_iters: int = 1000):
    """
    Benchmark async all-reduce with simulated work between start and wait.
    This simulates the real scenario where we do other stats processing.
    """
    times = []
    overlap_times = []  # Time for work that overlaps with communication

    for _ in range(n_iters):
        start = time.perf_counter()

        # Start async all-reduce
        local_effective = sum(accumulated_effective)
        local_total = sum(accumulated_total)
        token_tensor = torch.tensor([local_effective, local_total],
                                    device=device,
                                    dtype=torch.float64)
        work = dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, async_op=True)

        # Simulate work that happens while communication is in flight
        # This mimics the existing stats processing in train.py
        work_start = time.perf_counter()
        dummy_stats = {}
        for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time']:
            dummy_list = [1.0] * 100  # Simulate 100 accumulated values
            dummy_stats[f'avg_{key}'] = mean(dummy_list)
            dummy_stats[f'max_{key}'] = max(dummy_list)
        overlap_times.append(time.perf_counter() - work_start)

        # Wait for communication
        work.wait()
        global_effective = token_tensor[0].item()
        global_total = token_tensor[1].item()
        efficiency = global_effective / global_total if global_total > 0 else 1.0

        times.append(time.perf_counter() - start)

    return times, overlap_times, global_effective, global_total, efficiency


def main():
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Simulate accumulated token stats (100 batches worth, as in stats_log_interval=100)
    # Each batch might have different effective tokens due to padding
    n_batches = 100
    batch_size = 16
    seq_len = 2048

    # Simulate varying padding ratios per batch (60-95% efficiency)
    import random
    random.seed(42 + rank)  # Different data per rank

    accumulated_effective = []
    accumulated_total = []
    for _ in range(n_batches):
        total = batch_size * seq_len
        efficiency_ratio = random.uniform(0.6, 0.95)
        effective = int(total * efficiency_ratio)
        accumulated_effective.append(effective)
        accumulated_total.append(total)

    n_iters = 1000
    warmup = 100

    if rank == 0:
        print(f"=" * 60)
        print(f"Token Sync Benchmark")
        print(f"=" * 60)
        print(f"World size: {world_size}")
        print(f"Simulated batches per log interval: {n_batches}")
        print(f"Batch size: {batch_size}, Seq len: {seq_len}")
        print(f"Iterations: {n_iters} (warmup: {warmup})")
        print(f"=" * 60)

    dist.barrier()

    # Warmup
    for _ in range(warmup):
        benchmark_sync_allreduce(accumulated_effective, accumulated_total, device, n_iters=1)

    dist.barrier()

    # Benchmark 1: Local only
    times_local, eff_local, tot_local, ratio_local = benchmark_local_only(
        accumulated_effective, accumulated_total, n_iters)

    dist.barrier()

    # Benchmark 2: Sync all-reduce
    times_sync, eff_sync, tot_sync, ratio_sync = benchmark_sync_allreduce(
        accumulated_effective, accumulated_total, device, n_iters)

    dist.barrier()

    # Benchmark 3: Async all-reduce (immediate wait)
    times_async, eff_async, tot_async, ratio_async = benchmark_async_allreduce(
        accumulated_effective, accumulated_total, device, n_iters)

    dist.barrier()

    # Benchmark 4: Async all-reduce with simulated work
    times_async_work, overlap_times, eff_async_work, tot_async_work, ratio_async_work = benchmark_async_with_work(
        accumulated_effective, accumulated_total, device, n_iters)

    dist.barrier()

    if rank == 0:
        print(f"\n{'Method':<30} {'Mean (μs)':<12} {'Std (μs)':<12} {'Min (μs)':<12} {'Max (μs)':<12}")
        print("-" * 78)

        def print_stats(name, times):
            times_us = [t * 1e6 for t in times]
            print(f"{name:<30} {mean(times_us):<12.2f} {stdev(times_us):<12.2f} {min(times_us):<12.2f} {max(times_us):<12.2f}")

        print_stats("Local only (no comm)", times_local)
        print_stats("Sync all-reduce", times_sync)
        print_stats("Async all-reduce (immed wait)", times_async)
        print_stats("Async + work overlap", times_async_work)

        print(f"\n{'Overlap work time (μs)':<30} {mean([t*1e6 for t in overlap_times]):<12.2f}")

        print(f"\n" + "=" * 60)
        print("Results verification:")
        print(f"  Local (rank 0):  effective={eff_local:,}, total={tot_local:,}, ratio={ratio_local:.4f}")
        print(f"  Sync global:     effective={eff_sync:,}, total={tot_sync:,}, ratio={ratio_sync:.4f}")
        print(f"  Async global:    effective={eff_async:,}, total={tot_async:,}, ratio={ratio_async:.4f}")

        print(f"\n" + "=" * 60)
        print("Overhead analysis:")
        local_mean = mean(times_local) * 1e6
        sync_mean = mean(times_sync) * 1e6
        async_mean = mean(times_async) * 1e6
        async_work_mean = mean(times_async_work) * 1e6
        overlap_mean = mean(overlap_times) * 1e6

        print(f"  Sync overhead vs local:       +{sync_mean - local_mean:.2f} μs")
        print(f"  Async (immed) overhead:       +{async_mean - local_mean:.2f} μs")
        print(f"  Async + work overhead:        +{async_work_mean - local_mean:.2f} μs")
        print(f"  Effective overlap savings:    {overlap_mean:.2f} μs (work done during comm)")

        if async_work_mean < sync_mean:
            print(f"\n  ✓ Async with work overlap is faster than sync by {sync_mean - async_work_mean:.2f} μs")
        else:
            print(f"\n  ✗ Sync is faster than async with work by {async_work_mean - sync_mean:.2f} μs")

    cleanup()


if __name__ == "__main__":
    main()
