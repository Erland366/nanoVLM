#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from models.activation_checkpointing import get_default_sac_policy
from models.config import TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel

BASELINES_DIR = REPO_ROOT / "benchmarks" / "baselines" / "train_step_e2e"


@dataclass(frozen=True)
class GitInfo:
    repo_root: str
    commit: str
    branch: str
    dirty: bool


@dataclass(frozen=True)
class EnvInfo:
    hostname: str
    os: str
    python: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: str | None
    driver: str | None
    gpu_name: str | None
    gpu_count: int
    sm: str | None
    total_vram_mb: int | None


@dataclass(frozen=True)
class Sample:
    step_idx: int
    step_time_s: float
    tokens: int
    tokens_per_sec: float
    peak_vram_mb: float | None
    loss: float | None


@dataclass(frozen=True)
class BenchmarkRunV1:
    schema_version: int
    run_id: str
    benchmark_name: str
    baseline_name: str | None
    git: GitInfo
    env: EnvInfo
    config: dict[str, Any]
    metrics_summary: dict[str, Any]
    samples: list[Sample]
    notes: str


def _iso_utc_now() -> str:
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _get_git_info(*, repo_root: Path, allow_dirty: bool) -> GitInfo:
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root, text=True).strip()
    except Exception as exc:
        raise RuntimeError(f"Expected a git repository at {repo_root}") from exc

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
    if dirty and not allow_dirty:
        raise RuntimeError("Refusing to save baseline: git tree is dirty (pass --allow-dirty to override).")
    return GitInfo(repo_root=str(repo_root), commit=commit, branch=branch, dirty=dirty)


def _get_env_info(device: torch.device) -> EnvInfo:
    gpu_name = None
    gpu_count = 0
    sm = None
    total_vram_mb = None
    driver = None

    if device.type == "cuda" and torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = str(props.name)
        total_vram_mb = int(props.total_memory // (1024**2))
        cap = torch.cuda.get_device_capability(torch.cuda.current_device())
        sm = str(cap[0] * 10 + cap[1])
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0].strip()
        except Exception:
            driver = None

    return EnvInfo(
        hostname=socket.gethostname(),
        os=platform.platform(),
        python=sys.version.split()[0],
        torch_version=str(getattr(torch, "__version__", "")),
        cuda_version=str(getattr(torch.version, "cuda", None)) if getattr(torch.version, "cuda", None) else None,
        cudnn_version=str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        driver=driver,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        sm=sm,
        total_vram_mb=total_vram_mb,
    )


def _build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    env_vars = {}
    for key in ["TORCH_LOGS", "CUDA_VISIBLE_DEVICES"]:
        if key in os.environ:
            env_vars[key] = os.environ[key]
    return {
        "cmd": list(os.sys.argv),
        "env_vars": env_vars,
        "args": vars(args),
        "torch_state": {
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "deterministic_algorithms": (
                bool(torch.are_deterministic_algorithms_enabled())
                if hasattr(torch, "are_deterministic_algorithms_enabled")
                else None
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }


def _validate_benchmark_run_v1(run: dict[str, Any]) -> None:
    if int(run.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported schema_version={run.get('schema_version')!r}; expected 1.")
    for key in ["run_id", "benchmark_name", "git", "env", "config", "metrics_summary", "samples"]:
        if key not in run:
            raise ValueError(f"Missing required key: {key}")
    metrics = run["metrics_summary"]
    for key in ["step_time_mean_s", "tokens_per_sec_mean", "peak_vram_mb"]:
        if key not in metrics:
            raise ValueError(f"metrics_summary missing required key: {key}")


def compare_benchmark_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    throughput_threshold_pct: float = 5.0,
    vram_threshold_pct: float = 5.0,
) -> dict[str, Any]:
    _validate_benchmark_run_v1(baseline)
    _validate_benchmark_run_v1(current)
    if baseline.get("benchmark_name") != current.get("benchmark_name"):
        raise ValueError(
            f"benchmark_name mismatch: baseline={baseline.get('benchmark_name')!r} current={current.get('benchmark_name')!r}"
        )

    b = baseline["metrics_summary"]
    c = current["metrics_summary"]

    def _pct_delta(cur: float, base: float) -> float:
        if base == 0:
            return float("inf") if cur != 0 else 0.0
        return (cur - base) / base * 100.0

    tokens_base = float(b["tokens_per_sec_mean"])
    tokens_cur = float(c["tokens_per_sec_mean"])
    vram_base = float(b["peak_vram_mb"]) if b["peak_vram_mb"] is not None else None
    vram_cur = float(c["peak_vram_mb"]) if c["peak_vram_mb"] is not None else None

    deltas: dict[str, Any] = {
        "tokens_per_sec_mean": {
            "baseline": tokens_base,
            "current": tokens_cur,
            "delta": tokens_cur - tokens_base,
            "pct_delta": _pct_delta(tokens_cur, tokens_base),
        },
        "peak_vram_mb": {
            "baseline": vram_base,
            "current": vram_cur,
            "delta": (vram_cur - vram_base) if (vram_base is not None and vram_cur is not None) else None,
            "pct_delta": _pct_delta(vram_cur, vram_base) if (vram_base is not None and vram_cur is not None) else None,
        },
    }

    verdict = "pass"
    reasons: list[str] = []

    if deltas["tokens_per_sec_mean"]["pct_delta"] < -float(throughput_threshold_pct):
        verdict = "regression"
        reasons.append(
            f"throughput regression: {deltas['tokens_per_sec_mean']['pct_delta']:.2f}% < -{throughput_threshold_pct:.2f}%"
        )

    if (
        deltas["peak_vram_mb"]["pct_delta"] is not None
        and deltas["peak_vram_mb"]["pct_delta"] > float(vram_threshold_pct)
    ):
        verdict = "regression"
        reasons.append(
            f"VRAM regression: {deltas['peak_vram_mb']['pct_delta']:.2f}% > {vram_threshold_pct:.2f}%"
        )

    return {
        "verdict": verdict,
        "reasons": reasons,
        "deltas": deltas,
    }


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    momh_enabled: bool
    compile: bool
    compile_mode: str | None
    activation_checkpointing: bool
    activation_checkpointing_mode: str
    activation_checkpointing_policy: str | None
    activation_memory_budget: float | None
    device: str
    dtype: str
    seed: int
    matmul_precision: str
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    compile_time_ms: float | None
    batch_size: int
    seq_len: int
    num_images: int
    tiles_per_image: int
    warmup_steps: int
    steps: int
    tokens_per_step: int
    step_time_ms_mean: float
    tokens_per_second: float
    base_vram_mb: float | None
    peak_vram_mb: float | None
    training_vram_mb: float | None


class _DummyTokenizer:
    def __init__(self, *, image_token_id: int, pad_token_id: int, eos_token_id: int):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Unsloth-style training-step benchmark for this repo.\n\n"
            "Measures step time, tokens/s, and VRAM (base/peak/training delta) for a short "
            "forward+backward+optimizer loop.\n\n"
            "By default uses synthetic batches to isolate model performance."
        )
    )
    p.add_argument("--mode", choices=["synthetic", "hf"], default="synthetic")

    p.add_argument("--momh", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable regional torch.compile (vision blocks/decoder/MP) for the benchmark.",
    )
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode to use when --compile is enabled.",
    )
    p.add_argument(
        "--activation-memory-budget",
        type=float,
        default=None,
        help="torch.compile activation memory budget (0-1). Requires --compile.",
    )
    p.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable activation checkpointing. When --compile is enabled, "
            "this uses selective activation checkpointing; otherwise it uses manual "
            "checkpointing."
        ),
    )

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-images", type=int, default=1, help="Number of images per sample (synthetic mode).")
    p.add_argument("--tiles-per-image", type=int, default=1, help="Number of ViT-sized tiles per image (synthetic mode).")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")

    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument(
        "--vary-batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes for a shape-sweep run (e.g., '4,3,4').",
    )
    p.add_argument(
        "--vary-seq-lens",
        type=str,
        default=None,
        help="Comma-separated seq lengths for a shape-sweep run (e.g., '2048,1536,2048').",
    )
    p.add_argument(
        "--shape-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps per shape in shape-sweep mode (default: 0).",
    )
    p.add_argument(
        "--shape-steps",
        type=int,
        default=1,
        help="Measured steps per shape in shape-sweep mode (default: 1).",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Autocast dtype for CUDA runs.",
    )
    p.add_argument("--device", default="cuda", help="Device string (default: cuda).")

    # HF mode (optional)
    p.add_argument("--dataset", default="patrickamadeus/the_cauldron")
    p.add_argument("--config", default="sample_1pct")
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--search-limit", type=int, default=200)
    p.add_argument("--min-raw-images", type=int, default=1)
    p.add_argument("--min-total-tiles", type=int, default=1)

    p.add_argument("--out-jsonl", default="benchmark_results/train_step.jsonl")
    p.add_argument("--out-json", default=None, help="Write a single BenchmarkRun JSON to this path.")
    p.add_argument("--save-baseline", default=None, help="Save a committed baseline JSON under benchmarks/baselines/train_step_e2e/.")
    p.add_argument("--allow-dirty", action="store_true", help="Allow saving baseline when git tree is dirty.")
    p.add_argument("--compare", action="store_true", help="Compare two BenchmarkRun JSON files and exit.")
    p.add_argument("--baseline", default=None, help="Baseline path or baseline name under benchmarks/baselines/train_step_e2e/.")
    p.add_argument("--current", default=None, help="Current BenchmarkRun JSON path.")
    p.add_argument("--throughput-threshold-pct", type=float, default=5.0)
    p.add_argument("--vram-threshold-pct", type=float, default=5.0)
    return p.parse_args(argv)


def _autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "bf16":
        return torch.bfloat16
    return torch.float16


def _compile_module_list(modules, *, dynamic: bool | None = None, mode: str | None = "reduce-overhead") -> None:
    for idx, block in enumerate(modules):
        modules[idx] = torch.compile(block, dynamic=dynamic, mode=mode)


def _compile_regions(
    model: torch.nn.Module, *, dynamic: bool | None = None, mode: str | None = "reduce-overhead"
) -> None:
    if not hasattr(model, "vision_encoder") or not hasattr(model, "decoder") or not hasattr(model, "MP"):
        raise AttributeError("Model must expose vision_encoder, decoder, and MP for regional compile.")
    if hasattr(model.vision_encoder, "blocks"):
        _compile_module_list(model.vision_encoder.blocks, dynamic=dynamic, mode=mode)
    else:
        model.vision_encoder = torch.compile(model.vision_encoder, dynamic=dynamic, mode=mode)
    if hasattr(model.decoder, "blocks"):
        _compile_module_list(model.decoder.blocks, dynamic=dynamic, mode=mode)
    else:
        model.decoder = torch.compile(model.decoder, dynamic=dynamic, mode=mode)
    model.MP = torch.compile(model.MP, dynamic=dynamic, mode=mode)


def _make_synthetic_batch(
    *,
    cfg: VLMConfig,
    tokenizer: _DummyTokenizer,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    num_images: int,
    tiles_per_image: int,
) -> dict[str, Any]:
    total_tiles_per_sample = num_images * tiles_per_image
    required_placeholders = total_tiles_per_sample * int(cfg.mp_image_token_length)
    if required_placeholders >= seq_len:
        raise ValueError(
            f"seq_len={seq_len} too small for placeholders={required_placeholders} "
            f"(num_images={num_images}, tiles_per_image={tiles_per_image}, mp_image_token_length={cfg.mp_image_token_length})"
        )

    # Avoid sampling `image_token_id` outside intended placeholder positions.
    input_ids = torch.randint(0, cfg.lm_vocab_size - 1, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    # Place image placeholders at the start of the content.
    input_ids[:, :required_placeholders] = tokenizer.image_token_id

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[input_ids == tokenizer.image_token_id] = -100

    H = int(cfg.vit_img_size)
    # images format: list[batch] of list[num_images] of tensor[tiles_per_image, 3, H, W]
    images = []
    for _ in range(batch_size):
        per_sample: list[torch.Tensor] = []
        for _ in range(num_images):
            per_sample.append(torch.randn((tiles_per_image, 3, H, H), device=device))
        images.append(per_sample)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images,
        "tokens_per_step": int(attention_mask.sum().item()),
    }


def _parse_int_list(value: str | None, *, name: str) -> list[int] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    try:
        return [int(part) for part in items]
    except ValueError as exc:
        raise ValueError(f"{name} must contain only integers") from exc


def _build_shape_variants(args: argparse.Namespace) -> list[tuple[int, int]] | None:
    batch_sizes = _parse_int_list(args.vary_batch_sizes, name="--vary-batch-sizes")
    seq_lens = _parse_int_list(args.vary_seq_lens, name="--vary-seq-lens")
    if batch_sizes is None and seq_lens is None:
        return None
    if batch_sizes is None:
        batch_sizes = [int(args.batch_size)]
    if seq_lens is None:
        seq_lens = [int(args.seq_len)]
    if len(batch_sizes) == len(seq_lens):
        return list(zip(batch_sizes, seq_lens))
    if len(batch_sizes) == 1:
        return [(batch_sizes[0], seq_len) for seq_len in seq_lens]
    if len(seq_lens) == 1:
        return [(batch_size, seq_lens[0]) for batch_size in batch_sizes]
    raise ValueError(
        "Shape sweep expects matching lengths for --vary-batch-sizes and --vary-seq-lens, "
        "or one of them must be length 1."
    )


def _find_hf_batch(
    args: argparse.Namespace,
    cfg: VLMConfig,
    device: torch.device,
    *,
    batch_size: int | None = None,
    seq_len: int | None = None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("HF mode requires `datasets` installed and an active environment.") from e

    from data.collators import VQACollator
    from data.datasets import VQAIterableDataset
    from data.processors import get_image_processor, get_tokenizer

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, cfg.resize_to_max_side_len)
    vqa_iter = VQAIterableDataset(ds, tokenizer, image_processor, cfg.mp_image_token_length)
    seq_len = int(seq_len if seq_len is not None else args.seq_len)
    batch_size = int(batch_size if batch_size is not None else args.batch_size)
    collator = VQACollator(tokenizer, max_length=seq_len)

    candidates: list[dict[str, Any]] = []
    for idx, sample in enumerate(vqa_iter):
        if idx >= args.search_limit:
            break
        if sample is None:
            continue
        raw_images = len(sample["images"])
        total_tiles = sum(int(t.shape[0]) for t in sample["images"])
        if raw_images < args.min_raw_images or total_tiles < args.min_total_tiles:
            continue
        if len(sample["input_ids"]) > seq_len:
            continue

        candidates.append(sample)
        if len(candidates) < batch_size:
            continue

        batch = collator(candidates)
        # Collator returns lists when it drops all samples (e.g. all too long).
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            # Keep searching; drop the oldest candidate to make room.
            candidates = candidates[1:]
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Ensure we do not train on image placeholders.
        labels[input_ids == tokenizer.image_token_id] = -100

        tokens_per_step = int(attention_mask.sum().item())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": batch["images"],
            "tokens_per_step": tokens_per_step,
        }

    if not candidates:
        raise RuntimeError(
            f"No suitable sample found within search-limit={args.search_limit} "
            f"(min_raw_images={args.min_raw_images}, min_total_tiles={args.min_total_tiles}, seq_len={seq_len})."
        )
    raise RuntimeError(
        f"Unable to build a valid batch of size {batch_size} within search-limit={args.search_limit} "
        f"(min_raw_images={args.min_raw_images}, min_total_tiles={args.min_total_tiles}, seq_len={seq_len}). "
        "Increase --search-limit, increase --seq-len, or relax constraints."
    )


def _run_train_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    images: Any,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    tokens_per_step: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup_steps: int,
    steps: int,
    measure_compile_time: bool,
    cudagraph_mark_step: bool,
) -> tuple[float, float, float | None, float | None, float | None, float | None, list[Sample], float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base_vram_bytes = torch.cuda.memory_allocated(device)
    else:
        base_vram_bytes = None

    def _train_step(*, measure_peak_vram: bool) -> tuple[float, float, float | None]:
        if cudagraph_mark_step and hasattr(torch, "compiler"):
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            if measure_peak_vram:
                torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        else:
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        if loss is None:
            raise RuntimeError("Model returned loss=None; cannot benchmark training step.")
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss_value = float(loss.detach().float().item())
        peak_vram_mb = None
        if device.type == "cuda" and measure_peak_vram:
            peak_vram_mb = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
        return (t1 - t0), loss_value, peak_vram_mb

    compile_time_ms = None
    if measure_compile_time:
        compile_time_ms = float(_train_step(measure_peak_vram=False)[0] * 1000.0)

    for _ in range(warmup_steps):
        _train_step(measure_peak_vram=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        base_vram_bytes = torch.cuda.memory_allocated(device)

    step_times: list[float] = []
    samples: list[Sample] = []
    losses: list[float] = []
    start = time.perf_counter()
    for step_idx in range(steps):
        dt, loss_value, peak_step_vram_mb = _train_step(measure_peak_vram=True)
        step_times.append(dt)
        losses.append(loss_value)
        samples.append(
            Sample(
                step_idx=step_idx,
                step_time_s=float(dt),
                tokens=int(tokens_per_step),
                tokens_per_sec=float(tokens_per_step) / float(dt) if dt > 0 else 0.0,
                peak_vram_mb=peak_step_vram_mb,
                loss=loss_value,
            )
        )
    end = time.perf_counter()

    wall = end - start
    step_time_mean = sum(step_times) / len(step_times)
    tokens_per_second = (tokens_per_step * steps) / wall

    if device.type == "cuda":
        base_vram_mb = float(base_vram_bytes) / (1024**2) if base_vram_bytes is not None else None
        peak_candidates = [s.peak_vram_mb for s in samples if s.peak_vram_mb is not None]
        peak_vram_mb = float(max(peak_candidates)) if peak_candidates else None
        training_vram_mb = (
            float(peak_vram_mb - base_vram_mb) if (peak_vram_mb is not None and base_vram_mb is not None)
            else None
        )
    else:
        base_vram_mb = peak_vram_mb = training_vram_mb = None

    loss_mean = float(sum(losses) / len(losses)) if losses else None
    return step_time_mean, tokens_per_second, base_vram_mb, peak_vram_mb, training_vram_mb, compile_time_ms, samples, loss_mean


def _resolve_baseline_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if value.endswith(".json") or "/" in value or "\\" in value:
        return candidate
    safe = value.strip()
    if not safe or any(ch in safe for ch in " \t\n\r"):
        raise ValueError("Baseline name must be a non-empty token without whitespace.")
    return BASELINES_DIR / f"{safe}.json"


def _maybe_mark_dynamic(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    images: Any,
    compile_enabled: bool,
) -> None:
    if not compile_enabled:
        return
    torch._dynamo.maybe_mark_dynamic(input_ids, 0)
    torch._dynamo.maybe_mark_dynamic(input_ids, 1)
    torch._dynamo.maybe_mark_dynamic(labels, 0)
    torch._dynamo.maybe_mark_dynamic(labels, 1)
    torch._dynamo.maybe_mark_dynamic(attention_mask, 0)
    torch._dynamo.maybe_mark_dynamic(attention_mask, 1)
    if isinstance(images, torch.Tensor):
        torch._dynamo.maybe_mark_dynamic(images, 0)


def _ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.compare:
        if not args.baseline or not args.current:
            raise ValueError("--compare requires --baseline and --current.")
        baseline_path = _resolve_baseline_path(str(args.baseline))
        current_path = Path(str(args.current))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = json.loads(current_path.read_text(encoding="utf-8"))
        result = compare_benchmark_runs(
            baseline,
            current,
            throughput_threshold_pct=float(args.throughput_threshold_pct),
            vram_threshold_pct=float(args.vram_threshold_pct),
        )
        baseline_commit = baseline.get("git", {}).get("commit", "unknown")
        current_commit = current.get("git", {}).get("commit", "unknown")
        print(f"baseline: {baseline_path} (commit {baseline_commit})")
        print(f"current:  {current_path} (commit {current_commit})")
        print("")
        deltas = result["deltas"]
        t = deltas["tokens_per_sec_mean"]
        v = deltas["peak_vram_mb"]
        print("metric              baseline        current         delta        %delta")
        print(f"tokens_per_sec_mean  {t['baseline']:<13.3f} {t['current']:<13.3f} {t['delta']:<11.3f} {t['pct_delta']:<8.2f}")
        if v["baseline"] is not None and v["current"] is not None:
            print(f"peak_vram_mb         {v['baseline']:<13.3f} {v['current']:<13.3f} {v['delta']:<11.3f} {v['pct_delta']:<8.2f}")
        else:
            print("peak_vram_mb         n/a            n/a            n/a         n/a")
        print("")
        print(f"verdict: {result['verdict']}")
        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"- {reason}")
        return 1 if result["verdict"] == "regression" else 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")

    train_cfg = TrainConfig()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    compile_enabled = bool(args.compile or train_cfg.compile)
    compile_mode = args.compile_mode if compile_enabled else None
    activation_memory_budget = args.activation_memory_budget
    if activation_memory_budget is not None:
        if not 0.0 <= activation_memory_budget <= 1.0:
            raise ValueError("--activation-memory-budget must be between 0 and 1.")
        if not hasattr(torch._dynamo.config, "activation_memory_budget"):
            raise RuntimeError("activation_memory_budget is not supported in this PyTorch build.")
        if compile_enabled:
            torch._dynamo.config.activation_memory_budget = activation_memory_budget

    cfg = VLMConfig()
    cfg.momh_enabled = bool(args.momh)
    activation_checkpointing = bool(args.activation_checkpointing)
    use_selective_ac = bool(compile_enabled and activation_checkpointing)
    activation_checkpointing_mode = (
        "off"
        if not activation_checkpointing
        else ("selective" if use_selective_ac else "manual")
    )
    activation_checkpointing_policy = get_default_sac_policy() if use_selective_ac else None

    cfg.activation_checkpointing = activation_checkpointing

    # Synthetic mode uses a dummy tokenizer to avoid HF tokenizer overhead and to ensure a stable image_token_id.
    tokenizer = None
    if args.mode == "synthetic":
        tokenizer = _DummyTokenizer(
            image_token_id=cfg.lm_vocab_size - 1,
            pad_token_id=0,
            eos_token_id=1,
        )

    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer)
    if hasattr(model, "set_activation_checkpointing_mode"):
        model.set_activation_checkpointing_mode(
            use_selective=use_selective_ac,
            allow_cache_entry_mutation=use_selective_ac,
            policy=activation_checkpointing_policy,
        )
        if use_selective_ac:
            print("Using selective activation checkpointing under torch.compile (allow_cache_entry_mutation=True).")
    if compile_enabled:
        _compile_regions(model, dynamic=None, mode=compile_mode)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype = _autocast_dtype(args)
    variants = _build_shape_variants(args)
    if variants is not None and (args.out_json or args.save_baseline):
        raise ValueError("--out-json/--save-baseline are only supported for single-run mode (no shape-sweep).")

    if variants is None:
        if args.mode == "synthetic":
            batch = _make_synthetic_batch(
                cfg=cfg,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                device=device,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_images=args.num_images,
                tiles_per_image=args.tiles_per_image,
            )
        else:
            batch = _find_hf_batch(args, cfg, device)

        input_ids = batch["input_ids"]
        images = batch["images"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        tokens_per_step = int(batch["tokens_per_step"])
        _maybe_mark_dynamic(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images,
            compile_enabled=compile_enabled,
        )

        (
            step_time_mean,
            tokens_per_second,
            base_vram_mb,
            peak_vram_mb,
            training_vram_mb,
            compile_time_ms,
            samples,
            loss_mean,
        ) = _run_train_steps(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            labels=labels,
            tokens_per_step=tokens_per_step,
            device=device,
            amp_dtype=amp_dtype,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            measure_compile_time=compile_enabled,
            cudagraph_mark_step=compile_enabled,
        )

        result = BenchmarkResult(
            mode=args.mode,
            momh_enabled=bool(args.momh),
            compile=compile_enabled,
            compile_mode=compile_mode,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_mode=activation_checkpointing_mode,
            activation_checkpointing_policy=activation_checkpointing_policy,
            activation_memory_budget=activation_memory_budget,
            device=str(device),
            dtype=args.dtype,
            seed=int(args.seed),
            matmul_precision=str(torch.get_float32_matmul_precision()),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            compile_time_ms=compile_time_ms,
            batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            num_images=int(args.num_images),
            tiles_per_image=int(args.tiles_per_image),
            warmup_steps=int(args.warmup_steps),
            steps=int(args.steps),
            tokens_per_step=int(tokens_per_step),
            step_time_ms_mean=float(step_time_mean * 1000.0),
            tokens_per_second=float(tokens_per_second),
            base_vram_mb=base_vram_mb,
            peak_vram_mb=peak_vram_mb,
            training_vram_mb=training_vram_mb,
        )

        out_path = _ensure_parent(args.out_jsonl)
        record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **asdict(result)}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        git_info = _get_git_info(
            repo_root=REPO_ROOT,
            allow_dirty=bool(args.allow_dirty) if args.save_baseline else True,
        )
        env_info = _get_env_info(device)
        run = BenchmarkRunV1(
            schema_version=1,
            run_id=f"{_iso_utc_now()}__train_step_e2e",
            benchmark_name="train_step_e2e",
            baseline_name=str(args.save_baseline) if args.save_baseline else None,
            git=git_info,
            env=env_info,
            config=_build_run_config(args),
            metrics_summary={
                "step_time_mean_s": float(step_time_mean),
                "tokens_per_sec_mean": float(tokens_per_second),
                "peak_vram_mb": peak_vram_mb,
                "loss_mean": loss_mean,
            },
            samples=samples,
            notes="",
        )
        run_dict = asdict(run)
        _validate_benchmark_run_v1(run_dict)

        if args.out_json:
            out_json_path = _ensure_parent(str(args.out_json))
            out_json_path.write_text(json.dumps(run_dict, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote: {out_json_path}")

        if args.save_baseline:
            BASELINES_DIR.mkdir(parents=True, exist_ok=True)
            baseline_path = _resolve_baseline_path(str(args.save_baseline))
            baseline_path.write_text(json.dumps(run_dict, indent=2) + "\n", encoding="utf-8")
            print(f"Saved baseline: {baseline_path}")

        print(json.dumps(record, indent=2))
        print(f"Wrote: {out_path}")
        return 0

    # Shape-sweep mode: run multiple shapes in a single process to surface recompiles.
    out_path = _ensure_parent(args.out_jsonl)
    for batch_size, seq_len in variants:
        if args.mode == "synthetic":
            batch = _make_synthetic_batch(
                cfg=cfg,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                num_images=args.num_images,
                tiles_per_image=args.tiles_per_image,
            )
        else:
            batch = _find_hf_batch(args, cfg, device, batch_size=batch_size, seq_len=seq_len)

        input_ids = batch["input_ids"]
        images = batch["images"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        tokens_per_step = int(batch["tokens_per_step"])
        _maybe_mark_dynamic(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images,
            compile_enabled=compile_enabled,
        )

        (
            step_time_mean,
            tokens_per_second,
            base_vram_mb,
            peak_vram_mb,
            training_vram_mb,
            compile_time_ms,
            _samples,
            _loss_mean,
        ) = _run_train_steps(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            labels=labels,
            tokens_per_step=tokens_per_step,
            device=device,
            amp_dtype=amp_dtype,
            warmup_steps=args.shape_warmup_steps,
            steps=args.shape_steps,
            measure_compile_time=compile_enabled,
            cudagraph_mark_step=compile_enabled,
        )

        result = BenchmarkResult(
            mode=args.mode,
            momh_enabled=bool(args.momh),
            compile=compile_enabled,
            compile_mode=compile_mode,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_mode=activation_checkpointing_mode,
            activation_checkpointing_policy=activation_checkpointing_policy,
            activation_memory_budget=activation_memory_budget,
            device=str(device),
            dtype=args.dtype,
            seed=int(args.seed),
            matmul_precision=str(torch.get_float32_matmul_precision()),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            compile_time_ms=compile_time_ms,
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            num_images=int(args.num_images),
            tiles_per_image=int(args.tiles_per_image),
            warmup_steps=int(args.shape_warmup_steps),
            steps=int(args.shape_steps),
            tokens_per_step=int(tokens_per_step),
            step_time_ms_mean=float(step_time_mean * 1000.0),
            tokens_per_second=float(tokens_per_second),
            base_vram_mb=base_vram_mb,
            peak_vram_mb=peak_vram_mb,
            training_vram_mb=training_vram_mb,
        )

        record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **asdict(result)}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(json.dumps(record, indent=2))
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
