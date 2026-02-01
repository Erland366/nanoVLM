#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from models.config import TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    momh_enabled: bool
    compile: bool
    device: str
    dtype: str
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

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-images", type=int, default=1, help="Number of images per sample (synthetic mode).")
    p.add_argument("--tiles-per-image", type=int, default=1, help="Number of ViT-sized tiles per image (synthetic mode).")

    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
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
) -> tuple[float, float, float | None, float | None, float | None, float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base_vram_bytes = torch.cuda.memory_allocated(device)
    else:
        base_vram_bytes = None

    def _train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
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
        return t1 - t0

    compile_time_ms = None
    if measure_compile_time:
        compile_time_ms = float(_train_step() * 1000.0)

    for _ in range(warmup_steps):
        _train_step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        base_vram_bytes = torch.cuda.memory_allocated(device)

    step_times: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        step_times.append(_train_step())
    end = time.perf_counter()

    wall = end - start
    step_time_mean = sum(step_times) / len(step_times)
    tokens_per_second = (tokens_per_step * steps) / wall

    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        base_vram_mb = float(base_vram_bytes) / (1024**2) if base_vram_bytes is not None else None
        peak_vram_mb = float(peak_vram_bytes) / (1024**2)
        training_vram_mb = (
            float(peak_vram_bytes - base_vram_bytes) / (1024**2)
            if base_vram_bytes is not None
            else None
        )
    else:
        base_vram_mb = peak_vram_mb = training_vram_mb = None

    return step_time_mean, tokens_per_second, base_vram_mb, peak_vram_mb, training_vram_mb, compile_time_ms


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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")

    train_cfg = TrainConfig()
    compile_enabled = bool(train_cfg.compile)

    cfg = VLMConfig()
    cfg.momh_enabled = bool(args.momh)

    # Synthetic mode uses a dummy tokenizer to avoid HF tokenizer overhead and to ensure a stable image_token_id.
    tokenizer = None
    if args.mode == "synthetic":
        tokenizer = _DummyTokenizer(
            image_token_id=cfg.lm_vocab_size - 1,
            pad_token_id=0,
            eos_token_id=1,
        )

    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer)
    if compile_enabled:
        _compile_regions(model, dynamic=None, mode="reduce-overhead")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype = _autocast_dtype(args)
    variants = _build_shape_variants(args)

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

        step_time_mean, tokens_per_second, base_vram_mb, peak_vram_mb, training_vram_mb, compile_time_ms = _run_train_steps(
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
        )

        result = BenchmarkResult(
            mode=args.mode,
            momh_enabled=bool(args.momh),
            compile=compile_enabled,
            device=str(device),
            dtype=args.dtype,
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

        step_time_mean, tokens_per_second, base_vram_mb, peak_vram_mb, training_vram_mb, compile_time_ms = _run_train_steps(
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
        )

        result = BenchmarkResult(
            mode=args.mode,
            momh_enabled=bool(args.momh),
            compile=compile_enabled,
            device=str(device),
            dtype=args.dtype,
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
