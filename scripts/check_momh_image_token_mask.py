#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.collators import VQACollator
from data.datasets import VQAIterableDataset
from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from models.momh_attention import generate_momh_mask_mod_from_modality

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `datasets`. Install dependencies and rerun:\n"
        "  source .venv/bin/activate\n"
        "  uv sync\n"
    ) from e


@dataclass(frozen=True)
class SampleSummary:
    raw_images: int
    patches_per_raw_image: list[int]
    total_patches: int
    image_token_placeholders: int
    content_start: int
    image_token_first: int | None
    image_token_last: int | None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate that MoMH masking in `models/momh_attention.py` correctly treats all image-token "
            "placeholders as 'vision' tokens, using a real dataset example and a reference mask derived "
            "from token IDs.\n\n"
            "This catches the failure mode where MoMH mis-identifies which tokens are vision (image) "
            "tokens, especially for multi-image or multi-patch samples."
        )
    )
    p.add_argument("--dataset", default="patrickamadeus/the_cauldron", help="HF dataset path (owner/name).")
    p.add_argument(
        "--config",
        default="sample_1pct",
        help="HF dataset config/subset name (passed as the second argument to load_dataset).",
    )
    p.add_argument("--split", default="train", help="Dataset split to scan (e.g. train/validation/test).")
    p.add_argument("--streaming", action="store_true", help="Use streaming=True.")

    p.add_argument(
        "--search-limit",
        type=int,
        default=500,
        help="Max processed examples to try before giving up.",
    )
    p.add_argument(
        "--min-raw-images",
        type=int,
        default=2,
        help="Require at least this many raw images in the sample (processed `len(sample['images'])`).",
    )
    p.add_argument(
        "--min-total-patches",
        type=int,
        default=2,
        help="Require at least this many total patches across all images.",
    )

    p.add_argument(
        "--vision-token-set",
        choices=["image_only", "vlm_extra_tokens"],
        default="image_only",
        help=(
            "What to treat as 'vision' tokens for the reference mask. "
            "`image_only` uses only `<|image|>` placeholders. "
            "`vlm_extra_tokens` also includes `<|global_image|>` and the `<row_X_col_Y>` tokens."
        ),
    )

    p.add_argument(
        "--max-qkv",
        type=int,
        default=512,
        help="Number of query/key positions to sample for mask comparison (keeps the check fast).",
    )
    p.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to write a JSON report with counts and sample metadata.",
    )
    p.add_argument(
        "--fail-on-any-mismatch",
        action="store_true",
        help="Exit with code 2 if the current MoMH mask differs from the reference on sampled positions.",
    )
    return p.parse_args(argv)


def _select_positions(pos: torch.Tensor, max_n: int) -> torch.Tensor:
    pos = pos.flatten()
    if pos.numel() <= max_n:
        return pos
    idx = torch.linspace(0, pos.numel() - 1, steps=max_n, dtype=torch.long)
    return pos[idx]


def _token_id_set_for_reference(tokenizer, cfg: VLMConfig, mode: str) -> set[int]:
    if mode == "image_only":
        return {int(tokenizer.image_token_id)}

    ids: set[int] = set()
    for tok in cfg.vlm_extra_tokens.values():
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or int(tok_id) < 0:
            raise RuntimeError(f"Tokenizer does not know extra token: {tok!r}")
        ids.add(int(tok_id))
    return ids


def _iter_processed_samples(ds_iterable: Iterable[dict[str, Any]], cfg: VLMConfig):
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, cfg.resize_to_max_side_len)
    vqa = VQAIterableDataset(
        ds_iterable,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
        relevance_min_rating=1,
        image_correspondence_min_rating=1,
        visual_dependency_min_rating=1,
        formatting_min_rating=1,
    )
    return tokenizer, vqa


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    cfg = VLMConfig()

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    tokenizer, vqa_iter = _iter_processed_samples(ds, cfg)

    collator = VQACollator(tokenizer, max_length=cfg.lm_max_length)
    vision_ids = _token_id_set_for_reference(tokenizer, cfg, args.vision_token_set)

    found = None
    for processed_idx, sample in enumerate(vqa_iter):
        if processed_idx >= args.search_limit:
            break
        if sample is None:
            continue
        raw_images = len(sample["images"])
        patches_per_raw_image = [int(t.shape[0]) for t in sample["images"]]
        total_patches = int(sum(patches_per_raw_image))

        if raw_images < args.min_raw_images:
            continue
        if total_patches < args.min_total_patches:
            continue

        found = sample
        break

    if found is None:
        print(
            f"No matching sample found within search-limit={args.search_limit}. "
            f"Tried requiring min_raw_images={args.min_raw_images}, min_total_patches={args.min_total_patches}.",
            file=sys.stderr,
        )
        return 3

    batch = collator([found])
    input_ids: torch.Tensor = batch["input_ids"]  # [1, T]
    attention_mask: torch.Tensor = batch["attention_mask"]  # [1, T]

    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise RuntimeError(f"Expected a single-example batch, got input_ids shape={tuple(input_ids.shape)}")

    content_start = int(attention_mask[0].argmax(dim=0).item())

    image_pos = (input_ids[0] == tokenizer.image_token_id).nonzero(as_tuple=False).flatten()
    image_token_placeholders = int(image_pos.numel())

    raw_images = len(found["images"])
    patches_per_raw_image = [int(t.shape[0]) for t in found["images"]]
    total_patches = int(sum(patches_per_raw_image))
    expected_placeholders = total_patches * int(cfg.mp_image_token_length)

    if image_token_placeholders != expected_placeholders:
        raise RuntimeError(
            "Invariant violated: number of `<|image|>` placeholders does not match processed image patches.\n"
            f"  image_token_placeholders={image_token_placeholders}\n"
            f"  total_patches={total_patches}\n"
            f"  mp_image_token_length={cfg.mp_image_token_length}\n"
            f"  expected_placeholders={expected_placeholders}\n"
            "If this is intended, update this script's invariant (and document the new contract)."
        )

    image_token_first = int(image_pos.min().item()) if image_pos.numel() else None
    image_token_last = int(image_pos.max().item()) if image_pos.numel() else None

    summary = SampleSummary(
        raw_images=raw_images,
        patches_per_raw_image=patches_per_raw_image,
        total_patches=total_patches,
        image_token_placeholders=image_token_placeholders,
        content_start=content_start,
        image_token_first=image_token_first,
        image_token_last=image_token_last,
    )

    # Build a reference "vision token" mask based on token IDs.
    is_content = attention_mask.to(torch.bool)
    is_vision_ref = torch.isin(
        input_ids, torch.tensor(sorted(vision_ids), device=input_ids.device)
    ).to(torch.bool)
    is_padding = ~is_content

    # Current MoMH behavior: treat only `<|image|>` placeholders as vision.
    is_vision_current = (input_ids == tokenizer.image_token_id)

    # Compare current MoMH mask_mod against a reference mask_mod on a sampled subset of positions.
    H = int(cfg.lm_n_heads)
    pct_v = float(getattr(cfg, "momh_head_pct_vision", 0.4))
    pct_t = float(getattr(cfg, "momh_head_pct_text", 0.4))
    H_V = int(H * pct_v)
    H_T = int(H * pct_t)
    H_T_start = H_V
    H_VT_start = H_V + H_T

    current_mask_mod = generate_momh_mask_mod_from_modality(
        H,
        is_vision=is_vision_current,
        attention_mask=is_content,
        q_offset=0,
        pct_v=pct_v,
        pct_t=pct_t,
    )

    def ref_mask_mod(b, h, q_idx, kv_idx):
        q_is_padding = is_padding[b, q_idx]
        kv_is_padding = is_padding[b, kv_idx]
        not_padding = ~q_is_padding & ~kv_is_padding

        q_is_vision = is_vision_ref[b, q_idx]
        kv_is_vision = is_vision_ref[b, kv_idx]
        q_is_text = ~q_is_vision
        kv_is_text = ~kv_is_vision

        head_V = (h < H_T_start) & q_is_vision & kv_is_vision & not_padding
        head_T = (h >= H_T_start) & (h < H_VT_start) & q_is_text & kv_is_text & (q_idx >= kv_idx) & not_padding
        head_VT = (h >= H_VT_start) & not_padding & (kv_is_vision | (q_idx >= kv_idx))
        return head_V | head_T | head_VT

    # Choose q/kv positions to sample: include both vision placeholders and text.
    content_pos = is_content[0].nonzero(as_tuple=False).flatten()
    vision_pos = (is_content[0] & is_vision_ref[0]).nonzero(as_tuple=False).flatten()
    text_pos = (is_content[0] & ~is_vision_ref[0]).nonzero(as_tuple=False).flatten()

    q_sel = torch.unique(
        torch.cat(
            [
                _select_positions(vision_pos, max(1, args.max_qkv // 2)),
                _select_positions(text_pos, max(1, args.max_qkv // 2)),
            ]
        )
    )
    kv_sel = _select_positions(content_pos, args.max_qkv)

    # Shapes for broadcasting to [B=1, H, Q, K]
    b = torch.zeros(1, 1, 1, 1, dtype=torch.long, device=input_ids.device)
    h = torch.arange(H, dtype=torch.long, device=input_ids.device).view(1, H, 1, 1)
    q_idx = q_sel.to(torch.long).view(1, 1, -1, 1)
    kv_idx = kv_sel.to(torch.long).view(1, 1, 1, -1)

    cur = current_mask_mod(b, h, q_idx, kv_idx)
    ref = ref_mask_mod(b, h, q_idx, kv_idx)
    mismatch = (cur ^ ref)
    mismatch_count = int(mismatch.sum().item())

    # Count mismatches where the query is an image placeholder token.
    q_is_image_placeholder = (input_ids[0, q_sel] == tokenizer.image_token_id).view(1, 1, -1, 1)
    mismatch_image_q = int((mismatch & q_is_image_placeholder).sum().item())

    report = {
        "dataset": {"path": args.dataset, "config": args.config, "split": args.split, "streaming": bool(args.streaming)},
        "config": {
            "lm_max_length": int(cfg.lm_max_length),
            "lm_n_heads": int(cfg.lm_n_heads),
            "mp_image_token_length": int(cfg.mp_image_token_length),
            "momh_head_pct_vision": pct_v,
            "momh_head_pct_text": pct_t,
        },
        "sample_summary": {
            **summary.__dict__,
        },
        "checks": {
            "mask_mismatch_count_sampled": mismatch_count,
            "mask_mismatch_count_sampled_image_q": mismatch_image_q,
            "sampled": {"q": int(q_sel.numel()), "kv": int(kv_sel.numel()), "H": H, "H_V": H_V, "H_T": H_T, "H_VT": int(H - H_VT_start)},
        },
    }

    print(f"Dataset: {args.dataset} config={args.config} split={args.split} streaming={args.streaming}")
    print(
        "Sample: "
        f"raw_images={summary.raw_images} patches_per_raw_image={summary.patches_per_raw_image} total_patches={summary.total_patches}"
    )
    print(
        "Image placeholders: "
        f"count={summary.image_token_placeholders} first={summary.image_token_first} last={summary.image_token_last}"
    )
    print(
        "Mask comparison (sampled): "
        f"mismatch_total={mismatch_count} mismatch_image_q={mismatch_image_q} "
        f"(q={int(q_sel.numel())}, kv={int(kv_sel.numel())}, H={H}, token_set={args.vision_token_set})"
    )

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report: {args.dump_json}")

    if args.fail_on_any_mismatch and mismatch_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
