#!/usr/bin/env python3
"""
Minimal torch.compile debugging script with synthetic data.

NO network calls, NO disk I/O - starts in ~2 seconds.

Usage:
    # Basic run
    python train_compile_debug.py

    # With graph break logging
    TORCH_LOGS="graph_breaks" python train_compile_debug.py

    # With recompilation logging
    TORCH_LOGS="recompiles" python train_compile_debug.py

    # Full trace for tlparse
    TORCH_TRACE=/tmp/trace python train_compile_debug.py
    tlparse /tmp/trace

    # Without compile (baseline)
    python train_compile_debug.py --no-compile

    # More steps
    python train_compile_debug.py --steps 20
"""

import argparse
import time
import torch
import torch.optim as optim
from statistics import mean

from configs.config_debug import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.synthetic_dataset import get_synthetic_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--backend", type=str, default="inductor",
                       choices=["inductor", "eager", "aot_eager"],
                       help="Compile backend for ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Steps: {args.steps}, Batch size: {args.batch_size}")
    print(f"[DEBUG] Compile: {not args.no_compile}, Backend: {args.backend}")

    # Load config
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()

    # Create synthetic dataloader (instant!)
    print("[DEBUG] Creating synthetic dataloader...")
    print(f"[DEBUG] Config: seq_len={vlm_cfg.lm_max_length}, mp_tokens={vlm_cfg.mp_image_token_length}")
    dataloader = get_synthetic_dataloader(
        batch_size=args.batch_size,
        num_samples=args.steps * 2,  # Enough samples
        seq_len=vlm_cfg.lm_max_length,
        vocab_size=vlm_cfg.lm_vocab_size,
        image_size=vlm_cfg.vit_img_size,
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        images_per_sample=1,
    )
    print("[DEBUG] Dataloader ready.")

    # Create model (random init)
    print("[DEBUG] Creating model...")
    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] Model created: {num_params:,} parameters")
    model.to(device)

    # Compile
    if not args.no_compile:
        print(f"[DEBUG] Compiling with backend={args.backend}...")
        compile_start = time.time()
        model = torch.compile(model, backend=args.backend)
        print(f"[DEBUG] Compile setup: {time.time() - compile_start:.2f}s")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    print(f"[DEBUG] Starting {args.steps} training steps...")
    model.train()

    step_times = []
    fw_times = []
    bw_times = []

    data_iter = iter(dataloader)
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"]

        step_start = time.time()
        optimizer.zero_grad()

        # Forward
        fw_start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        fw_time = time.time() - fw_start

        # Backward
        bw_start = time.time()
        loss.backward()
        bw_time = time.time() - bw_start

        optimizer.step()
        step_time = time.time() - step_start

        step_times.append(step_time)
        fw_times.append(fw_time)
        bw_times.append(bw_time)

        if step == 0:
            print(f"[DEBUG] Step 0 (includes compile): {step_time:.2f}s (fw: {fw_time:.2f}s, bw: {bw_time:.2f}s)")
        elif step % 5 == 0 or step == args.steps - 1:
            print(f"[DEBUG] Step {step}: {step_time:.3f}s, loss: {loss.item():.4f}")

    # Summary (exclude first step which includes compile)
    if len(step_times) > 1:
        avg_step = mean(step_times[1:])
        avg_fw = mean(fw_times[1:])
        avg_bw = mean(bw_times[1:])
        print(f"\n[DEBUG] === Summary (excluding step 0) ===")
        print(f"[DEBUG] Avg step time: {avg_step:.3f}s")
        print(f"[DEBUG] Avg forward:   {avg_fw:.3f}s")
        print(f"[DEBUG] Avg backward:  {avg_bw:.3f}s")
        print(f"[DEBUG] First step:    {step_times[0]:.2f}s (includes compile)")
        print(f"[DEBUG] Speedup from compile: {step_times[0]/avg_step:.1f}x (first vs avg)")


if __name__ == "__main__":
    main()
