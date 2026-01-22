#!/usr/bin/env python3
"""
Debug training script for torch.compile optimization testing.

Uses smaller model config for ~10-20x faster iteration:
- 4 LM blocks (vs 32)
- 4 ViT blocks (vs 12)
- 256 hidden dim (vs 960)
- No pretrained weights (random init)
- 20 training steps
- No wandb logging

Usage:
    # Basic run
    python train_debug.py

    # With torch.compile diagnostics
    TORCH_LOGS="graph_breaks,recompiles" python train_debug.py

    # Full trace for tlparse
    TORCH_TRACE=/tmp/nanovlm_trace python train_debug.py
    tlparse /tmp/nanovlm_trace
"""

import os
import re
import json
import math
import time
import torch
import wandb
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
import torch.distributed as dist
from tqdm import tqdm

from data.data_utils import synchronized_dataloader_step

# Import DEBUG config instead of vanilla
from configs.config_debug import VLMConfig, TrainConfig, GlobalConfig
from models.vision_language_model import VisionLanguageModel
from train_utils import (
    create_cpu_group, destroy_dist, dist_gather, dist_mean_scalar,
    evaluate_validation, get_dataloaders, get_rank, get_run_name,
    get_world_size, init_dist, is_dist, is_master,
    save_model_checkpoint, set_pg_cpu, set_seed, wrap_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg, vlm_cfg, global_cfg):
    train_loader, val_loader, iter_train_loader, _ = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project=train_cfg.wandb_project,
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg),
                "GlobalConfig": asdict(global_cfg),
            },
            name=run_name,
        )
        lmms_eval_step = "<lmms-eval-step>"
        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # Initialize model with random weights (no pretrained for debug)
    model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    if is_master():
        print(f"[DEBUG] nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"[DEBUG] Training for {train_cfg.max_training_steps} steps")

    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp})
    else:
        for p in list(model.MP.parameters()):
            p.requires_grad = False
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_vision_backbone})
    else:
        for p in list(model.vision_encoder.parameters()):
            p.requires_grad = False
    if train_cfg.lr_language_backbone > 0:
        param_groups.append({'params': list(model.decoder.parameters()), 'lr': train_cfg.lr_language_backbone})
    else:
        for p in list(model.decoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()

    print(f"[DEBUG] Using device: {device}")
    model.to(device)

    if train_cfg.compile:
        print("[DEBUG] Compiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model)
        print(f"[DEBUG] torch.compile setup took {time.time() - compile_start:.2f}s")

    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    global_step = 0
    epoch = 0
    train_pbar = None
    current_lrs = {}

    if train_cfg.stream_dataset:
        train_pbar = tqdm(
            total=train_cfg.max_training_steps,
            desc="Training",
            leave=False,
            disable=not is_master(),
        )

    accumulated_stats = {
        'tokens_per_second': [],
        'fw_bw_time': [],
    }

    print("[DEBUG] Starting training loop...")
    first_step_time = None

    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist())):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()
            fw_bw_time = time.time() - fw_bw_start

            if is_update_step:
                if first_step_time is None:
                    first_step_time = time.time() - batch_start_time
                    print(f"[DEBUG] First step (including compile): {first_step_time:.2f}s")

                current_lrs = {}
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    param_group_idx += 1

                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration

            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)

            if train_pbar is not None and is_master():
                if train_cfg.stream_dataset:
                    if is_update_step:
                        train_pbar.update(1)
                train_pbar.set_postfix({
                    "Loss": f"{batch_loss:.4f}",
                    "Step": f"{global_step}",
                    "fw_bw": f"{fw_bw_time:.3f}s",
                })

            if is_update_step:
                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break

        iter_train_loader = iter(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

    if train_pbar is not None:
        train_pbar.close()

    # Summary
    if is_master() and len(accumulated_stats['fw_bw_time']) > 0:
        avg_fw_bw = mean(accumulated_stats['fw_bw_time'][1:])  # Skip first (compile)
        avg_tokens_per_sec = mean(accumulated_stats['tokens_per_second'][1:])
        print(f"\n[DEBUG] === Summary ===")
        print(f"[DEBUG] Steps completed: {global_step}")
        print(f"[DEBUG] First step (w/ compile): {first_step_time:.2f}s")
        print(f"[DEBUG] Avg fw+bw time (after compile): {avg_fw_bw:.3f}s")
        print(f"[DEBUG] Avg tokens/sec: {avg_tokens_per_sec:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', type=bool, default=True, help='Use torch.compile')
    parser.add_argument('--max_training_steps', type=int, default=None, help='Override max steps')
    args = parser.parse_args()

    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.max_training_steps is not None:
        train_cfg.max_training_steps = args.max_training_steps

    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        set_pg_cpu(create_cpu_group())

    set_seed(global_cfg)

    if is_master():
        print("[DEBUG] === Debug Config ===")
        print(f"[DEBUG] LM blocks: {vlm_cfg.lm_n_blocks}, hidden: {vlm_cfg.lm_hidden_dim}")
        print(f"[DEBUG] ViT blocks: {vlm_cfg.vit_n_blocks}, hidden: {vlm_cfg.vit_hidden_dim}")
        print(f"[DEBUG] Compile: {train_cfg.compile}")
        print(f"[DEBUG] Max steps: {train_cfg.max_training_steps}")

    train(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
