from dotenv import load_dotenv

load_dotenv()
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

from models.config import VLMConfig, TrainConfig, GlobalConfig
from models.vision_language_model import VisionLanguageModel
from train_utils import (
    create_cpu_group, destroy_dist, dist_gather, dist_mean_scalar,
    evaluate_validation, get_dataloaders, get_rank, get_run_name,
    get_world_size, init_dist, is_dist, is_master,
    save_model_checkpoint, set_pg_cpu, set_seed, wrap_model,
)
from utils.checkpointing import (
    capture_rng_state,
    load_model_optimizer_state,
    load_trainer_state,
    restore_rng_state,
    save_model_optimizer_state,
    save_trainer_state,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

def str2bool(value):
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y", "t"}:
        return True
    if value_lower in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def fast_forward_dataloader(iter_loader, num_batches: int, *, strict: bool):
    if num_batches <= 0:
        return iter_loader

    skipped = 0
    for _ in synchronized_dataloader_step(iter_loader, is_dist()):
        skipped += 1
        if skipped >= num_batches:
            break

    if skipped < num_batches and strict:
        raise RuntimeError(
            f"Unable to fast-forward dataloader by {num_batches} batches; only skipped {skipped}."
        )
    return iter_loader


def _compile_module_list(modules, *, dynamic: bool | None = None, mode: str | None = "reduce-overhead"):
    for idx, block in enumerate(modules):
        modules[idx] = torch.compile(block, dynamic=dynamic, mode=mode)


def compile_regions(model, *, dynamic: bool | None = None, mode: str | None = "reduce-overhead"):
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


def _resolve_compile_mode(mode: str | None) -> str | None:
    allowed_modes = {"default", "reduce-overhead", "max-autotune"}
    if mode is None:
        return None
    if mode not in allowed_modes:
        raise ValueError(f"Unsupported compile_mode: {mode}. Allowed values: {sorted(allowed_modes)}")
    return mode


def compute_effective_token_scale(
    effective_tokens: int, denom_tokens: int, exponent: float
) -> tuple[float, float]:
    ratio = effective_tokens / max(denom_tokens, 1)
    ratio = min(max(ratio, 1e-6), 1.0)
    return ratio, ratio**exponent

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg, vlm_cfg, global_cfg):
    resume_state = None
    resume_global_step = 0
    resume_epoch = 0
    resume_micro_step = 0
    resume_warmup_batches = 0
    resume_tokens_processed_global = 0
    resume_source_run_name = None

    if train_cfg.resume_from_checkpoint:
        resume_state = load_trainer_state(train_cfg.resume_from_checkpoint, strict=True)
        if resume_state:
            resume_global_step = int(resume_state.get("global_step", 0))
            resume_epoch = int(resume_state.get("epoch", 0))
            resume_micro_step = int(resume_state.get("micro_step_in_epoch", 0))
            resume_warmup_batches = int(resume_state.get("warmup_batches", 0))
            resume_tokens_processed_global = int(resume_state.get("tokens_processed_global", 0))
            resume_source_run_name = resume_state.get("run_name")
            rng_state = resume_state.get("rng_state")
            if rng_state:
                restore_rng_state(rng_state)

    do_warmup = not train_cfg.resume_from_checkpoint
    warmup_batches = 1 if do_warmup else 0

    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(
        train_cfg, vlm_cfg, global_cfg, do_warmup=do_warmup
    )

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers have gotten dataloaders.")

    resume_skip_batches = resume_warmup_batches + resume_micro_step
    if resume_skip_batches > 0:
        if train_cfg.stream_dataset:
            if is_master():
                print("Warning: resume_from_checkpoint with stream_dataset=True will not be bitwise deterministic.")
        else:
            iter_train_loader = fast_forward_dataloader(
                iter_train_loader,
                resume_skip_batches,
                strict=True,
            )

    run_name = get_run_name(train_cfg, vlm_cfg)
    if train_cfg.resume_from_checkpoint:
        run_name = resume_source_run_name or run_name
    if train_cfg.checkpoint_format not in ("dcp", "torch"):
        raise ValueError(f"Unsupported checkpoint_format: {train_cfg.checkpoint_format}")
    if train_cfg.resume_from_checkpoint and is_master():
        print(
            f"Resuming from checkpoint: {train_cfg.resume_from_checkpoint} "
            f"(global_step={resume_global_step}, epoch={resume_epoch}, micro_step_in_epoch={resume_micro_step})"
        )
    tokens_step_metric = "tokens/consumed"
    lmms_eval_step = "<lmms-eval-step>"
    run = None
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
        if getattr(train_cfg, "wandb_xaxis_tokens", False):
            run.define_metric(tokens_step_metric)
            run.define_metric("batch_loss", step_metric=tokens_step_metric)
            run.define_metric("val_loss", step_metric=tokens_step_metric)
            run.define_metric("grad_norm", step_metric=tokens_step_metric)
            run.define_metric("training_stats/*", step_metric=tokens_step_metric)
            run.define_metric("epoch_*", step_metric=tokens_step_metric)
            lmms_eval_step = tokens_step_metric

        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # Initialize model
    if train_cfg.resume_from_checkpoint:
        if vlm_cfg.vlm_load_backbone_weights:
            if is_master():
                print("resume_from_checkpoint enabled: disabling backbone weight loading.")
            vlm_cfg.vlm_load_backbone_weights = False
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    elif train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    use_selective_ac = bool(train_cfg.compile and getattr(vlm_cfg, "activation_checkpointing", False))
    if hasattr(model, "set_activation_checkpointing_mode"):
        model.set_activation_checkpointing_mode(
            use_selective=use_selective_ac,
            allow_cache_entry_mutation=use_selective_ac,
        )
        if is_master() and use_selective_ac:
            print(
                "Using selective activation checkpointing under torch.compile "
                "(allow_cache_entry_mutation=True)."
            )
    
    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
        print(f"Training summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
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
    
    print(f"Using device: {device}")
    model.to(device)

    if train_cfg.resume_from_checkpoint:
        use_dcp = train_cfg.checkpoint_format == "dcp"
        load_model_optimizer_state(
            train_cfg.resume_from_checkpoint,
            model,
            optimizer,
            use_dcp=use_dcp,
            strict=True,
        )

    compile_mode = _resolve_compile_mode(getattr(train_cfg, "compile_mode", "default"))

    if getattr(train_cfg, "activation_memory_budget", None) is not None:
        if not 0.0 <= train_cfg.activation_memory_budget <= 1.0:
            raise ValueError("activation_memory_budget must be between 0 and 1.")
        budget_cfg = None
        if hasattr(torch._dynamo.config, "activation_memory_budget"):
            budget_cfg = torch._dynamo.config
        elif hasattr(torch._functorch.config, "activation_memory_budget"):
            budget_cfg = torch._functorch.config
        if budget_cfg is None:
            raise RuntimeError("activation_memory_budget is not supported in this PyTorch build.")
        if train_cfg.compile:
            budget_cfg.activation_memory_budget = train_cfg.activation_memory_budget
            if is_master():
                print(f"Using activation_memory_budget={train_cfg.activation_memory_budget}")
        else:
            if is_master():
                print("activation_memory_budget set but compile is disabled; ignoring.")

    if train_cfg.compile:
        compile_regions(model, mode=compile_mode)
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    best_model_path = None
    logged_eval_steps = set()
    global_step = resume_global_step
    epoch = resume_epoch
    micro_step_in_epoch = resume_micro_step
    train_pbar = None
    current_lrs = {}
    tokens_processed_global = resume_tokens_processed_global
    effective_tokens_accum = 0
    if train_cfg.stream_dataset:
        train_pbar = tqdm(
            total=train_cfg.max_training_steps,
            desc="Training",
            leave=False,
            disable=not is_master(),
        )
    
    # Training stats accumulators
    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
        'effective_tokens': [],  # Tokens without padding
        'total_tokens': [],      # Total tokens including padding
    }
    
    while global_step < train_cfg.max_training_steps:
        current_epoch = epoch + 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        num_batches = 0
        optimizer.zero_grad()
        data_load_start = time.time()
        accumulated_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_loss_tokens = torch.zeros((), device=device, dtype=torch.float32)

        if not train_cfg.stream_dataset:
            train_pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {current_epoch}",
                leave=False,
                disable=not is_master(),
            )

        print("Starting training loop")
        for i, batch in enumerate(
            synchronized_dataloader_step(iter_train_loader, is_dist()),
            start=micro_step_in_epoch,
        ):
            if (
                train_cfg.compile
                and device.type == "cuda"
                and hasattr(torch, "compiler")
                and hasattr(torch.compiler, "cudagraph_mark_step_begin")
            ):
                torch.compiler.cudagraph_mark_step_begin()
            num_batches += 1
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            micro_step_in_epoch = i + 1
            step_effective_tokens = None
            step_effective_token_ratio = None
            step_effective_token_lr_scale = 1.0
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            data_load_time = time.time() - data_load_start

            num_tokens = int(torch.sum(attention_mask).item())
            tokens_processed_global += num_tokens
            effective_tokens_accum += num_tokens

            if train_cfg.compile:
                # Always mark (B,T) dynamic when compiling to reduce recompiles from variable batch/seq.
                torch._dynamo.maybe_mark_dynamic(input_ids, 0)
                torch._dynamo.maybe_mark_dynamic(input_ids, 1)
                torch._dynamo.maybe_mark_dynamic(labels, 0)
                torch._dynamo.maybe_mark_dynamic(labels, 1)
                torch._dynamo.maybe_mark_dynamic(attention_mask, 0)
                torch._dynamo.maybe_mark_dynamic(attention_mask, 1)

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
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
                    _, loss, loss_token_count = model(
                        input_ids,
                        images,
                        attention_mask=attention_mask,
                        targets=labels,
                        loss_reduction="sum",
                        return_loss_count=True,
                    )

            loss_token_count_value = int(loss_token_count.item())
            if loss_token_count_value == 0:
                raise ValueError("Found a batch with no valid target tokens; check label masking.")

            accumulated_loss_sum += loss.detach().to(dtype=torch.float32)
            accumulated_loss_tokens += loss_token_count.to(dtype=torch.float32)
            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                current_lrs = {}
                total_loss_tokens = accumulated_loss_tokens.clone()
                total_loss_sum = accumulated_loss_sum.clone()
                if is_dist():
                    dist.all_reduce(total_loss_tokens, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)

                total_loss_tokens_value = total_loss_tokens.item()
                if total_loss_tokens_value == 0:
                    raise ValueError("Gradient accumulation produced zero total tokens; check label masking.")

                grad_scale = get_world_size() / total_loss_tokens_value
                for param in all_params:
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                denom_tokens = (
                    train_cfg.batch_size
                    * train_cfg.gradient_accumulation_steps
                    * get_world_size()
                    * vlm_cfg.lm_max_length
                )
                if getattr(train_cfg, "effective_token_lr_scale", False) or train_cfg.log_wandb:
                    step_effective_tokens = effective_tokens_accum
                    if is_dist():
                        step_effective_tokens = sum(dist_gather(step_effective_tokens))
                    step_effective_token_ratio, step_effective_token_lr_scale = compute_effective_token_scale(
                        step_effective_tokens,
                        denom_tokens,
                        getattr(train_cfg, "effective_token_lr_exponent", 0.5),
                    )
                    if not getattr(train_cfg, "effective_token_lr_scale", False):
                        step_effective_token_lr_scale = 1.0

                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = (
                        get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                        * step_effective_token_lr_scale
                    )
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    current_lrs["train/lr_mp"] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = (
                        get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                        * step_effective_token_lr_scale
                    )
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    current_lrs["train/lr_vision_backbone"] = adj_lr_vision_backbone
                    param_group_idx += 1

                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = (
                        get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                        * step_effective_token_lr_scale
                    )
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone
                    current_lrs["train/lr_language_backbone"] = adj_lr_language_backbone
              
                optimizer.step()
                optimizer.zero_grad()
                effective_tokens_accum = 0
                accumulated_loss_tokens.zero_()
                accumulated_loss_sum.zero_()

            batch_loss = loss.item() / loss_token_count_value
            total_train_loss += batch_loss

            num_tokens = int(num_tokens) # Sum of attention mask gives number of tokens (effective tokens)
            total_batch_tokens = attention_mask.numel()   # Total tokens including padding
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['data_load_time'].append(data_load_time)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)
            accumulated_stats['post_process_time'].append(post_process_time)
            accumulated_stats['images_per_sample'].extend(images_per_sample)
            accumulated_stats['effective_tokens'].append(num_tokens)
            accumulated_stats['total_tokens'].append(int(total_batch_tokens))

            if train_pbar is not None and is_master():
                if train_cfg.stream_dataset:
                    if is_update_step:
                        train_pbar.update(1)
                else:
                    train_pbar.update(1)
                train_pbar.set_postfix({
                    "Loss": f"{batch_loss:.4f}",
                    "Step": f"{global_step}",
                })

            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0 and is_update_step:
                print("Starting evaluation")
                avg_val_loss, min_val_loss, max_val_loss = evaluate_validation(
                    model, val_loader, device, train_cfg
                )
                iter_val_loader = iter(val_loader)

                checkpoint_path_step = ""
                if is_master():
                    # Save a checkpoint for this evaluation step only when explicitly needed.
                    if train_cfg.save_local or train_cfg.use_lmms_eval:
                        checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
                        save_model = model.module if is_dist() else model # unwrap the model for saving if DDP
                        save_model.save_pretrained(save_directory=checkpoint_path_step)

                    if train_cfg.use_lmms_eval and checkpoint_path_step and global_step % (train_cfg.eval_interval*2) == 0:
                        # Submit evaluation job
                        cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {train_cfg.lmms_eval_limit} {train_cfg.lmms_eval_tasks} {train_cfg.lmms_eval_batch_size}"
                        print(f"Submitting evaluation job: {cmd}")
                        subprocess.run(cmd, shell=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if is_master():
                        best_model_path = checkpoint_path_step

                if is_master():
                    print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                    if train_cfg.log_wandb:
                        log_payload = {
                            "val_loss": avg_val_loss,
                            "val/min_val_loss": min_val_loss,
                            "val/max_val_loss": max_val_loss,
                        }
                        if getattr(train_cfg, "wandb_xaxis_tokens", False):
                            tokens_step_value = sum(dist_gather(tokens_processed_global)) if is_dist() else tokens_processed_global
                            log_payload[tokens_step_metric] = tokens_step_value
                        run.log(log_payload, step=global_step)

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops)
            if global_step % train_cfg.stats_log_interval == 0 and len(accumulated_stats['tokens_per_second']) > 0 and is_update_step:
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f'avg_{key}'] = mean(all_values_flat)
                    else:
                        stats[f'avg_{key}'] = mean(accumulated_stats[key])
                
                for key in ['data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f'max_{key}'] = max(all_values_flat)
                    else:
                        stats[f'max_{key}'] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats['images_per_sample'])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats['min_images_per_sample'] = min(all_images_flat)
                else:
                    stats['min_images_per_sample'] = min(accumulated_stats['images_per_sample'])

                # Compute token efficiency (sync all-reduce if enabled, otherwise local only)
                local_effective = sum(accumulated_stats['effective_tokens'])
                local_total = sum(accumulated_stats['total_tokens'])

                if train_cfg.sync_token_efficiency and is_dist():
                    token_tensor = torch.tensor([local_effective, local_total],
                                                device=torch.cuda.current_device(),
                                                dtype=torch.float64)
                    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
                    global_effective = token_tensor[0].item()
                    global_total = token_tensor[1].item()
                else:
                    global_effective = local_effective
                    global_total = local_total

                token_efficiency = global_effective / global_total if global_total > 0 else 1.0

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    log_payload = {
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                        "training_stats/effective_tokens": global_effective,
                        "training_stats/total_tokens": global_total,
                        "training_stats/token_efficiency": token_efficiency,
                    }
                    if getattr(train_cfg, "wandb_xaxis_tokens", False):
                        tokens_step_value = sum(dist_gather(tokens_processed_global)) if is_dist() else tokens_processed_global
                        log_payload[tokens_step_metric] = tokens_step_value
                    if step_effective_tokens is not None:
                        log_payload["effective_tokens"] = step_effective_tokens
                    if step_effective_token_ratio is not None:
                        log_payload["effective_token_ratio"] = step_effective_token_ratio
                        log_payload["effective_token_lr_scale"] = step_effective_token_lr_scale
                    run.log(log_payload, step=global_step)

                    # Check for and log new lmms-eval results
                    eval_results_dir = os.path.join('eval_results', run_name)
                    if os.path.exists(eval_results_dir):
                        logged_results_count = 0
                        for result_file in os.listdir(eval_results_dir):
                            # Match only files like "step_1234.json" (no extra text)
                            match = re.fullmatch(r"step_(\d+)\.json", result_file)
                            if not match:
                                continue  # skip if the filename has extra text like taskname

                            try:
                                step = int(match.group(1))
                                if step not in logged_eval_steps:
                                    with open(os.path.join(eval_results_dir, result_file), 'r') as f:
                                        eval_data = json.load(f)

                                    lmms_results = eval_data.get('results', {})
                                    if lmms_results:
                                        metrics = {f"lmms_eval/{key}": value for key, value in lmms_results.items()}
                                        metrics[lmms_eval_step] = eval_data['global_step']
                                        if logged_results_count > 0:
                                            print(f"Logging more than one lmms-eval result for step {global_step}, try to avoid this.")
                                        run.log(metrics, step=global_step + logged_results_count)
                                        logged_results_count += 1
                                        print(f"Logged lmms-eval results from step {eval_data['global_step']}")

                                    logged_eval_steps.add(step)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                print(f"Warning: Could not process eval result file {result_file}. Error: {e}")
                                continue
                
                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss  
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = dist_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss
                    
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    log_payload = {
                        "batch_loss": batch_loss_gathered,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {}),
                        **current_lrs,
                    }
                    if getattr(train_cfg, "wandb_xaxis_tokens", False):
                        tokens_step_value = sum(dist_gather(tokens_processed_global)) if is_dist() else tokens_processed_global
                        log_payload[tokens_step_metric] = tokens_step_value
                    run.log(log_payload, step=global_step)
                
            if is_update_step:
                global_step += 1
                if train_cfg.save_model_every_n_steps and global_step % train_cfg.save_model_every_n_steps == 0:
                    save_model_checkpoint(model, train_cfg, global_step=global_step)
                if (
                    train_cfg.checkpoint_every_n_steps > 0
                    and global_step % train_cfg.checkpoint_every_n_steps == 0
                ):
                    checkpoint_path = os.path.join(
                        train_cfg.checkpoint_dir,
                        run_name,
                        f"step_{global_step}",
                    )
                    use_dcp = train_cfg.checkpoint_format == "dcp"
                    save_format = save_model_optimizer_state(
                        checkpoint_path,
                        model,
                        optimizer,
                        use_dcp=use_dcp,
                        strict=True,
                    )
                    trainer_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "micro_step_in_epoch": micro_step_in_epoch,
                        "warmup_batches": warmup_batches,
                        "tokens_processed_global": tokens_processed_global,
                        "run_name": run_name,
                        "rng_state": capture_rng_state(),
                        "train_cfg": asdict(train_cfg),
                        "vlm_cfg": asdict(vlm_cfg),
                    }
                    save_trainer_state(checkpoint_path, trainer_state)

                    if is_master():
                        meta_path = os.path.join(checkpoint_path, "meta.json")
                        with open(meta_path, "w") as f:
                            json.dump(
                                {
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "micro_step_in_epoch": micro_step_in_epoch,
                                    "warmup_batches": warmup_batches,
                                    "tokens_processed_global": tokens_processed_global,
                                    "run_name": run_name,
                                    "checkpoint_format": save_format,
                                },
                                f,
                                indent=2,
                            )
                        print(f"Saved checkpoint to {checkpoint_path} ({save_format})")
                if getattr(train_cfg, "max_training_tokens", None) is not None:
                    tokens_processed = sum(dist_gather(tokens_processed_global)) if is_dist() else tokens_processed_global
                    if tokens_processed >= train_cfg.max_training_tokens:
                        if is_master():
                            print(
                                f"Stopping: tokens_processed={tokens_processed} "
                                f">= max_training_tokens={train_cfg.max_training_tokens}"
                            )
                        return
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        if not train_cfg.stream_dataset and train_pbar is not None and is_master():
            train_pbar.close()
        if num_batches == 0:
            break
        avg_train_loss = total_train_loss / num_batches
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                log_payload = {
                    "epoch_loss": avg_train_loss,
                    "epoch_duration": epoch_duration,
                    "epoch_tokens_per_second": epoch_tokens_per_second,
                }
                if getattr(train_cfg, "wandb_xaxis_tokens", False):
                    tokens_step_value = sum(dist_gather(tokens_processed_global)) if is_dist() else tokens_processed_global
                    log_payload[tokens_step_metric] = tokens_step_value
                run.log(log_payload)

            print(f"Epoch: {current_epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")
        epoch += 1
        micro_step_in_epoch = 0

    if train_cfg.stream_dataset and train_pbar is not None and is_master():
        train_pbar.close()

    if train_cfg.save_local or train_cfg.save_hf:
        save_model_checkpoint(model, train_cfg, is_final=True)

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None and best_model_path:
            print(f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub...")
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_vision_backbone', type=float, help='Learning rate for the vision backbone')
    parser.add_argument('--lr_language_backbone', type=float, help='Learning rate for the language backbone')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', type=str2bool, help='Use torch.compile to optimize the model')
    parser.add_argument(
        '--compile_mode',
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        help='torch.compile mode when compile=True',
    )
    parser.add_argument('--activation_checkpointing', type=str2bool, help='Enable activation checkpointing for LM/VIT blocks')
    parser.add_argument('--activation_memory_budget', type=float, help='torch.compile activation memory budget (0-1)')
    parser.add_argument('--momh_enabled', type=str2bool, help='Enable MoMH attention')
    parser.add_argument('--max_training_steps', type=int, help='Maximum number of training steps')
    parser.add_argument('--max_training_tokens', type=int, help='Stop after this many effective tokens (non-padding)')
    parser.add_argument('--pack_sequences', type=str2bool, help='Enable packing multiple samples per sequence')
    parser.add_argument('--effective_token_lr_scale', type=str2bool, help='Scale LR by effective token ratio each step')
    parser.add_argument('--effective_token_lr_exponent', type=float, help='Exponent for effective token LR scaling')
    parser.add_argument('--wandb_xaxis_tokens', type=str2bool, help='Use tokens as wandb x-axis')
    parser.add_argument('--log_wandb', type=str2bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=str2bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--checkpoint_every_n_steps', type=int, help='Save training state every N optimizer steps')
    parser.add_argument('--checkpoint_dir', type=str, help='Base directory for training checkpoints')
    parser.add_argument('--checkpoint_format', type=str, help='Checkpoint format: dcp or torch')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint directory to resume from')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')
    parser.add_argument('--train_dataset_path', type=str, help='Train dataset path')
    parser.add_argument('--relevance_min_rating', type=int, help='Minimum relevance rating of images per sample')
    parser.add_argument('--image_correspondence_min_rating', type=int, help='Minimum image correspondence rating of images per sample')
    parser.add_argument('--visual_dependency_min_rating', type=int, help='Minimum visual dependency rating of images per sample')
    parser.add_argument('--formatting_min_rating', type=int, help='Minimum formatting rating of images per sample')

    args = parser.parse_args()

    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_vision_backbone is not None:
        train_cfg.lr_vision_backbone = args.lr_vision_backbone
    if args.lr_language_backbone is not None:
        train_cfg.lr_language_backbone = args.lr_language_backbone
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.compile_mode is not None:
        train_cfg.compile_mode = args.compile_mode
    if args.activation_checkpointing is not None:
        vlm_cfg.activation_checkpointing = args.activation_checkpointing
    if args.activation_memory_budget is not None:
        train_cfg.activation_memory_budget = args.activation_memory_budget
    if args.momh_enabled is not None:
        vlm_cfg.momh_enabled = args.momh_enabled
    if args.checkpoint_every_n_steps is not None:
        train_cfg.checkpoint_every_n_steps = args.checkpoint_every_n_steps
    if args.checkpoint_dir is not None:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_format is not None:
        train_cfg.checkpoint_format = args.checkpoint_format
    if args.resume_from_checkpoint is not None:
        train_cfg.resume_from_checkpoint = args.resume_from_checkpoint
    if args.max_training_steps is not None:
        train_cfg.max_training_steps = args.max_training_steps
    if args.max_training_tokens is not None:
        train_cfg.max_training_tokens = args.max_training_tokens
    if args.pack_sequences is not None:
        train_cfg.pack_sequences = args.pack_sequences
    if args.effective_token_lr_scale is not None:
        train_cfg.effective_token_lr_scale = args.effective_token_lr_scale
    if args.effective_token_lr_exponent is not None:
        train_cfg.effective_token_lr_exponent = args.effective_token_lr_exponent
    if args.wandb_xaxis_tokens is not None:
        train_cfg.wandb_xaxis_tokens = args.wandb_xaxis_tokens
    if args.log_wandb is not None:
        train_cfg.log_wandb = args.log_wandb
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.train_dataset_path is not None:
        train_cfg.train_dataset_path = args.train_dataset_path
    if args.relevance_min_rating is not None:
        train_cfg.relevance_min_rating = args.relevance_min_rating
    if args.image_correspondence_min_rating is not None:
        train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
    if args.visual_dependency_min_rating is not None:
        train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
    if args.formatting_min_rating is not None:
        train_cfg.formatting_min_rating = args.formatting_min_rating

    if args.resume_from_checkpoint is not None:
        train_cfg.resume_from_vlm_checkpoint = False
        vlm_cfg.vlm_load_backbone_weights = False
    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        set_pg_cpu(create_cpu_group())

    set_seed(global_cfg)

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)
        print("--- Global Config ---")
        print(global_cfg)

    train(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()
