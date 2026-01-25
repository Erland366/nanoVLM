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

from configs.config_vanilla import VLMConfig, TrainConfig, GlobalConfig
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

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

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
        # Define a custom x-axis for lmms-eval metrics
        lmms_eval_step = "<lmms-eval-step>"
        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
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
    
    if train_cfg.compile:
        print("[DEBUG] Compiling repeated blocks with torch.compile (regional)...")
        compile_start = time.time()
        model._compile_dynamic = bool(train_cfg.compile_dynamic)
        model.compile_regional(
            backend=train_cfg.compile_backend,
            compile_dynamic=model._compile_dynamic,
        )
        print(f"[DEBUG] torch.compile setup took {time.time() - compile_start:.2f}s")
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    best_model_path = None
    logged_eval_steps = set()
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
    
    # Training stats accumulators
    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
    }
    
    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        if not train_cfg.stream_dataset:
            train_pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False,
                disable=not is_master(),
            )

        print("Starting training loop")
        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist())):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            data_load_time = time.time() - data_load_start

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
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                current_lrs = {}
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    current_lrs["train/lr_mp"] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    current_lrs["train/lr_vision_backbone"] = adj_lr_vision_backbone
                    param_group_idx += 1

                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone
                    current_lrs["train/lr_language_backbone"] = adj_lr_language_backbone
              
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
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
                        run.log({
                            "val_loss": avg_val_loss,
                            "val/min_val_loss": min_val_loss,
                            "val/max_val_loss": max_val_loss,
                        }, step=global_step)

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
                
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                    }, step=global_step)

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
                    run.log({
                        "batch_loss": batch_loss_gathered,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {}),
                        **current_lrs,
                    }, step=global_step)
                
            if is_update_step:
                global_step += 1
                if train_cfg.save_model_every_n_steps and global_step % train_cfg.save_model_every_n_steps == 0:
                    save_model_checkpoint(model, train_cfg, global_step=global_step)
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        if not train_cfg.stream_dataset and train_pbar is not None and is_master():
            train_pbar.close()
        avg_train_loss = total_train_loss / i
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
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second})

            print(f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

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
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
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
