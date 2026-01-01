import os
import sys
import math
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from data.data_utils import synchronized_dataloader_step
from models.vision_language_model import VisionLanguageModel
from configs.config_vanilla import VLMConfig, TrainConfig, GlobalConfig
from train_utils.utils import (
    set_seed, init_dist, destroy_dist, is_dist, is_master,
    get_run_name, get_lr, setup_param_groups, wrap_model,
    compute_gradient_stats, compute_batch_stats, gather_batch_loss,
    compute_epoch_stats, save_model_checkpoint, evaluate_validation
)
from train_utils.dataloader import get_dataloaders
from train_utils.logging import (
    init_wandb, log_baseline_validation, log_training_step,
    log_validation_step, log_epoch_stats, finish_wandb,
    print_baseline_validation, print_validation_step, print_epoch_stats
)
import contextlib


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/huggingface/"


def train(train_cfg, vlm_cfg, global_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, gen_loader, tokenizer = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        print(f"Rank {torch.distributed.get_rank()}: Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        torch.distributed.barrier()
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    run = init_wandb(train_cfg, vlm_cfg, global_cfg, run_name, total_dataset_size)

    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    param_groups, total_params, trainable_params = setup_param_groups(model, train_cfg)
    if is_master():
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = optim.AdamW(param_groups)
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    global_step, epoch = 0, 0
    
    if train_cfg.use_epochs:
        batches_per_epoch = len(train_loader)
        updates_per_epoch = math.ceil(batches_per_epoch / train_cfg.gradient_accumulation_steps)
        max_steps = train_cfg.max_epochs * updates_per_epoch
        if is_master():
            print(f"Training for {train_cfg.max_epochs} epochs = {max_steps} optimizer updates ({batches_per_epoch} batches/epoch, grad_accum={train_cfg.gradient_accumulation_steps})")
    else:
        max_steps = train_cfg.max_training_steps
        if is_master():
            print(f"Training for {max_steps} optimizer updates")

    baseline_val_loss, baseline_min_loss, baseline_max_loss, baseline_metric_score = evaluate_validation(
        model, val_loader, gen_loader, device, train_cfg, global_cfg,
        run_name=run_name, global_step=0, tokenizer=tokenizer, metric=train_cfg.metric
    )
    
    print_baseline_validation(baseline_val_loss, baseline_min_loss, baseline_max_loss)
    log_baseline_validation(run, train_cfg, baseline_val_loss, baseline_min_loss, 
                           baseline_max_loss, baseline_metric_score, global_step=0)

    epoch_times = []

    while global_step < max_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss, total_tokens_processed = 0, 0
        optimizer.zero_grad()
        if is_master():
            print(f"Starting training epoch {epoch}")

        train_pbar = tqdm(
            enumerate(synchronized_dataloader_step(train_loader, is_dist())),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False,
            disable=not is_master()
        )

        for i, batch in train_pbar:
            if len(batch["images"]) == 0:
                continue

            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            batch_start_time = time.time()

            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Context manager for DDP gradient synchronization
            if is_dist() and train_cfg.gradient_accumulation_steps > 1 and not is_update_step:
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16
            )

            with context:
                with autocast_context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss /= train_cfg.gradient_accumulation_steps

            loss.backward()

            batch_end_time = time.time()
            batch_stats = compute_batch_stats(
                loss, attention_mask, train_cfg.gradient_accumulation_steps, 
                batch_start_time, batch_end_time
            )
            batch_loss = batch_stats["batch_loss"]
            total_train_loss += batch_loss
            total_tokens_processed += batch_stats["num_tokens"]

            if is_master():
                train_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Step": global_step})
            
            if is_update_step:
                gradient_stats = compute_gradient_stats(all_params, train_cfg.max_grad_norm)
                
                # LR updates
                idx = 0
                if train_cfg.lr_mp > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_mp, max_steps)
                    idx += 1
                if train_cfg.lr_vision_backbone > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_vision_backbone, max_steps)
                    idx += 1
                if train_cfg.lr_language_backbone > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_language_backbone, max_steps)
                    idx += 1
                
                optimizer.step()
                optimizer.zero_grad()
                
                batch_loss_gathered = gather_batch_loss(batch_loss)
                log_training_step(run, train_cfg, batch_loss_gathered, gradient_stats, global_step)
                
                global_step += 1
                
                # Validation
                if global_step % train_cfg.eval_every_n_steps == 0 and global_step > 0:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    avg_val_loss, min_val_loss, max_val_loss, metric_score = evaluate_validation(
                        model, val_loader, gen_loader, device, train_cfg, global_cfg,
                        run_name=run_name, global_step=global_step, tokenizer=tokenizer,
                        compute_metric=True, metric=train_cfg.metric
                    )
                    
                    print_validation_step(global_step, avg_val_loss)
                    log_validation_step(run, train_cfg, avg_val_loss, min_val_loss, max_val_loss, 
                                       metric_score, global_step)
                    model.train()

                if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0:
                    save_model_checkpoint(model, train_cfg, global_step=global_step)
                
                if global_step >= max_steps:
                    break

        epoch_end_time = time.time()
        epoch_stats = compute_epoch_stats(
            total_train_loss, total_tokens_processed, len(train_loader),
            epoch_start_time, epoch_end_time
        )
        epoch_times.append(epoch_stats["epoch_duration"])
        
        print_epoch_stats(epoch, epoch_stats["avg_loss"], epoch_stats["epoch_duration"], 
                         epoch_stats["epoch_tokens_per_second"])
        log_epoch_stats(run, train_cfg, epoch_stats["avg_loss"], epoch_stats["epoch_duration"], 
                       epoch_stats["epoch_tokens_per_second"])

    # Final save
    save_model_checkpoint(model, train_cfg, is_final=True)
    finish_wandb(run, train_cfg, epoch_times)


def main():
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home
        print(f"HF_HOME set to {global_cfg.hf_home}")

    if not global_cfg.same_dir_as_nanovlm_repo:
        sys.path.append(os.path.join(os.getcwd(), "nanoVLM"))

    # Initialize distributed training if applicable
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        print(f"Starting training on vanilla model")
    
    set_seed(global_cfg)
    train(train_cfg, vlm_cfg, global_cfg)
    
    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()