import os
import math
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import contextlib
import warnings
from pydantic import warnings as pydantic_warnings
warnings.filterwarnings("ignore", category=pydantic_warnings.UnsupportedFieldAttributeWarning)

# dataloader and distributed training utils
from train_utils.utils import (
    set_seed, init_dist, destroy_dist, is_dist, is_master,
    get_lr, wrap_model,
    compute_gradient_stats, compute_batch_stats, gather_batch_loss,
    compute_epoch_stats, save_model_checkpoint, evaluate_validation_dual_tower
)
from train_utils.dataloader import get_dataloaders
from data.data_utils import synchronized_dataloader_step

# models and configs
from models.dual_tower.dual_tower import DualTowerVLM
from configs.config_dual_tower import VLMConfig, TrainConfig, GlobalConfig

# logging utils
from train_utils.logging import (
    init_wandb, log_code_to_wandb, log_baseline_validation, log_training_step,
    log_validation_step, log_epoch_stats, finish_wandb,
    print_baseline_validation, print_validation_step, print_epoch_stats
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/huggingface/"


def get_run_name_dual_tower(train_cfg, vlm_cfg):
    """Custom run name for dual tower that includes right tower info."""
    from train_utils.utils import get_world_size
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    right_tower_lr = f"right_{train_cfg.lr_right_tower}" if not train_cfg.freeze_right_tower else "right_frozen"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-mp_{train_cfg.lr_mp}-{right_tower_lr}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"DualTower_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"


def train(train_cfg, vlm_cfg, global_cfg):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load train, val, and generation dataloaders
    train_loader, val_loader, gen_loader, tokenizer = get_dataloaders(train_cfg, vlm_cfg, global_cfg, is_dual_tower=True)

    # detect rank and initiate workers
    if is_dist():
        print(f"Rank {torch.distributed.get_rank()}: Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        torch.distributed.barrier()
        if is_master():
            print("All workers have gotten dataloaders.")

    # init run_name & detect data size & start wandb run
    run_name = get_run_name_dual_tower(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    run = init_wandb(train_cfg, vlm_cfg, global_cfg, run_name, total_dataset_size)
    
    # log code and config to wandb artifacts before training starts
    if is_master():
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dual_tower.py")
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config_dual_tower.py")
        log_code_to_wandb(run, train_cfg, script_path, config_path)
    
    # load / initialize model
    model = DualTowerVLM(
        vlm_cfg, 
        load_backbone=vlm_cfg.vlm_load_backbone_weights,
        freeze_left_vision=(train_cfg.lr_vision_backbone <= 0),
        freeze_left_projector=(train_cfg.lr_mp <= 0),
        freeze_left_decoder=(train_cfg.lr_language_backbone <= 0),
        freeze_right_decoder=(train_cfg.freeze_right_tower or train_cfg.lr_right_tower <= 0),
    )
    
    # count total params and log
    if is_master():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print(f"DualTowerVLM initialized with {total_params:,} parameters") 
        print(f"  ├── Trainable: {trainable_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
        print(f"  └── Frozen:    {frozen_params:,} parameters ({100*frozen_params/total_params:.1f}%)")
        print()
        from train_utils.utils import get_world_size
        print(f"Training summary: {len(train_loader)} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}")

    # setup optimizers and param groups
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.left_tower.MP.parameters()), 'lr': train_cfg.lr_mp})
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.left_tower.vision_encoder.parameters()), 'lr': train_cfg.lr_vision_backbone})
    if train_cfg.lr_language_backbone > 0:
        left_tower_params = list(model.left_tower.decoder.parameters())
        param_groups.append({'params': left_tower_params, 'lr': train_cfg.lr_language_backbone})
    
    if not train_cfg.freeze_right_tower and train_cfg.lr_right_tower > 0:
        param_groups.append({'params': list(model.right_tower.parameters()), 'lr': train_cfg.lr_right_tower})

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    # init global step and epoch
    global_step, epoch = 0, 0

    # calculate max steps for training, either in epoch vs step mode
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

    # compute baseline validation loss & metric score
    baseline_val_loss, baseline_min_loss, baseline_max_loss, baseline_metric_score = evaluate_validation_dual_tower(
        model, val_loader, gen_loader, device, train_cfg, global_cfg,
        run_name=run_name, global_step=0, tokenizer=tokenizer, metric=train_cfg.eval_metric
    )
    
    # print and log baseline validation results
    print_baseline_validation(baseline_val_loss, baseline_min_loss, baseline_max_loss)
    log_baseline_validation(run, train_cfg, baseline_val_loss, baseline_min_loss, 
                           baseline_max_loss, baseline_metric_score, global_step=0)

    # t(second) per epoch for statistics only
    epoch_times = []

    while global_step < max_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()

        print(f"Starting epoch {epoch}")
        train_pbar = tqdm(
            enumerate(synchronized_dataloader_step(train_loader, is_dist())),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for i, batch in train_pbar:
            # Skip empty batches
            if len(batch["images"]) == 0:
                continue
                
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            batch_start_time = time.time()
            
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            last_img_idx = batch["last_img_idx"]
            
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss = model(
                        input_ids=input_ids, 
                        images=images, 
                        attention_mask=attention_mask, 
                        targets=labels,
                        last_img_idx=last_img_idx
                    )

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps
            
            loss.backward()

            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    # Compute raw gradient norm BEFORE clipping
                    raw_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None]), 2)
                    # Clip gradients (this modifies gradients in-place and returns the pre-clip norm)
                    total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
                    # Compute gradient norm AFTER clipping
                    clipped_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None]), 2)
                    
                    if train_cfg.log_wandb and is_master():
                        run.log({
                            "train/raw_grad_norm": raw_grad_norm.item(),
                            "train/total_grad_norm": total_norm.item(),  # Pre-clip norm from PyTorch
                            "train/clipped_grad_norm": clipped_grad_norm.item()  # Post-clip norm
                        }, step=global_step)

                # Update learning rates
                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, max_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(global_step, train_cfg.lr_vision_backbone, max_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    param_group_idx += 1
                
                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(global_step, train_cfg.lr_language_backbone, max_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone
                    param_group_idx += 1
                
                if not train_cfg.freeze_right_tower and train_cfg.lr_right_tower > 0:
                    adj_lr_right_tower = get_lr(global_step, train_cfg.lr_right_tower, max_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_right_tower

                optimizer.step()
                optimizer.zero_grad()

            # compute per batch stats / step loss and tokens
            batch_end_time = time.time()
            batch_stats = compute_batch_stats(
                loss, attention_mask, train_cfg.gradient_accumulation_steps, 
                batch_start_time, batch_end_time
            )
            batch_loss = batch_stats["batch_loss"]
            total_train_loss += batch_loss
            total_tokens_processed += batch_stats["num_tokens"]

            # print progress bar if master
            if is_master():
                train_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Step": global_step})
            
            # update weights if update step
            if is_update_step:
                # compute gradient stats
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
                if not train_cfg.freeze_right_tower and train_cfg.lr_right_tower > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_right_tower, max_steps)
                
                # update weights
                optimizer.step()
                # clear excess gradients from prev. steps
                optimizer.zero_grad()
                
                # gather batch loss and log
                batch_loss_gathered = gather_batch_loss(batch_loss)
                log_training_step(run, train_cfg, batch_loss_gathered, gradient_stats, global_step)
                
                global_step += 1
                
                # validation every n steps + run objective scoring via eval. metrics
                if global_step % train_cfg.eval_every_n_steps == 0 and global_step > 0:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # switch model to eval mode for validation
                    model.eval()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    # evaluate validation loss & metric score
                    avg_val_loss, min_val_loss, max_val_loss, metric_score = evaluate_validation_dual_tower(
                        model, val_loader, gen_loader, device, train_cfg, global_cfg,
                        run_name=run_name, global_step=global_step, tokenizer=tokenizer, metric=train_cfg.eval_metric,
                    )
                    
                    # print and log validation stats and results
                    print_validation_step(global_step, avg_val_loss)
                    log_validation_step(run, train_cfg, avg_val_loss, min_val_loss, max_val_loss, 
                                       metric_score, global_step)
                    # switch model back to train mode for next training step
                    model.train()

                if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0:
                    save_model_checkpoint(model, train_cfg, global_step=global_step)
                
                # end training if max steps reached
                if global_step >= max_steps:
                    break

        # print and log per epoch stats
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
    
    # final save model checkpoint and finish wandb run
    save_model_checkpoint(model, train_cfg, is_final=True)
    finish_wandb(run, train_cfg, epoch_times)


def main():
    # init configs
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    # set hf_home if provided
    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home
        print(f"HF_HOME set to {global_cfg.hf_home}")

    # init distributed training if applicable
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    # print training start if master
    if is_master():
        print(f"Starting training on dual tower model")
    
    # set seed and train
    set_seed(global_cfg)
    train(train_cfg, vlm_cfg, global_cfg)
    
    # destroy distributed training if applicable
    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()

