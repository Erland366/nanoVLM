import os
import sys
import math
import time
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
import random
import numpy as np
from tqdm import tqdm
from statistics import mean
import json
import shutil

from data.coco_captions import COCODataset, COCOCollator
from data.data_utils import synchronized_dataloader_step
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from configs.config_vanilla import VLMConfig, TrainConfig, GlobalConfig
from evaluation.cider_utils import VanillaCOCOGenerationDataset, VanillaGenerationCollator, compute_cider_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/huggingface/"

# --------------------------
# Utility Functions
# --------------------------

def set_seed(global_cfg):
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    np.random.seed(global_cfg.seed)

def get_world_size():
    return 1

def get_run_name(train_cfg, vlm_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-mp_{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"Vanilla_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg, global_cfg):
    """Loads COCO Captions dataset only."""
    print(f"Loading COCO captions dataset: {train_cfg.direct_train_dataset_path}")
    full_dataset = load_dataset(train_cfg.direct_train_dataset_path)
    train_ds = full_dataset["train"].shuffle(seed=global_cfg.seed)
    val_ds = full_dataset["validation"]

    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len
    )
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    train_dataset = COCODataset(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = COCODataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    collator = COCOCollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(global_cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        generator=g
    )

    # Prepare CIDEr dataloader
    cider_loader = None
    print("Loading COCO validation for CIDEr evaluation...")
    cider_dataset_hf = load_dataset("patrickamadeus/coco_caption_val_unique", split="train")
    cider_dataset = VanillaCOCOGenerationDataset(
        cider_dataset_hf,
        tokenizer,
        image_processor,
        prompt="Describe the image.",
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        total_samples=5000  # Limit for evaluation speed
    )
    cider_collator = VanillaGenerationCollator(tokenizer, max_length=2048)
    cider_loader = DataLoader(
        cider_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=cider_collator,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, cider_loader, tokenizer


def get_lr(it, max_lr, max_steps):
    """Cosine LR schedule with warmup."""
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_validation(model, val_loader, device, is_dist_flag=False, save_checkpoint=False, 
                        vlm_cfg=None, run_name=None, global_step=None, train_cfg=None, compute_cider=False, cider_loader=None, tokenizer=None):
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        total_val_loss = 0
        val_batches = 0
        min_batch_loss = float('inf')
        max_batch_loss = float('-inf')
        
        val_pbar = tqdm(
            synchronized_dataloader_step(val_loader, is_dist_flag),
            total=len(val_loader),
            desc="Validation",
            leave=False
        )
        
        for batch in val_pbar:
            # Skip empty batches
            if len(batch["images"]) == 0:
                continue
                
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(
                    input_ids=input_ids, 
                    images=images, 
                    attention_mask=None, 
                    targets=labels
                )
            
            batch_loss = loss.item()
            min_batch_loss = min(min_batch_loss, batch_loss)
            max_batch_loss = max(max_batch_loss, batch_loss)
            total_val_loss += batch_loss
            val_batches += 1
            
            current_val_loss = total_val_loss / val_batches
            val_pbar.set_postfix({
                'Val Loss': f'{current_val_loss:.4f}',
                'Batch': f'{val_batches}'
            })
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        # avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist_flag else avg_val_loss
        
        # if is_dist_flag:
        #     min_batch_loss = min(dist_gather(min_batch_loss))
        #     max_batch_loss = max(dist_gather(max_batch_loss))
        
        cider_score = None
        if compute_cider and tokenizer is not None and cider_loader is not None:
            print("\nComputing CIDEr score on COCO validation set...")
            
            log_samples_path = None
            if run_name and global_step is not None and train_cfg is not None:
                timestamp = run_name.split('_')[-1]
                base_name = train_cfg.hf_model_cp_path.replace("/", "_") + "_" + timestamp
                os.makedirs("eval_samples", exist_ok=True)
                log_samples_path = os.path.join("eval_samples", base_name, f"samples_step_{global_step}.jsonl")
            
            cider_score = compute_cider_score(
                model, cider_loader, device, tokenizer,
                max_new_tokens=30,
                max_samples=5000,
                log_samples_path=log_samples_path
            )
            print(f"CIDEr Score: {cider_score:.4f}")
        
        if save_checkpoint:
            checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
            save_model = model # .module if is_dist_flag else model
            save_model.save_pretrained(save_directory=checkpoint_path_step)
    
    return avg_val_loss, min_batch_loss, max_batch_loss, cider_score


# --------------------------
# Main Training Loop
# --------------------------

def train(train_cfg, vlm_cfg, global_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, cider_loader, tokenizer = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)

    if train_cfg.log_wandb:
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        run = wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            config={
                "train_config": train_cfg.__dict__,
                "vlm_config": vlm_cfg.__dict__,
                "global_config": global_cfg.__dict__,
            },
        )

    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    # Set up optimizers
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({"params": list(model.MP.parameters()), "lr": train_cfg.lr_mp})
    else:
        for p in model.MP.parameters():
            p.requires_grad = False

    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({"params": list(model.vision_encoder.parameters()), "lr": train_cfg.lr_vision_backbone})
    else:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False

    if train_cfg.lr_language_backbone > 0:
        param_groups.append({"params": list(model.decoder.parameters()), "lr": train_cfg.lr_language_backbone})
    else:
        for p in model.decoder.parameters():
            p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = optim.AdamW(param_groups)
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)

    global_step, epoch = 0, 0
    
    if train_cfg.use_epochs:
        # Calculate number of optimizer updates (not batches) per epoch
        batches_per_epoch = len(train_loader)
        updates_per_epoch = math.ceil(batches_per_epoch / train_cfg.gradient_accumulation_steps)
        max_steps = train_cfg.max_epochs * updates_per_epoch
        print(f"Training for {train_cfg.max_epochs} epochs = {max_steps} optimizer updates ({batches_per_epoch} batches/epoch, grad_accum={train_cfg.gradient_accumulation_steps})")
    else:
        max_steps = train_cfg.max_training_steps
        print(f"Training for {max_steps} optimizer updates")

    # Compute baseline validation loss at step 0 before training
    print("\n" + "="*50)
    print("Computing baseline validation loss (step 0)...")
    print("="*50)
    
    baseline_val_loss, baseline_min_loss, baseline_max_loss, baseline_cider = evaluate_validation(
        model, val_loader, device, is_dist_flag=False,
        save_checkpoint=False, vlm_cfg=vlm_cfg, run_name=run_name,
        global_step=0, train_cfg=train_cfg, compute_cider=True, cider_loader=cider_loader, tokenizer=tokenizer
    )
    
    print(f"\nBaseline Validation Loss (step 0): {baseline_val_loss:.4f}")
    print(f"  ├── Min batch loss: {baseline_min_loss:.4f}")
    print(f"  └── Max batch loss: {baseline_max_loss:.4f}")
    print("="*50 + "\n")
    
    if train_cfg.log_wandb:
        log_dict = {
            "val/val_loss": baseline_val_loss,
            "val/min_val_loss": baseline_min_loss,
            "val/max_val_loss": baseline_max_loss,
        }
        if baseline_cider is not None:
            log_dict["val/cider_score"] = baseline_cider
        run.log(log_dict, step=0)

    epoch_times = []

    while global_step < max_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss, total_tokens_processed = 0, 0
        optimizer.zero_grad()
        print(f"Starting training epoch {epoch}")

        train_pbar = tqdm(
            enumerate(synchronized_dataloader_step(train_loader, False)),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False,
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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Use attention_mask=None for training with left-padded inputs  
                # Left padding + attention_mask + causal mask causes NaN
                _, loss = model(input_ids, images, attention_mask=None, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss /= train_cfg.gradient_accumulation_steps

            loss.backward()

            if is_update_step:
                if train_cfg.max_grad_norm:
                    # Compute raw gradient norm BEFORE clipping
                    raw_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None]), 2)
                    # Clip gradients
                    total_norm = torch.nn.utils.clip_grad_norm_(all_params, train_cfg.max_grad_norm)
                    # Compute gradient norm AFTER clipping
                    clipped_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None]), 2)
                    
                    if train_cfg.log_wandb:
                        run.log({
                            "train/raw_grad_norm": raw_grad_norm.item(),
                            "train/total_grad_norm": total_norm.item(),
                            "train/clipped_grad_norm": clipped_grad_norm.item()
                        }, step=global_step)

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

            batch_loss = loss.item() * (train_cfg.gradient_accumulation_steps if train_cfg.gradient_accumulation_steps > 1 else 1)
            total_train_loss += batch_loss
            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration

            train_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Step": global_step})
            
            if is_update_step:
                if train_cfg.log_wandb:
                    run.log({"train/batch_loss": batch_loss}, step=global_step)
                
                global_step += 1
                
                # Validation
                if global_step % train_cfg.eval_every_n_steps == 0 and global_step > 0:
                    torch.cuda.empty_cache()
                    avg_val_loss, min_val_loss, max_val_loss, cider_score = evaluate_validation(
                        model, val_loader, device, is_dist_flag=False, 
                        save_checkpoint=False, vlm_cfg=vlm_cfg, run_name=run_name, 
                        global_step=global_step, train_cfg=train_cfg, compute_cider=True, cider_loader=cider_loader, tokenizer=tokenizer
                    )
                    print(f"\nStep {global_step} | Val Loss: {avg_val_loss:.4f}")
                    
                    if train_cfg.log_wandb:
                        log_dict = {
                            "val/val_loss": avg_val_loss, 
                            "val/min_val_loss": min_val_loss, 
                            "val/max_val_loss": max_val_loss
                        }
                        if cider_score is not None:
                            log_dict["val/cider_score"] = cider_score
                        run.log(log_dict, step=global_step)
                    
                    model.train()

                if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0:
                    if train_cfg.save_local:
                        model.save_pretrained(f"{train_cfg.local_model_cp_path}-{global_step}")
                    if train_cfg.save_hf:
                        model.push_to_hub(f"{train_cfg.hf_model_cp_path}-{global_step}")
                
                if global_step >= max_steps:
                    break

        avg_loss = total_train_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        print(f"Epoch {epoch} done | Train Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")
        
        if train_cfg.log_wandb:
            run.log({
                "epoch_loss": avg_loss,
                "epoch_duration": epoch_duration,
                "epoch_tokens_per_second": epoch_tokens_per_second
            })

    # Final save
    if train_cfg.save_local:
        model.save_pretrained(train_cfg.local_model_cp_path)
    if train_cfg.save_hf:
        model.push_to_hub(train_cfg.hf_model_cp_path)
    if train_cfg.log_wandb:
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        run.summary["avg_epoch_time"] = avg_epoch_time
        wandb.finish()


def main():
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home
        print(f"HF_HOME set to {global_cfg.hf_home}")

    if not global_cfg.same_dir_as_nanovlm_repo:
        sys.path.append(os.path.join(os.getcwd(), "nanoVLM"))

    print(f"Starting COCO-caption training on vanilla model")
    set_seed(global_cfg)
    train(train_cfg, vlm_cfg, global_cfg)


if __name__ == "__main__":
    main()
