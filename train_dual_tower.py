import os
import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
from statistics import mean
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
import json
import warnings
from pydantic import warnings as pydantic_warnings
import shutil
warnings.filterwarnings("ignore", category=pydantic_warnings.UnsupportedFieldAttributeWarning)

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PG_CPU = None

from configs.config_dual_tower import VLMConfig, TrainConfig, GlobalConfig
from models.dual_tower.dual_tower import DualTowerVLM
from data.data_utils import synchronized_dataloader_step
from data.processors import get_image_processor, get_tokenizer
from data.datasets import VQADualDataset
from data.collators import VQADualCollator
from evaluation.cider_utils import COCOGenerationDataset, GenerationCollator, compute_cider_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(obj):
    """Gather any picklable object from every rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)
    return result

def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

def get_run_name(train_cfg, vlm_cfg):
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

def get_dataloaders(train_cfg, vlm_cfg, global_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    full_dataset = load_dataset(train_cfg.direct_train_dataset_path)
    train_ds = full_dataset["train"].shuffle(seed=global_cfg.seed)
    val_ds = full_dataset["validation"]

    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    train_dataset = VQADualDataset(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = VQADualDataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)

    collator = VQADualCollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(global_cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False
    ) if is_dist() else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("Warming up dataloaders...")   
    next(iter(train_loader))
    next(iter(val_loader))
    print("Warmup complete.")

    # Prepare CIDEr dataloader (only on master to save memory/time)
    cider_loader = None
    if is_master():
        print("Loading COCO validation for CIDEr evaluation...")
        cider_dataset_hf = load_dataset("patrickamadeus/coco_caption_val_unique", split="train")
        cider_dataset = COCOGenerationDataset(
            cider_dataset_hf,
            tokenizer,
            image_processor,
            prompt="Describe the image.",
            mp_image_token_length=vlm_cfg.mp_image_token_length,
            total_samples=5000  # Limit for evaluation speed
        )
        cider_collator = GenerationCollator(tokenizer, max_length=2048)
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
            last_img_idx = batch["last_img_idx"]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(
                    input_ids=input_ids, 
                    images=images, 
                    attention_mask=attention_mask, 
                    targets=labels,
                    last_img_idx=last_img_idx
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
        avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist_flag else avg_val_loss
        
        if is_dist_flag:
            min_batch_loss = min(dist_gather(min_batch_loss))
            max_batch_loss = max(dist_gather(max_batch_loss))
        
        cider_score = None
        if compute_cider and tokenizer is not None and is_master() and cider_loader is not None:
            if is_master():
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
            if is_master():
                print(f"CIDEr Score: {cider_score:.4f}")
        
        if save_checkpoint and is_master():
            checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
            save_model = model.module if is_dist_flag else model
            save_model.save_pretrained(save_directory=checkpoint_path_step)
    
    return avg_val_loss, min_batch_loss, max_batch_loss, cider_score


def train(train_cfg, vlm_cfg, global_cfg):
    train_loader, val_loader, cider_loader, tokenizer = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    
    if train_cfg.log_wandb and is_master():
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        run = wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            config={
                "train_config": train_cfg.__dict__,
                "vlm_config": vlm_cfg.__dict__,
                "global_config": global_cfg.__dict__
            }
        )
    
    model = DualTowerVLM(
        vlm_cfg, 
        load_backbone=vlm_cfg.vlm_load_backbone_weights,
        freeze_left_vision=(train_cfg.lr_vision_backbone <= 0),
        freeze_left_projector=(train_cfg.lr_mp <= 0),
        freeze_left_decoder=(train_cfg.lr_language_backbone <= 0),
        freeze_right_decoder=(train_cfg.freeze_right_tower or train_cfg.lr_right_tower <= 0),
    )
    
    if is_master():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print(f"DualTowerVLM initialized with {total_params:,} parameters") 
        print(f"  ├── Trainable: {trainable_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
        print(f"  └── Frozen:    {frozen_params:,} parameters ({100*frozen_params/total_params:.1f}%)")
        print()
        print(f"Training summary: {len(train_loader)} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}")

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
        model = torch.compile(model)
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_model_path = None
    global_step = 0
    epoch = 0

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
    if is_master():
        print("\n" + "="*50)
        print("Computing baseline validation loss (step 0)...")
        print("="*50)
    
    baseline_val_loss, baseline_min_loss, baseline_max_loss, baseline_cider = evaluate_validation(
        model, val_loader, device, is_dist_flag=is_dist(),
        save_checkpoint=False, vlm_cfg=vlm_cfg, run_name=run_name,
        global_step=0, train_cfg=train_cfg, compute_cider=True, cider_loader=cider_loader, tokenizer=tokenizer
    )
    
    if is_master():
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

            fw_bw_start = time.time()
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

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration

            running_loss = total_train_loss / (i + 1)
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Running Loss': f'{running_loss:.4f}',
                'Tokens/s': f'{tokens_per_second:.1f}',
                'Step': global_step
            })

            # Validation
            if global_step % train_cfg.eval_every_n_steps == 0 and is_update_step and global_step > 0:
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    avg_val_loss, min_val_loss, max_val_loss, cider_score = evaluate_validation(
                        model, val_loader, device, is_dist_flag=is_dist(), 
                        save_checkpoint=False, vlm_cfg=vlm_cfg, run_name=run_name, 
                        global_step=global_step, train_cfg=train_cfg, compute_cider=True, cider_loader=cider_loader, tokenizer=tokenizer
                    )
                    
                    if is_master():
                        print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
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

            # Save checkpoint
            if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0 and is_master():
                save_model = model.module if is_dist() else model
                checkpoint_dir = f"{train_cfg.local_model_cp_path}-{global_step}"
                
                if train_cfg.save_local:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict': save_model.state_dict(),
                        'config': vlm_cfg.__dict__,
                        'global_step': global_step,
                    }, os.path.join(checkpoint_dir, 'dual_tower_model.pt'))
                    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                        json.dump(vlm_cfg.__dict__, f, indent=2)
                    print(f"DualTowerVLM saved locally to {checkpoint_dir}")
                
                if train_cfg.save_hf:
                    from huggingface_hub import HfApi, create_repo
                    api = HfApi()
                    repo_id = f"{train_cfg.hf_model_cp_path}-{global_step}"
                    
                    try:
                        create_repo(repo_id, exist_ok=True)
                    except Exception as e:
                        print(f"Repo creation note: {e}")
                    
                    temp_dir = f"/tmp/dual_tower_checkpoint_{global_step}"
                    os.makedirs(temp_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict': save_model.state_dict(),
                        'config': vlm_cfg.__dict__,
                        'global_step': global_step,
                    }, os.path.join(temp_dir, 'dual_tower_model.pt'))
                    with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
                        json.dump(vlm_cfg.__dict__, f, indent=2)
                    
                    api.upload_folder(
                        folder_path=temp_dir,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    print(f"DualTowerVLM pushed to HuggingFace Hub: {repo_id}")
                    shutil.rmtree(temp_dir)

            if is_update_step:
                if is_dist():
                    batch_loss_gathered = dist_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss
                
                if train_cfg.log_wandb and is_master():
                    run.log({"train/batch_loss": batch_loss_gathered}, step=global_step)
            
            if is_update_step:
                global_step += 1
                if global_step >= max_steps:
                    break

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            print(f"Epoch {epoch} completed. Avg Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.1f}s, Tokens/s: {epoch_tokens_per_second:.1f}")
            if train_cfg.log_wandb:
                run.log({
                    "epoch_loss": avg_train_loss,
                    "epoch_duration": epoch_duration,
                    "epoch_tokens_per_second": epoch_tokens_per_second
                })
    
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed if total_samples_processed > 0 else 0
        
        print(f"\nTraining complete!")
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")
        print(f"Total steps: {global_step}")
        
        if train_cfg.save_hf and best_model_path:
            print(f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub...")
            from huggingface_hub import HfApi, create_repo
            api = HfApi()
            repo_id = f"{train_cfg.hf_model_cp_path}-best"
            
            # Create repo if it doesn't exist
            try:
                create_repo(repo_id, exist_ok=True)
            except Exception as e:
                print(f"Repo creation note: {e}")
            
            # Upload the best model checkpoint folder to hub
            api.upload_folder(
                folder_path=best_model_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"DualTowerVLM best model pushed to HuggingFace Hub: {repo_id}")

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()


def main(config_name: str = "config_twin_tower"):
    global PG_CPU
    global_cfg = GlobalConfig()
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    
    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home
        print(f"Changed HF_HOME to {global_cfg.hf_home}")
    
    print(f"Starting Dual Tower VLM training with config: {config_name}")
    print(f"VLM Config: {vlm_cfg}")
    print(f"Train Config: {train_cfg}")
    print(f"Global Config: {global_cfg}")
    
    # Set seed
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    numpy.random.seed(global_cfg.seed)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dual Tower VLM model')
    parser.add_argument('--config', type=str, default='config_twin_tower',
                       help='Config name from configs/ folder')
    args = parser.parse_args()
    main(args.config)

