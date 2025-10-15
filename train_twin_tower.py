import os
import sys
import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk
from tqdm import tqdm
import argparse

# Suppress pydantic warnings about unsupported field attributes
import warnings
from pydantic import warnings as pydantic_warnings
warnings.filterwarnings("ignore", category=pydantic_warnings.UnsupportedFieldAttributeWarning)


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PG_CPU = None

from utils_temp.config_utils import VLMConfig, TrainConfig, GlobalConfig, load_config

# import sys, os
# nanovlm_root = os.path.join(os.getcwd(), "nanoVLM")
# if nanovlm_root not in sys.path:
#     sys.path.insert(0, nanovlm_root)

from models.twin_tower import TwinTowerModel
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer
from data.coco_caption import COCODataset, COCOCollator

#Otherwise, the tokenizer will throw a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
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
    # torch.cuda.manual_seed(0)           # seed *this* GPU only

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
    """
    Gather *any* picklable object from every rank without allocating
    temporary CUDA buffers.  Returns a list [rank0_obj, rank1_obj, …].

    Falls back to a single-rank list when torch.distributed is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)  # CPU path
    return result

def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)           # in‑place, returns None
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

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg, global_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    full_dataset = load_dataset(train_cfg.direct_train_dataset_path)
    train_ds = full_dataset["train"].shuffle(seed=global_cfg.seed)
    val_ds = full_dataset["validation"]
    # dataset_names_to_load = train_cfg.train_dataset_name
    # if "shards" in train_cfg.train_dataset_name:
    #     print("Loading shards")
    #     total_shards = 56
    #     dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    # if "all" in dataset_names_to_load:
    #     dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # # load and combine all training datasets
    # combined_train_data = []

    # for dataset_name in dataset_names_to_load:
    #     print(f"Loading dataset: {dataset_name}")
    #     if "shard_" in dataset_name:
    #         try:
    #             train_ds = load_from_disk(dataset_name)
    #             combined_train_data.append(train_ds)
    #             continue
    #         except Exception as e:
    #             print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
    #             continue
    #     try:
    #         train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)['train']
    #         train_ds[0] # Check if the dataset is loaded correctly
    #         combined_train_data.append(train_ds)
    #     except Exception as e:
    #         if is_master():
    #             print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
    #         continue

    # if not combined_train_data:
    #     raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")
    
    # train_ds = concatenate_datasets(combined_train_data)
    # # apply cutoff if specified
    # if train_cfg.data_cutoff_idx is None:
    #     total_samples = len(train_ds)  # Use the entire dataset
    # else:
    #     total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    train_ds = train_ds.shuffle(seed=global_cfg.seed) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  

    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # val_size = int(len(train_ds) * train_cfg.val_ratio)
    # print(f"Val size: {val_size}")

    # val_ds = train_ds.select(range(val_size))
    # train_ds = train_ds.select(range(val_size, len(train_ds)))

    train_dataset = COCODataset(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = COCODataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)

    train_dataset = ConstantLengthDataset(train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=8,
                                        max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    # create collator
    collator = COCOCollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(global_cfg.seed)

    # TODO: determine the proper setting for DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        collate_fn=collator,
        # num_workers=4,
        # pin_memory=True,
        # persistent_workers=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # TODO: determine if this should be true or false
    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False
    )

    # TODO: determine the proper setting for DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=collator,
        # num_workers=2,
        # pin_memory=True,
        # persistent_workers=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    print("Warming up dataloaders...")   
    next(iter(train_loader))
    next(iter(val_loader))
    print("Warmup complete.")

    return train_loader, val_loader


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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def evaluate_validation(model, val_loader, device, is_dist=False, save_checkpoint=False, 
                        vlm_cfg=None, run_name=None, global_step=None, train_cfg=None):
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        total_val_loss = 0
        val_batches = 0
        min_batch_loss = float('inf')
        max_batch_loss = float('-inf')
        
        val_pbar = tqdm(
            synchronized_dataloader_step(val_loader, is_dist),
            total=len(val_loader),
            desc="Validation",
            leave=False
        )
        
        for batch in val_pbar:
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            
            batch_loss = loss.item()
            min_batch_loss = min(min_batch_loss, batch_loss)
            max_batch_loss = max(max_batch_loss, batch_loss)
            total_val_loss += batch_loss
            val_batches += 1
            
            # Update progress bar with running validation loss
            current_val_loss = total_val_loss / val_batches
            val_pbar.set_postfix({
                'Val Loss': f'{current_val_loss:.4f}',
                'Batch': f'{val_batches}/64'
            })
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist else avg_val_loss
        
        if is_dist:
            min_batch_loss = min(dist_gather(min_batch_loss))
            max_batch_loss = max(dist_gather(max_batch_loss))
        
        # Optional checkpoint saving
        if save_checkpoint and is_master():
            # Save a checkpoint for this evaluation step
            checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
            save_model = model.module if is_dist else model  # unwrap the model for saving if DDP
            save_model.save_pretrained(save_directory=checkpoint_path_step)

            # TODO: uncomment if wanna use lmms-eval
            # if train_cfg and train_cfg.use_lmms_eval and global_step % (train_cfg.eval_interval*2) == 0:
            #     # Submit evaluation job
            #     cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {train_cfg.lmms_eval_limit} {train_cfg.lmms_eval_tasks} {train_cfg.lmms_eval_batch_size}"
            #     print(f"Submitting evaluation job: {cmd}")
            #     subprocess.run(cmd, shell=True)
    
    return avg_val_loss, min_batch_loss, max_batch_loss


def train(train_cfg, vlm_cfg, global_cfg):
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    if train_cfg.log_wandb and is_master():
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            config={
                "train_config": train_cfg.__dict__,
                "vlm_config": vlm_cfg.__dict__,
                "global_config": global_cfg.__dict__
            }
        )

        # TODO: uncomment if wanna use lmms-eval
        # lmms_eval_step = "<lmms-eval-step>"
        # run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)
    
    # initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = TwinTowerModel.from_pretrained(vlm_cfg)
    else:
        model = TwinTowerModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    if is_master():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print(f"nanoVLM initialized with {total_params:,} parameters") 
        print(f"  ├── Trainable: {trainable_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
        print(f"  └── Frozen:    {frozen_params:,} parameters ({100*frozen_params/total_params:.1f}%)")
        print()
        print(f"Training summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.left_tower.MP.parameters()), 'lr': train_cfg.lr_mp})
    else:
        for p in list(model.left_tower.MP.parameters()):
            p.requires_grad = False
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.left_tower.vision_encoder.parameters()), 'lr': train_cfg.lr_vision_backbone})
    else:
        for p in list(model.left_tower.vision_encoder.parameters()):
            p.requires_grad = False
    if train_cfg.lr_language_backbone > 0:
        # Only include left tower decoder for language backbone training
        left_tower_params = list(model.left_tower.decoder.parameters())
        param_groups.append({'params': left_tower_params, 'lr': train_cfg.lr_language_backbone})
    else:
        for p in list(model.left_tower.decoder.parameters()):
            p.requires_grad = False
    
    # Handle right tower freezing based on config
    if train_cfg.freeze_right_tower or train_cfg.lr_right_tower <= 0:
        for p in list(model.right_tower.parameters()):
            p.requires_grad = False
    else:
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
    best_val_loss = float('inf')
    best_model_path = None
    logged_eval_steps = set()
    global_step = 0
    epoch = 0

    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
    }

    if train_cfg.use_epochs:
        max_steps = train_cfg.max_epochs * len(train_loader)
    else:
        max_steps = train_cfg.max_training_steps

    while global_step < max_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        print("Starting training loop")
        train_pbar = tqdm(
            enumerate(synchronized_dataloader_step(train_loader, is_dist())),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for i, batch in train_pbar:
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
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
                if train_cfg.max_grad_norm is not None:
                    raw_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params if p.grad is not None]), 2)
                    if train_cfg.log_wandb and is_master():
                        run.log({"train/raw_grad_norm": raw_grad_norm.item()}, step=global_step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
                    if train_cfg.log_wandb and is_master():
                        run.log({"train/clipped_grad_norm": grad_norm}, step=global_step)

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

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration  # Multiply by world size to get global tokens/s

            # Update progress bar with running loss and metrics
            running_loss = total_train_loss / (i + 1)
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Running Loss': f'{running_loss:.4f}',
                'Tokens/s': f'{tokens_per_second:.1f}',
                'Step': global_step
            })

            # # Accumulate training stats TODO: future comprehensive logging
            # accumulated_stats['tokens_per_second'].append(tokens_per_second)
            # accumulated_stats['data_load_time'].append(data_load_time)
            # accumulated_stats['fw_bw_time'].append(fw_bw_time)
            # accumulated_stats['post_process_time'].append(post_process_time)
            # accumulated_stats['images_per_sample'].extend(images_per_sample)

            if global_step % train_cfg.eval_every_n_steps == 0 and is_update_step and global_step > 0:
                model.eval()
                if device == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    avg_val_loss, min_val_loss, max_val_loss = evaluate_validation(model, val_loader, device, is_dist=is_dist(), save_checkpoint=False, 
                                                      vlm_cfg=vlm_cfg, run_name=run_name, global_step=global_step, train_cfg=train_cfg)

                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     if is_master():
                    #         best_model_path = os.path.join(vlm_cfg.vlm_checkpoint_path, f"best_model_step_{global_step}")
                    #         save_model = model.module if is_dist() else model
                    #         save_model.save_pretrained(save_directory=best_model_path)
                    #         print(f"New best model saved at step {global_step} with val loss: {avg_val_loss:.4f}")
                    
                    if is_master():
                        print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                        if train_cfg.log_wandb:
                            run.log({"val/val_loss": avg_val_loss, "val/min_val_loss": min_val_loss, "val/max_val_loss": max_val_loss}, step=global_step)

                model.train()
            

            # Save model checkpoint
            if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0 and is_master():
                if train_cfg.save_local:
                    model.save_pretrained(save_directory=f"{train_cfg.local_model_cp_path}-{global_step}")
                    print(f"Model saved locally to {train_cfg.local_model_cp_path}-{global_step}")
                
                if train_cfg.save_hf:
                    model.push_to_hub(f"{train_cfg.hf_model_cp_path}-{global_step}")
                    print(f"Model pushed to HuggingFace Hub: {train_cfg.hf_model_cp_path}-{global_step}")

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
            data_load_start = time.time()

        avg_train_loss = total_train_loss / len(train_loader)
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
    
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")
        
        # Push the best model to the hub
        if train_cfg.save_hf and best_model_path:
            print(f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub...")
            hf_model = TwinTowerModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(f"{train_cfg.hf_model_cp_path}-best")
            print(f"Best model pushed to HuggingFace Hub: {train_cfg.hf_model_cp_path}-best")

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()
    

def main(config_name: str = "config_twin_tower"):
    """Main function loading config from YAML
    
    Args:
        config_name: Name of the config file in configs/ folder (without .yaml extension)
    """
    global PG_CPU
    global_cfg, vlm_cfg, train_cfg = load_config(config_name)
    
    if global_cfg.hf_home:
        os.environ["HF_HOME"] = global_cfg.hf_home
        print(f"Changed HF_HOME to {global_cfg.hf_home}")
    
    # if not global_cfg.same_dir_as_nanovlm_repo:
    #     sys.path.insert(0, os.path.join(os.getcwd(), 'nanoVLM'))
    
    print(f"Starting training with config: {config_name}")
    print(f"VLM Config: {vlm_cfg}")
    print(f"Train Config: {train_cfg}")
    print(f"Global Config: {global_cfg}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    
    # Set seed
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    numpy.random.seed(global_cfg.seed)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")   # host‑RAM, zero GPU allocations

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg, global_cfg)

    if is_dist():
        destroy_dist()

def mock_main():
    global PG_CPU
    config_name = "config_twin_tower"
    global_cfg, vlm_cfg, train_cfg = load_config(config_name)
    print("Successfully ran mock main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Twin Tower VLM model')
    parser.add_argument('--config', type=str, default='config_twin_tower',
                       help='Config name from configs/ folder (without .yaml extension)')
    args = parser.parse_args()
    main(args.config)
    
    # mock_main()
