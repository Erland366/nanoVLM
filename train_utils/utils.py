import torch
import random
import numpy as np
import time
import torch.distributed as dist
import os
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
import math
from tqdm import tqdm
from evaluation.coco_captions import compute_cider_score
from evaluation.ocr_vqa import compute_ocrvqa_accuracy
from data.data_utils import synchronized_dataloader_step


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

def dist_gather(obj, group=None):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=group)  # CPU path
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(global_cfg):
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    np.random.seed(global_cfg.seed)


def get_run_name(train_cfg, vlm_cfg):
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def setup_param_groups(model, train_cfg):
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
    
    return param_groups, total_params, trainable_params


def compute_gradient_stats(all_params, max_grad_norm):
    if max_grad_norm is None:
        return None
    
    # Get all parameters with gradients
    params_with_grad = [p for p in all_params if p.grad is not None]
    
    # If no parameters have gradients, return None or zero stats
    if len(params_with_grad) == 0:
        return {
            "raw_grad_norm": 0.0,
            "total_grad_norm": 0.0,
            "clipped_grad_norm": 0.0,
        }
    
    # Compute raw gradient norm BEFORE clipping
    raw_grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 
        2
    )
    
    # Clip gradients
    total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    
    # Compute gradient norm AFTER clipping
    clipped_grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 
        2
    )
    
    return {
        "raw_grad_norm": raw_grad_norm.item(),
        "total_grad_norm": total_norm.item(),
        "clipped_grad_norm": clipped_grad_norm.item(),
    }


def compute_batch_stats(loss, attention_mask, gradient_accumulation_steps, batch_start_time, batch_end_time):
    batch_loss = loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
    num_tokens = torch.sum(attention_mask).item()
    batch_duration = batch_end_time - batch_start_time
    tokens_per_second = num_tokens / batch_duration if batch_duration > 0 else 0
    
    return {
        "batch_loss": batch_loss,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_second,
        "batch_duration": batch_duration,
    }


def gather_batch_loss(batch_loss):
    if is_dist():
        return dist_mean_scalar(batch_loss)
    return batch_loss


def compute_epoch_stats(total_train_loss, total_tokens_processed, num_batches, 
                       epoch_start_time, epoch_end_time):
    avg_loss = total_train_loss / num_batches if num_batches > 0 else 0
    
    if is_dist():
        avg_loss = dist_mean_scalar(avg_loss)
        total_tokens_processed = sum(dist_gather(total_tokens_processed))
    
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_tokens_per_second = total_tokens_processed / epoch_duration if epoch_duration > 0 else 0
    
    return {
        "avg_loss": avg_loss,
        "total_tokens_processed": total_tokens_processed,
        "epoch_duration": epoch_duration,
        "epoch_tokens_per_second": epoch_tokens_per_second,
    }


def save_model_checkpoint(model, train_cfg, global_step=None, is_final=False):
    if not is_master():
        return
    
    save_model = model.module if is_dist() else model
    
    if is_final:
        local_path = train_cfg.local_model_cp_path
        hf_path = train_cfg.hf_model_cp_path
    else:
        local_path = f"{train_cfg.local_model_cp_path}-{global_step}"
        hf_path = f"{train_cfg.hf_model_cp_path}-{global_step}"
    
    if train_cfg.save_local:
        save_model.save_pretrained(local_path)
    if train_cfg.save_hf:
        save_model.push_to_hub(hf_path)


def evaluate_validation(model, val_loader, gen_loader, device, train_cfg, global_cfg, run_name=None, 
                        global_step=None, tokenizer=None, metric="cider"):
    # switch to eval mode
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # evaluate validation loss
    with torch.no_grad():
        total_val_loss = 0
        val_batches = 0
        min_batch_loss = float('inf')
        max_batch_loss = float('-inf')
        
        # tqdm progress bar
        val_pbar = tqdm(
            val_loader,
            total=min(len(val_loader), train_cfg.max_val_batches),
            desc="Validation",
            leave=False
        )
        
        for batch in val_pbar:
            if val_batches >= train_cfg.max_val_batches:
                break
                
            if len(batch["images"]) == 0:
                continue
                
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(
                    input_ids=input_ids, 
                    images=images, 
                    attention_mask=attention_mask, 
                    targets=labels
                )
            
            # compute avg, min, max loss
            batch_loss = loss.item()
            min_batch_loss = min(min_batch_loss, batch_loss)
            max_batch_loss = max(max_batch_loss, batch_loss)
            total_val_loss += batch_loss
            val_batches += 1

            # update progress bar
            current_val_loss = total_val_loss / val_batches
            val_pbar.set_postfix({
                'Val Loss': f'{current_val_loss:.4f}',
                'Batch': f'{val_batches}'
            })
        
        # compute avg loss
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0

        # compute score + generate samples
        metric_score = None
        log_samples_path = None

        # log samples path if generate
        if run_name and global_step is not None and train_cfg is not None:
            # Use run_name directly, sanitize for file paths
            base_name = run_name.replace("/", "_")
            os.makedirs(global_cfg.eval_dir, exist_ok=True)
            log_samples_path = os.path.join(global_cfg.eval_dir, base_name, f"samples_step_{global_step}.jsonl")
        
        # compute metric score
        if metric == "cider":
            metric_score = compute_cider_score(
                model, gen_loader, device, tokenizer,
                max_new_tokens=30, # TODO: make this a parameter
                max_samples=5000, # TODO: make this a parameter
                log_samples_path=log_samples_path,
                is_dual_tower=False
            )
        elif metric == "accuracy":
            metric_score = compute_ocrvqa_accuracy(
                model, gen_loader, device, tokenizer,
                max_new_tokens=50, # TODO: make this a parameter
                max_samples=5000, # TODO: make this a parameter
                log_samples_path=log_samples_path,
                is_dual_tower=False
            )
        elif metric == "bleu":
            pass
        else:
            raise ValueError(f"Metric {metric} not supported")

    return avg_val_loss, min_batch_loss, max_batch_loss, metric_score


def evaluate_validation_dual_tower(model, val_loader, gen_loader, device, train_cfg, global_cfg, run_name=None, 
                                   global_step=None, tokenizer=None, metric="cider"):
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # evaluate validation loss
    with torch.no_grad():
        total_val_loss = 0
        val_batches = 0
        min_batch_loss = float('inf')
        max_batch_loss = float('-inf')
        
        # tqdm progress bar
        val_pbar = tqdm(
            synchronized_dataloader_step(val_loader, is_dist()),
            total=min(len(val_loader), train_cfg.max_val_batches),
            desc="Validation",
            leave=False
        )
        
        for batch in val_pbar:
            if val_batches >= train_cfg.max_val_batches:
                break
                
            if len(batch["images"]) == 0:
                continue
                
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            last_img_idx = batch["last_img_idx"]
            
            # forward pass with mixed precision
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
                _, loss = model(
                    input_ids=input_ids, 
                    images=images, 
                    attention_mask=attention_mask, 
                    targets=labels,
                    last_img_idx=last_img_idx
                )
            
            # compute avg, min, max loss
            batch_loss = loss.item()
            min_batch_loss = min(min_batch_loss, batch_loss)
            max_batch_loss = max(max_batch_loss, batch_loss)
            total_val_loss += batch_loss
            val_batches += 1

            # update progress bar
            current_val_loss = total_val_loss / val_batches
            val_pbar.set_postfix({
                'Val Loss': f'{current_val_loss:.4f}',
                'Batch': f'{val_batches}'
            })
        
        # compute avg loss
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        avg_val_loss = dist_mean_scalar(avg_val_loss) if is_dist() else avg_val_loss
        
        if is_dist():
            min_batch_loss = min(dist_gather(min_batch_loss))
            max_batch_loss = max(dist_gather(max_batch_loss))

        # compute score + generate samples
        metric_score = None
        log_samples_path = None

        # log samples path if generate
        if run_name and global_step is not None and train_cfg is not None:
            base_name = run_name.replace("/", "_")
            os.makedirs(global_cfg.eval_dir, exist_ok=True)
            log_samples_path = os.path.join(global_cfg.eval_dir, base_name, f"samples_step_{global_step}.jsonl")
        
        # compute metric score (only on master for dual tower)
        if is_master() and gen_loader is not None and tokenizer is not None:
            if metric == "cider":
                metric_score = compute_cider_score(
                    model, gen_loader, device, tokenizer,
                    max_new_tokens=30,
                    max_samples=5000,
                    log_samples_path=log_samples_path,
                    is_dual_tower=True
                )
            elif metric == "accuracy":
                metric_score = compute_ocrvqa_accuracy(
                    model, gen_loader, device, tokenizer,
                    max_new_tokens=50,
                    max_samples=5000,
                    log_samples_path=log_samples_path,
                    is_dual_tower=True
                )
            elif metric == "bleu":
                pass
            else:
                raise ValueError(f"Metric {metric} not supported")

    return avg_val_loss, min_batch_loss, max_batch_loss, metric_score
