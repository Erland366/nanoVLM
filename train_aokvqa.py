import os
import sys
import math
import time
import gc
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
import random
import numpy as np
from tqdm import tqdm

from data.coco_captions import COCODataset, COCOCollator
from data.aokvqa import AOKVQADataset, AOKVQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from configs.config_vanilla import VLMConfig, TrainConfig, GlobalConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/vast/users/yova.kementchedjhieva/patrick/huggingface"

# --------------------------
# Utility Functions
# --------------------------

def set_seed(global_cfg):
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    np.random.seed(global_cfg.seed)


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

    train_dataset = AOKVQADataset(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = AOKVQADataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    collator = AOKVQACollator(tokenizer, vlm_cfg.lm_max_length)

    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size * 4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.eval_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )

    return train_loader, val_loader


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


def extract_answer_letter(text):
    """Extract answer letter (a, b, c, d) from generated text using regex.
    
    Handles multiple formats:
    - (a), (b), (c), (d)
    - a., b., c., d.
    - a), b), c), d)
    - Just 'a', 'b', 'c', 'd' at the beginning
    """
    text = text.lower().strip()
    
    # Pattern 1: (a), (b), (c), (d)
    match = re.search(r'\(([abcd])\)', text)
    if match:
        return match.group(1)
    
    # Pattern 2: a., b., c., d.
    match = re.search(r'\b([abcd])\.', text)
    if match:
        return match.group(1)
    
    # Pattern 3: a), b), c), d)
    match = re.search(r'\b([abcd])\)', text)
    if match:
        return match.group(1)
    
    # Pattern 4: Just the letter at the start
    match = re.match(r'^([abcd])\b', text)
    if match:
        return match.group(1)
    
    # Pattern 5: First occurrence of a, b, c, or d
    match = re.search(r'\b([abcd])\b', text)
    if match:
        return match.group(1)
    
    return None


def evaluate_validation(model, val_loader, tokenizer, device):
    """Validation loss and accuracy evaluation with text generation."""
    model.eval()
    total_val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        val_pbar = tqdm(
            enumerate(synchronized_dataloader_step(val_loader, False)),
            total=len(val_loader),
            desc="Validation",
            leave=False,
        )

        for batch_idx, batch in val_pbar:
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Calculate loss
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, val_loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            total_val_loss += val_loss.item()

            # Calculate accuracy by generating text
            batch_size = input_ids.shape[0]
            
            # Find where "Answer:" prompt ends for each sample (last non-padded position before answer)
            for i in range(batch_size):
                # Get the input sequence up to where generation should start
                # Find the last position before the label starts
                valid_labels = labels[i] != -100
                if valid_labels.any():
                    # Find the first valid label position (where generation should start)
                    first_label_pos = valid_labels.nonzero()[0].item()
                    
                    # Get the input up to that point
                    prompt_ids = input_ids[i:i+1, :first_label_pos]
                    prompt_attention = attention_mask[i:i+1, :first_label_pos]
                    
                    # Generate the answer (just need 1-2 tokens for 'a', 'b', 'c', 'd')
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        generated_ids = model.generate(
                            prompt_ids,
                            [images[i]],
                            attention_mask=prompt_attention,
                            max_new_tokens=5,
                            greedy=True,  # Use greedy decoding for deterministic results
                        )
                    
                    # Decode the generated text (only the new tokens)
                    generated_text = tokenizer.decode(generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
                    
                    # Extract the predicted answer letter
                    predicted_letter = extract_answer_letter(generated_text)
                    
                    # Get the ground truth letter
                    last_valid_label_pos = valid_labels.nonzero()[-1].item()
                    ground_truth_token_id = labels[i, last_valid_label_pos].item()
                    ground_truth_letter = tokenizer.decode([ground_truth_token_id]).strip().lower()
                    
                    # Check if prediction matches ground truth
                    if predicted_letter and predicted_letter == ground_truth_letter:
                        total_correct += 1
                    
                    total_samples += 1

            current_val_loss = total_val_loss / (batch_idx + 1)
            current_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            val_pbar.set_postfix({
                "Val Loss": f"{current_val_loss:.4f}",
                "Val Acc": f"{current_accuracy:.4f}",
                "Batch": f"{batch_idx + 1}/{len(val_loader)}",
            })

            del batch, images, input_ids, attention_mask, labels, val_loss

    avg_val_loss = total_val_loss / len(val_loader)
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_val_loss, avg_accuracy


# --------------------------
# Main Training Loop
# --------------------------

def train(train_cfg, vlm_cfg, global_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if train_cfg.log_wandb:
        run_name = (
            f"{train_cfg.prefix_run_name}-"
            f"lrV.{train_cfg.lr_vision_backbone}-"
            f"lrL.{train_cfg.lr_language_backbone}-"
            f"lrMP.{train_cfg.lr_mp}-"
            f"bs.{train_cfg.batch_size}"
        )
        wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            config={
                "train_config": train_cfg.__dict__,
                "vlm_config": vlm_cfg.__dict__,
                "global_config": global_cfg.__dict__,
            },
        )

    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg, global_cfg)

    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

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
    max_training_steps = (
        train_cfg.max_training_steps if not train_cfg.use_epochs else train_cfg.max_epochs * len(train_loader)
    )

    while global_step < max_training_steps:
        model.train()
        total_train_loss, total_tokens_processed = 0, 0
        optimizer.zero_grad()
        print(f"Starting training epoch {epoch + 1}")

        train_pbar = tqdm(
            enumerate(synchronized_dataloader_step(train_loader, False)),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )

        for i, batch in train_pbar:
            is_update = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss /= train_cfg.gradient_accumulation_steps

            loss.backward()

            if is_update:
                if train_cfg.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(all_params, train_cfg.max_grad_norm)

                # LR updates
                idx = 0
                if train_cfg.lr_mp > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_mp, max_training_steps)
                    idx += 1
                if train_cfg.lr_vision_backbone > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_vision_backbone, max_training_steps)
                    idx += 1
                if train_cfg.lr_language_backbone > 0:
                    optimizer.param_groups[idx]["lr"] = get_lr(global_step, train_cfg.lr_language_backbone, max_training_steps)

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item() * (train_cfg.gradient_accumulation_steps if train_cfg.gradient_accumulation_steps > 1 else 1)
            total_train_loss += batch_loss
            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens

            train_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Step": global_step})
            if train_cfg.log_wandb:
                wandb.log({"train/train_loss": batch_loss}, step=global_step)

            del images, input_ids, labels, attention_mask, loss

            if global_step % train_cfg.eval_every_n_steps == 0:
                torch.cuda.empty_cache()
                val_loss, val_accuracy = evaluate_validation(model, val_loader, tokenizer, device)
                print(f"\nStep {global_step} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
                if train_cfg.log_wandb:
                    wandb.log({
                        "val/val_loss": val_loss,
                        "val/val_accuracy": val_accuracy
                    }, step=global_step)
                model.train()

            if global_step % train_cfg.save_model_every_n_steps == 0 and global_step > 0:
                if train_cfg.save_local:
                    model.save_pretrained(f"{train_cfg.local_model_cp_path}-{global_step}")
                if train_cfg.save_hf:
                    model.push_to_hub(f"{train_cfg.hf_model_cp_path}-{global_step}")

            global_step += 1

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} done | Train Loss: {avg_loss:.4f}")
        epoch += 1

    # Final save
    if train_cfg.save_local:
        model.save_pretrained(train_cfg.local_model_cp_path)
    if train_cfg.save_hf:
        model.push_to_hub(train_cfg.hf_model_cp_path)
    if train_cfg.log_wandb:
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

    print(f"Starting AOKVQA training on vanilla model")
    set_seed(global_cfg)
    train(train_cfg, vlm_cfg, global_cfg)


if __name__ == "__main__":
    main()