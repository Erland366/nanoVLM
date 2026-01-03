import wandb
from train_utils.utils import is_master


def init_wandb(train_cfg, vlm_cfg, global_cfg, run_name, total_dataset_size):
    if not train_cfg.log_wandb or not is_master():
        return None
    
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
    return run


def log_baseline_validation(run, train_cfg, baseline_val_loss, baseline_min_loss, 
                           baseline_max_loss, baseline_metric_score, global_step=0):
    if not train_cfg.log_wandb or not is_master() or run is None:
        return
    
    log_dict = {
        "val/val_loss": baseline_val_loss,
        "val/min_val_loss": baseline_min_loss,
        "val/max_val_loss": baseline_max_loss,
    }
    if baseline_metric_score is not None:
        log_dict[f"val/{train_cfg.metric}_score"] = baseline_metric_score
    run.log(log_dict, step=global_step)


def log_training_step(run, train_cfg, batch_loss, gradient_stats, global_step):
    if not train_cfg.log_wandb or not is_master() or run is None:
        return
    
    log_dict = {"train/batch_loss": batch_loss}
    if gradient_stats is not None:
        log_dict.update({
            "train/raw_grad_norm": gradient_stats["raw_grad_norm"],
            "train/total_grad_norm": gradient_stats["total_grad_norm"],
            "train/clipped_grad_norm": gradient_stats["clipped_grad_norm"],
        })
    run.log(log_dict, step=global_step)


def log_validation_step(run, train_cfg, avg_val_loss, min_val_loss, max_val_loss, 
                       metric_score, global_step):
    if not train_cfg.log_wandb or not is_master() or run is None:
        return
    
    log_dict = {
        "val/val_loss": avg_val_loss,
        "val/min_val_loss": min_val_loss,
        "val/max_val_loss": max_val_loss,
    }
    if metric_score is not None:
        log_dict[f"val/{train_cfg.metric}_score"] = metric_score
    run.log(log_dict, step=global_step)


def log_epoch_stats(run, train_cfg, avg_loss, epoch_duration, epoch_tokens_per_second):
    if not train_cfg.log_wandb or not is_master() or run is None:
        return
    
    run.log({
        "epoch_loss": avg_loss,
        "epoch_duration": epoch_duration,
        "epoch_tokens_per_second": epoch_tokens_per_second
    })


def finish_wandb(run, train_cfg, epoch_times):
    if not train_cfg.log_wandb or not is_master() or run is None:
        return
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    run.summary["avg_epoch_time"] = avg_epoch_time
    wandb.finish()


def print_baseline_validation(baseline_val_loss, baseline_min_loss, baseline_max_loss):
    if not is_master():
        return
    
    print("\n" + "="*50)
    print("Computing baseline validation loss (step 0)...")
    print("="*50)
    print(f"\nBaseline Validation Loss (step 0): {baseline_val_loss:.4f}")
    print(f"  ├── Min batch loss: {baseline_min_loss:.4f}")
    print(f"  └── Max batch loss: {baseline_max_loss:.4f}")
    print("="*50 + "\n")


def print_validation_step(global_step, avg_val_loss):
    if not is_master():
        return
    print(f"\nStep {global_step} | Val Loss: {avg_val_loss:.4f}")


def print_epoch_stats(epoch, avg_loss, epoch_duration, epoch_tokens_per_second):
    if not is_master():
        return
    print(f"Epoch {epoch} done | Train Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

