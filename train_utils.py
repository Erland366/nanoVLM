import math
import os
import random
import time
from datetime import timedelta
from statistics import mean

import numpy
import torch
import torch.distributed as dist
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    interleave_datasets,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import VQADataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer

PG_CPU = None


def set_pg_cpu(group):
    global PG_CPU
    PG_CPU = group


def init_dist():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
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


def create_cpu_group():
    return dist.new_group(backend="gloo")


def dist_gather(obj):
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(global_cfg):
    torch.manual_seed(global_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_cfg.seed)
    random.seed(global_cfg.seed)
    numpy.random.seed(global_cfg.seed)


def get_run_name(train_cfg, vlm_cfg):
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    base_name = f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"
    if train_cfg.prefix_run_name:
        return f"{train_cfg.prefix_run_name}_{base_name}"
    return base_name


def _normalize_dataset_names(dataset_names):
    if isinstance(dataset_names, str):
        return (dataset_names,)
    return tuple(dataset_names)


def _maybe_get_dataset_size(train_cfg, dataset_name):
    try:
        builder = load_dataset_builder(train_cfg.train_dataset_path, dataset_name)
        split_info = builder.info.splits.get("train")
        if split_info is None:
            return None
        return split_info.num_examples
    except Exception as exc:
        if is_master():
            print(f"Warning: Failed to read dataset size for '{dataset_name}': {exc}")
        return None


def _get_dataset_sizes(train_cfg, dataset_names, show_progress):
    sizes = {}
    iterator = dataset_names
    if show_progress:
        iterator = tqdm(dataset_names, desc="Reading dataset sizes", leave=False)
    for dataset_name in iterator:
        if "shard_" in dataset_name:
            sizes[dataset_name] = None
            continue
        sizes[dataset_name] = _maybe_get_dataset_size(train_cfg, dataset_name)
    return sizes


def _resolve_interleave_probabilities(train_cfg, dataset_names, dataset_sizes):
    manual_probs = getattr(train_cfg, "interleave_probabilities", None)
    if manual_probs is not None:
        if len(manual_probs) != len(dataset_names):
            raise ValueError(
                "interleave_probabilities must match the number of datasets."
            )
        total = sum(manual_probs)
        return [float(p) / total for p in manual_probs]

    if all(size is not None for size in dataset_sizes):
        total = sum(dataset_sizes)
        if total > 0:
            return [size / total for size in dataset_sizes]
        return None

    missing = [name for name, size in zip(dataset_names, dataset_sizes) if size is None]
    raise ValueError(
        "Missing dataset sizes for interleaving: "
        f"{', '.join(missing)}. "
        "Ensure the dataset metadata is available."
    )


def _compute_stratified_counts(probabilities, total):
    raw_counts = [prob * total for prob in probabilities]
    base_counts = [int(count) for count in raw_counts]
    remainder = total - sum(base_counts)
    if remainder > 0:
        fractions = sorted(
            range(len(raw_counts)),
            key=lambda idx: raw_counts[idx] - base_counts[idx],
            reverse=True,
        )
        for idx in fractions[:remainder]:
            base_counts[idx] += 1
    return base_counts


def _maybe_set_val_size_from_ratio(train_cfg, total_size):
    val_ratio = getattr(train_cfg, "val_ratio", None)
    if val_ratio is None or total_size is None:
        return
    if total_size <= 0:
        return
    val_size = max(1, int(total_size * val_ratio))
    world_size = get_world_size()
    if val_size < world_size:
        val_size = world_size
    train_cfg.val_size = val_size


def _maybe_shuffle_streaming_dataset(dataset, train_cfg, global_cfg, seed_offset):
    if not train_cfg.stream_dataset:
        return dataset
    buffer_size = getattr(train_cfg, "streaming_shuffle_buffer", 0)
    if buffer_size is None or buffer_size <= 0:
        return dataset
    return dataset.shuffle(buffer_size=buffer_size, seed=global_cfg.seed + seed_offset)


def get_dataloaders(train_cfg, vlm_cfg, global_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    dataset_names_to_load = _normalize_dataset_names(train_cfg.train_dataset_name)
    if "shards" in train_cfg.train_dataset_name:
        print("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    show_progress = is_master()
    dataset_sizes = _get_dataset_sizes(train_cfg, dataset_names_to_load, show_progress)

    combined_train_data = []
    combined_names = []
    combined_sizes = []

    dataset_iter = enumerate(dataset_names_to_load)
    if show_progress:
        dataset_iter = tqdm(
            dataset_iter,
            total=len(dataset_names_to_load),
            desc="Loading datasets",
            leave=False,
        )
    for dataset_idx, dataset_name in dataset_iter:
        print(f"Loading dataset: {dataset_name}")
        if "shard_" in dataset_name:
            try:
                train_ds = load_from_disk(dataset_name)
                train_ds = _maybe_shuffle_streaming_dataset(train_ds, train_cfg, global_cfg, dataset_idx)
                combined_train_data.append(train_ds)
                combined_names.append(dataset_name)
                combined_sizes.append(dataset_sizes.get(dataset_name))
                continue
            except Exception as e:
                print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
                continue
        try:
            train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, streaming=train_cfg.stream_dataset, on_bad_files="warn")["train"]
            if train_cfg.stream_dataset:
                next(iter(train_ds))
            else:
                train_ds[0]
            train_ds = _maybe_shuffle_streaming_dataset(train_ds, train_cfg, global_cfg, dataset_idx)
            combined_train_data.append(train_ds)
            combined_names.append(dataset_name)
            combined_sizes.append(dataset_sizes.get(dataset_name))
        except Exception as e:
            if is_master():
                print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")

    stratified_val = getattr(train_cfg, "stratified_val_split", False)
    if stratified_val and not train_cfg.stream_dataset:
        if is_master():
            print("Stratified validation split is only supported in streaming mode; falling back to standard split.")
        stratified_val = False

    probs = None
    if getattr(train_cfg, "interleave_datasets", False) and len(combined_train_data) > 1:
        probs = _resolve_interleave_probabilities(train_cfg, combined_names, combined_sizes)
        if is_master():
            size_pairs = ", ".join(
                f"{name}={size}" for name, size in zip(combined_names, combined_sizes)
            )
            print(f"Interleave sizes: {size_pairs}")
            if probs is not None:
                pretty_probs = ", ".join(f"{name}={prob:.4f}" for name, prob in zip(combined_names, probs))
                print(f"Interleave probabilities: {pretty_probs}")

    val_ds = None
    if stratified_val:
        total_size = sum(combined_sizes)
        if total_size <= 0:
            raise ValueError("Cannot build stratified validation split without dataset sizes.")
        val_ratio = getattr(train_cfg, "val_ratio", None)
        if val_ratio is not None:
            val_total = max(1, int(total_size * val_ratio))
        else:
            val_total = min(int(train_cfg.val_size), total_size)
        train_cfg.val_size = val_total

        if probs is None:
            probs = _resolve_interleave_probabilities(train_cfg, combined_names, combined_sizes)
        val_counts = _compute_stratified_counts(probs, val_total)
        if is_master():
            count_pairs = ", ".join(
                f"{name}={count}" for name, count in zip(combined_names, val_counts)
            )
            print(f"Stratified val counts: {count_pairs}")
        train_splits = []
        val_splits = []
        val_names = []
        val_sizes = []
        split_iter = zip(combined_names, combined_train_data, val_counts)
        if is_master():
            split_iter = tqdm(
                split_iter,
                total=len(combined_names),
                desc="Stratifying val split",
                leave=False,
            )
        for name, ds, val_count in split_iter:
            if val_count <= 0:
                train_splits.append(ds)
                continue
            val_splits.append(ds.take(val_count))
            train_splits.append(ds.skip(val_count))
            val_names.append(name)
            val_sizes.append(val_count)

        if getattr(train_cfg, "interleave_datasets", False) and len(train_splits) > 1:
            train_ds = interleave_datasets(
                train_splits,
                probabilities=probs,
                seed=global_cfg.seed,
                stopping_strategy=getattr(train_cfg, "interleave_stopping_strategy", "all_exhausted"),
            )
        else:
            train_ds = concatenate_datasets(train_splits)

        if val_splits:
            val_probs = [size / sum(val_sizes) for size in val_sizes]
            if getattr(train_cfg, "interleave_datasets", False) and len(val_splits) > 1:
                val_ds = interleave_datasets(
                    val_splits,
                    probabilities=val_probs,
                    seed=global_cfg.seed,
                    stopping_strategy=getattr(train_cfg, "interleave_stopping_strategy", "all_exhausted"),
                )
            else:
                val_ds = concatenate_datasets(val_splits)
        else:
            raise ValueError("Stratified validation split produced empty validation data.")
    else:
        if getattr(train_cfg, "interleave_datasets", False) and len(combined_train_data) > 1:
            train_ds = interleave_datasets(
                combined_train_data,
                probabilities=probs,
                seed=global_cfg.seed,
                stopping_strategy=getattr(train_cfg, "interleave_stopping_strategy", "all_exhausted"),
            )
        else:
            train_ds = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=global_cfg.seed)

    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    if val_ds is None:
        if getattr(train_cfg, "val_ratio", None) is not None:
            total_size = None
            if all(size is not None for size in combined_sizes):
                total_size = sum(combined_sizes)
            elif not train_cfg.stream_dataset:
                total_size = sum(len(ds) for ds in combined_train_data)
            _maybe_set_val_size_from_ratio(train_cfg, total_size)

        val_size = int(train_cfg.val_size / get_world_size())
        print(f"Val size per GPU: {val_size}")
        if train_cfg.stream_dataset:
            val_ds = train_ds.take(val_size)
            train_ds = train_ds.skip(val_size)
        else:
            val_ds = train_ds.select(range(val_size))
            train_ds = train_ds.select(range(val_size, len(train_ds)))
    else:
        val_size = int(train_cfg.val_size / get_world_size())
        print(f"Val size per GPU: {val_size}")
        if is_dist():
            val_ds = val_ds.shard(num_shards=get_world_size(), index=get_rank())

    train_dataset = VQADataset(
        train_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size*4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    val_dataset = ConstantLengthDataset(
        val_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size*4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(global_cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=3,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("Warming up dataloaders...")
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader


def get_streaming_val_batches(train_cfg, val_loader):
    batch_size = val_loader.batch_size or train_cfg.batch_size
    val_size = getattr(train_cfg, "val_size", None)
    if val_size is None:
        return train_cfg.max_val_batches
    per_rank = max(1, int(val_size / get_world_size()))
    total = math.ceil(per_rank / batch_size)
    return min(total, train_cfg.max_val_batches)


def save_model_checkpoint(model, train_cfg, global_step=None, epoch=None, epoch_step=None, is_final=False):
    if not is_master():
        return

    save_model = model.module if is_dist() else model

    if is_final:
        local_path = train_cfg.local_model_cp_path
        hf_path = train_cfg.hf_model_cp_path
    else:
        if epoch is not None and epoch_step is not None:
            step_name = f"epoch{epoch}_step{epoch_step}"
        else:
            step_name = str(global_step)
        local_path = f"{train_cfg.local_model_cp_path}-{step_name}"
        hf_path = f"{train_cfg.hf_model_cp_path}-{step_name}"

    if train_cfg.save_local:
        save_model.save_pretrained(local_path)
    if train_cfg.save_hf:
        save_model.push_to_hub(hf_path)


def evaluate_validation(model, val_loader, device, train_cfg):
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        total_val_loss = 0
        val_batches = 0
        min_batch_loss = float("inf")
        max_batch_loss = float("-inf")

        if train_cfg.stream_dataset:
            max_val_batches = get_streaming_val_batches(train_cfg, val_loader)
        else:
            max_val_batches = min(len(val_loader), train_cfg.max_val_batches)

        val_iter = synchronized_dataloader_step(val_loader, is_dist())
        val_pbar = tqdm(
            val_iter,
            total=max_val_batches,
            desc="Validation",
            leave=False,
            disable=not is_master(),
        )

        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16,
        )
        for batch in val_pbar:
            if val_batches >= max_val_batches:
                break
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast_context:
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            batch_loss = loss.item()
            min_batch_loss = min(min_batch_loss, batch_loss)
            max_batch_loss = max(max_batch_loss, batch_loss)
            total_val_loss += batch_loss
            val_batches += 1

            current_val_loss = total_val_loss / val_batches
            val_pbar.set_postfix({
                "Val Loss": f"{current_val_loss:.4f}",
                "Batch": f"{val_batches}",
            })

        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
        if is_dist():
            min_batch_loss = min(dist_gather(min_batch_loss))
            max_batch_loss = max(dist_gather(max_batch_loss))

    return avg_val_loss, min_batch_loss, max_batch_loss
