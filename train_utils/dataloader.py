import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, concatenate_datasets, get_dataset_config_names
from data.processors import get_image_processor, get_tokenizer
from data.datasets import (
    VQADataset,
    COCOCaptionsVanillaDataset,
)
from data.collators import (
    VQACollator,
    COCOCaptionsVanillaCollator,
)
from data.advanced_datasets import ConstantLengthDataset
from evaluation.cider_utils import (
    VanillaCOCOGenerationDataset,
    VanillaGenerationCollator,
)
from train_utils.utils import is_dist, is_master, get_world_size, get_rank, seed_worker


DATASET_COLLATOR_MAP = {
    "coco_caption": (COCOCaptionsVanillaDataset, COCOCaptionsVanillaCollator),
    "vqa": (VQADataset, VQACollator),
}

EVAL_DATASET_COLLATOR_MAP = {
    "coco_caption": (VanillaCOCOGenerationDataset, VanillaGenerationCollator),
}


def get_custom_dataloaders(train_cfg, vlm_cfg, global_cfg, train_split = "train", val_split = "validation", generate_val_data = False, val_ratio = 0.2):
    print("Custom dataset loading mode...")
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")

    if train_cfg.custom_dataset_id not in DATASET_COLLATOR_MAP:
        raise ValueError(f"Custom dataset ID '{train_cfg.custom_dataset_id}' not found in DATASET_COLLATOR_MAP")
    DatasetClass, CollatorClass = DATASET_COLLATOR_MAP[train_cfg.custom_dataset_id]

    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    full_dataset = load_dataset(train_cfg.train_dataset_path)
    
    if generate_val_data or val_split not in full_dataset:
        if is_master():
            print(f"Generating validation split from training data with ratio {val_ratio}")
        
        if train_split not in full_dataset:
            raise ValueError(f"Train split '{train_split}' not found in dataset '{train_cfg.train_dataset_path}'")
        
        full_train_ds = full_dataset[train_split].shuffle(seed=global_cfg.seed)
        
        total_size = len(full_train_ds)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        if is_master():
            print(f"Total dataset size: {total_size}, Train size: {train_size}, Val size: {val_size}")
        
        train_ds = full_train_ds.select(range(train_size))
        val_ds = full_train_ds.select(range(train_size, total_size))
    else:
        if train_split not in full_dataset:
            raise ValueError(f"Train split '{train_split}' not found in dataset '{train_cfg.train_dataset_path}'")
        if val_split not in full_dataset:
            raise ValueError(f"Validation split '{val_split}' not found in dataset '{train_cfg.train_dataset_path}'")
        
        train_ds = full_dataset[train_split].shuffle(seed=global_cfg.seed)
        val_ds = full_dataset[val_split]
    
    # Shard datasets for distributed training
    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())
        val_ds = val_ds.shard(num_shards=get_world_size(), index=get_rank())
        if is_master():
            print(f"Sharded datasets across {get_world_size()} GPUs")

    train_dataset = DatasetClass(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = DatasetClass(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    collator = CollatorClass(tokenizer, vlm_cfg.lm_max_length)

    if train_cfg.use_packing:
        train_dataset = ConstantLengthDataset(train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=2,
                                            max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)
        val_dataset = ConstantLengthDataset(val_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=2,
                                            max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=False,
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

    return train_loader, val_loader

def get_cauldron_dataloaders(train_cfg, vlm_cfg, global_cfg):
    print("HuggingfaceM4/the_cauldron dataset loading mode...")
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    print(f"Dataset names to load: {train_cfg.train_dataset_name}")

    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    dataset_names_to_load = train_cfg.train_dataset_name
    if "shards" in train_cfg.train_dataset_name:
        print("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine all training datasets
    combined_train_data = []

    for dataset_name in dataset_names_to_load:
        print(f"Loading dataset: {dataset_name}")
        if "shard_" in dataset_name:
            try:
                train_ds = load_from_disk(dataset_name)
                combined_train_data.append(train_ds)
                continue
            except Exception as e:
                print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
                continue
        try:
            train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, streaming=train_cfg.stream_dataset, on_bad_files='warn')['train']
            if train_cfg.stream_dataset:
                next(iter(train_ds)) # Check if the dataset is loaded correctly
            else:
                train_ds[0] # Check if the dataset is loaded correctly
            combined_train_data.append(train_ds)
        except Exception as e:
            if is_master():
                print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")
    
    train_ds = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=global_cfg.seed) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  


    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # train_ds = train_ds.shuffle(buffer_size=10000, seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  

    val_size = int(train_cfg.val_size/get_world_size())
    print(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
    else:
        val_ds = train_ds.select(range(val_size))
        train_ds = train_ds.select(range(val_size, len(train_ds)))

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

    if train_cfg.use_packing:
        train_dataset = ConstantLengthDataset(train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=2,
                                            max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

        val_dataset = ConstantLengthDataset(val_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=2,
                                            max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(global_cfg.seed)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=False,
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
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    print("Warming up dataloaders...")   
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader


def get_eval_dataloaders(train_cfg, vlm_cfg, global_cfg, eval_split = "train", total_samples = 5000):
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    dataset_hf = load_dataset(train_cfg.custom_eval_dataset_path, split=eval_split)
    DatasetClass, CollatorClass = EVAL_DATASET_COLLATOR_MAP[train_cfg.eval_dataset_id]
    cider_dataset = DatasetClass(dataset_hf, tokenizer, image_processor, vlm_cfg.mp_image_token_length, total_samples)
    cider_collator = CollatorClass(tokenizer, max_length=2048)
    
    g = torch.Generator()
    g.manual_seed(global_cfg.seed)
    
    eval_loader = DataLoader(
        cider_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=cider_collator,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    print("Warming up eval dataloader...")
    next(iter(eval_loader))
    print("Eval dataloader warmup complete.")
    
    return eval_loader


def get_dataloaders(train_cfg, vlm_cfg, global_cfg):
    train_loader, val_loader = None, None
    gen_loader = None
    
    # Get tokenizer (used in both custom and cauldron dataloaders)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    if train_cfg.use_custom_dataset:
        train_loader, val_loader = get_custom_dataloaders(train_cfg, vlm_cfg, global_cfg, train_split = train_cfg.train_split, val_split = train_cfg.val_split, generate_val_data = train_cfg.generate_val_data, val_ratio = train_cfg.val_ratio)
    else:
        train_loader, val_loader = get_cauldron_dataloaders(train_cfg, vlm_cfg, global_cfg)
    gen_loader = get_eval_dataloaders(train_cfg, vlm_cfg, global_cfg, eval_split = train_cfg.eval_split, total_samples = train_cfg.total_samples)
    return train_loader, val_loader, gen_loader, tokenizer

