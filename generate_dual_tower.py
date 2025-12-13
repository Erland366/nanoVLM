import argparse
import torch
import json
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from models.config import VLMConfig
from huggingface_hub import hf_hub_download

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.dual_tower import DualTowerVLM
from data.processors import get_tokenizer, get_image_processor
from data.datasets import VQADualDataset
from data.collators import VQADualCollator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions for COCO images with DualTowerVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="patrickamadeus/nanoVLM-230M-8k-vanilla-cococap-full-10000",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--dataset", type=str, default="jxie/coco_captions",
                        help="Path to dataset with images and captions")
    parser.add_argument("--prompt", type=str, default="Describe the image.",
                        help="Text prompt to feed the model")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of tokens per output")
    parser.add_argument("--total_samples", type=int, default=30000,
                        help="Number of samples to process from the dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--measure_vram", action="store_true",
                        help="Measure and display VRAM usage during model loading and generation")
    return parser.parse_args()


class COCOGenerationDataset(Dataset):
    """
    Wrapper dataset for COCO caption generation.
    Prepares dataset for VQADualDataset and adds metadata for evaluation.
    """
    
    def __init__(self, dataset, tokenizer, image_processor, prompt, mp_image_token_length, total_samples):
        print("Preparing dataset...")
        self.samples = []
        seen_filenames = set()
        
        max_iter = min(total_samples, len(dataset)) if total_samples > 0 else len(dataset)
        for i in tqdm(range(max_iter), desc="Filtering unique samples"):
            sample = dataset[i]
            filename_id = sample.get("filename", str(i))
            
            if filename_id in seen_filenames:
                continue
            seen_filenames.add(filename_id)
            
            # Format for VQADualDataset: needs 'image' and 'caption' keys
            self.samples.append({
                "image": sample["image"],
                "caption": prompt,  # Use prompt as the "caption" for generation
                "gt_caption": sample.get("caption", ""),
                "filename": filename_id,
                "image_id": len(self.samples)
            })
            
            if len(self.samples) >= total_samples:
                break
        
        print(f"Loaded {len(self.samples)} unique samples")
        
        # Use VQADualDataset for processing
        self.vqa_dataset = VQADualDataset(
            dataset=self.samples,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mp_image_token_length=mp_image_token_length
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get processed sample from VQADualDataset and add metadata."""
        # Get processed data from VQADualDataset
        processed = self.vqa_dataset[idx]
        
        if processed is None:
            return None
        
        # Add metadata for evaluation
        processed["gt_caption"] = self.samples[idx]["gt_caption"]
        processed["filename"] = self.samples[idx]["filename"]
        processed["image_id"] = self.samples[idx]["image_id"]
        
        return processed


class GenerationCollator(VQADualCollator):
    """
    Collator for generation that extends VQADualCollator.
    Adds metadata fields (gt_captions, filenames, image_ids) to the batch.
    """
    
    def __call__(self, batch):
        """Collate batch with center padding and preserve metadata."""
        # Extract metadata before collating
        gt_captions = [item.pop("gt_caption") for item in batch]
        filenames = [item.pop("filename") for item in batch]
        image_ids = [item.pop("image_id") for item in batch]
        
        # Use parent class to handle center padding
        collated = super().__call__(batch)
        
        # Add metadata back
        collated["gt_captions"] = gt_captions
        collated["filenames"] = filenames
        collated["image_ids"] = image_ids
        
        return collated


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    
    # Build VLMConfig from checkpoint
    if os.path.isdir(source):
        config_path = os.path.join(source, "config.json")
        weights_path = os.path.join(source, "model.safetensors")
    else:
        config_path = hf_hub_download(repo_id=source, filename="config.json")
        weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")
    
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = VLMConfig(**cfg_dict)
    cfg.vlm_checkpoint_path = source
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    # Load DualTowerVLM model
    model = DualTowerVLM(
        cfg,
        load_backbone=False,  # We'll load from checkpoint
        freeze_left_vision=True,
        freeze_left_projector=True,
        freeze_left_decoder=True,
        freeze_right_decoder=True,
    )
    
    # Load weights from checkpoint (load into left_tower which is VisionLanguageModel)
    from safetensors.torch import load_model
    load_model(model.left_tower, weights_path)
    
    model = model.to(device)
    model.eval()
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")

    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    resize_to_max_side_len = getattr(cfg, "resize_to_max_side_len", False)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, resize_to_max_side_len)

    # Load dataset
    from datasets import load_dataset
    print("Loading HuggingFace dataset...")
    hf_dataset = load_dataset(args.dataset, split="validation", streaming=False)
    print(f"Dataset loaded with {len(hf_dataset)} total samples")
    
    coco_dataset = COCOGenerationDataset(
        hf_dataset,
        tokenizer,
        image_processor,
        args.prompt,
        cfg.mp_image_token_length,
        args.total_samples
    )
    
    # Use GenerationCollator with a reasonable max_length for generation
    collator = GenerationCollator(tokenizer, max_length=2048)
    
    dataloader = DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model_name = source.split("/")[-1] if "/" in source else source
    os.makedirs(os.path.join(os.getcwd(), "inference_results_dual_tower"), exist_ok=True)
    output_file = os.path.join(os.getcwd(), "inference_results_dual_tower", f"result_{model_name}.jsonl")
    
    print(f"Using batch size: {args.batch_size}, num_workers: {args.num_workers}")
    print(f"Starting inference on {len(coco_dataset)} samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Generating captions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            last_img_idx = batch["last_img_idx"]
            images = batch["images"]
            
            try:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        images=images,
                        attention_mask=attention_mask,
                        last_img_idx=last_img_idx,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        greedy=True
                    )
                pred_captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                pred_captions = [""] * len(batch["image_ids"])
            
            for i in range(len(batch["image_ids"])):
                record = {
                    "image_id": batch["image_ids"][i],
                    "filename": batch["filenames"][i],
                    "question": args.prompt,
                    "pred_caption": pred_captions[i],
                    "gt_caption": batch["gt_captions"][i]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            f.flush()
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()





