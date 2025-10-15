import argparse
import torch
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.config import VLMConfig
from huggingface_hub import hf_hub_download
import json
import os

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.twin_tower import TwinTowerModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions for COCO images with nanoVLM")
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


class COCODataset(Dataset):
    """Dataset for COCO caption generation with preprocessing."""
    
    def __init__(self, dataset, tokenizer, image_processor, prompt, mp_image_token_length, total_samples):
        """
        Args:
            dataset: HuggingFace dataset (non-streaming)
            tokenizer: The tokenizer
            image_processor: The image processor
            prompt: Text prompt for generation
            mp_image_token_length: Image token length from model config
            total_samples: Total number of samples to process
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.prompt = prompt
        self.mp_image_token_length = mp_image_token_length
        
        # Filter duplicates and limit samples
        print("Preparing dataset...")
        self.samples = []
        seen_filenames = set()
        
        # Limit iteration to total_samples or dataset length
        max_iter = min(total_samples, len(dataset)) if total_samples > 0 else len(dataset)
        for i in tqdm(range(max_iter), desc="Filtering unique samples"):
            sample = dataset[i]
            filename_id = sample.get("filename", str(i))
            
            if filename_id in seen_filenames:
                continue
            seen_filenames.add(filename_id)
            
            self.samples.append({
                "image": sample["image"],
                "gt_caption": sample.get("caption", ""),
                "filename": filename_id,
                "image_id": len(self.samples)  # Use actual index in unique samples
            })
            
            if len(self.samples) >= total_samples:
                break
        
        print(f"Loaded {len(self.samples)} unique samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert image to RGB
        image = sample["image"]
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        # Process image
        processed_image, splitted_image_ratio = self.image_processor(img)
        if not hasattr(self.tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_image) - 1:
            processed_image = processed_image[1:]
        
        # Create image string and tokenize
        image_string = get_image_string(self.tokenizer, [splitted_image_ratio], self.mp_image_token_length)
        messages = [{"role": "user", "content": image_string + self.prompt}]
        encoded_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        
        return {
            "processed_image": processed_image,
            "input_ids": torch.tensor(encoded_prompt),
            "gt_caption": sample["gt_caption"],
            "filename": sample["filename"],
            "image_id": sample["image_id"]
        }


def collate_fn(batch):
    """
    Collate function to batch samples with proper padding.
    
    Args:
        batch: List of samples from COCODataset
    
    Returns:
        Dictionary with batched tensors
    """
    # Extract components
    images = [item["processed_image"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]
    gt_captions = [item["gt_caption"] for item in batch]
    filenames = [item["filename"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    
    # Pad input_ids to the same length
    max_len = max(ids.size(0) for ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    # Get tokenizer pad token ID from first batch item (assuming all use same tokenizer)
    pad_token_id = 0  # Will be set by the model's tokenizer
    
    for ids in input_ids_list:
        pad_len = max_len - ids.size(0)
        # Pad on the left (as per collators.py lines 9, 65, 80)
        padded_ids = F.pad(ids, (pad_len, 0), value=pad_token_id)
        # Create attention mask (0 for padding, 1 for real tokens)
        attention_mask = F.pad(torch.ones_like(ids), (pad_len, 0), value=0)
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)
    
    return {
        "images": images,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
        "gt_captions": gt_captions,
        "filenames": filenames,
        "image_ids": image_ids
    }


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
    
    # --- build a VLMConfig, whether local or remote ---
    if os.path.isdir(source):
        # local checkpoint folder
        config_path = os.path.join(source, "config.json")
        weights_path = os.path.join(source, "model.safetensors")
    else:
        # remote HF repo
        config_path = hf_hub_download(repo_id=source, filename="config.json")
        weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")
    
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = VLMConfig(**cfg_dict)
    cfg.vlm_checkpoint_path = source  # make sure path field exists
    
    model = TwinTowerModel.from_pretrained(
        cfg,
        freeze_vision_encoder=True,
        freeze_modality_projector=False,
        freeze_left_tower_decoder=False,
        freeze_right_tower_decoder=True,
    ).to(device)
    model.eval()
    # source = args.checkpoint if args.checkpoint else args.hf_model
    # print(f"Loading weights from: {source}")
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    # model = TwinTowerModel.from_pretrained(source).to(device)
    # model.eval()
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, resize_to_max_side_len)

    # Load dataset (non-streaming for faster loading)
    from datasets import load_dataset
    print("Loading HuggingFace dataset...")
    hf_dataset = load_dataset(args.dataset, split="validation", streaming=False)
    print(f"Dataset loaded with {len(hf_dataset)} total samples")
    
    # Create Dataset and DataLoader
    coco_dataset = COCODataset(
        hf_dataset,
        tokenizer,
        image_processor,
        args.prompt,
        model.cfg.mp_image_token_length,
        args.total_samples
    )
    
    dataloader = DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model_name = source.split("/")[-1] if "/" in source else source
    os.makedirs(os.path.join(os.getcwd(),"inference_results_twin"), exist_ok=True)
    output_file = os.path.join(os.getcwd(),"inference_results_twin", f"result_{model_name}.jsonl")
    
    print(f"Using batch size: {args.batch_size}, num_workers: {args.num_workers}")
    print(f"Starting inference on {len(coco_dataset)} samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, desc="Generating captions"):
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"]
            
            # Generate captions
            try:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids,
                        images,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        greedy=True
                    )
                pred_captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
            except Exception as e:
                print(f"Error during generation: {e}")
                pred_captions = [""] * len(batch["image_ids"])
            
            # Write records to file
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
