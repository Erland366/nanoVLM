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
from data.collators import VQADualCollator

FOLDER_PATH = "inference_results_dual_tower_frope_sanity"

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
    parser.add_argument("--dataset", type=str, default="patrickamadeus/coco_caption_val_unique",
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
    Dataset for COCO caption generation.
    Creates prompts with only user message (no assistant response) for generation.
    """
    
    def __init__(self, dataset, tokenizer, image_processor, prompt, mp_image_token_length, total_samples):
        print("Preparing dataset...")
        self.samples = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prompt = prompt
        
        max_iter = min(total_samples, len(dataset)) if total_samples > 0 else len(dataset)
        for i in tqdm(range(max_iter), desc="Loading samples"):
            sample = dataset[i]
            
            self.samples.append({
                "image": sample["image"],
                "gt_caption": sample.get("caption", ""),
                "filename": sample.get("filename", str(i)),
                "image_id": sample.get("image_id", i)
            })
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Process sample for generation: user prompt only, no assistant response."""
        from data.processors import get_image_string
        
        item = self.samples[idx]
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        # Handle global image token case
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        # Create image string
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        # Create messages with ONLY user prompt (no assistant response for generation)
        messages = [
            {"role": "user", "content": image_string + self.prompt}
        ]
        
        # Tokenize with add_generation_prompt=True to add assistant turn prefix
        # This ensures the model generates content directly, not the assistant role marker
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,  # Critical: adds "<|im_start|>assistant\n" prefix
            return_dict=True,
        )
        
        input_ids = torch.tensor(conv_ids["input_ids"])
        attention_mask = torch.tensor(conv_ids["attention_mask"])
        
        # Find last image token position
        image_token_id = self.tokenizer.encode(self.tokenizer.image_token, add_special_tokens=False)[0]
        last_image_token_pos = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == image_token_id:
                last_image_token_pos = i
                break
        
        if last_image_token_pos == -1:
            return None
        
        return {
            "images": processed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_image_token_pos": last_image_token_pos,
            "gt_caption": item["gt_caption"],
            "filename": item["filename"],
            "image_id": item["image_id"],
        }


class GenerationCollator(VQADualCollator):
    """
    Collator for generation that extends VQADualCollator.
    Adds metadata fields (gt_captions, filenames, image_ids) to the batch.
    Handles generation-only data (no labels).
    """
    
    def _center_pad_batch(self, batch, split_points):
        """Apply center padding to batch components (no labels for generation)."""
        batch["input_ids"], max_left = self._split_and_center_pad(
            batch["input_ids"], split_points, self.tokenizer.pad_token_id
        )
        batch["attention_mask"], _ = self._split_and_center_pad(
            batch["attention_mask"], split_points, 0
        )
        # No labels for generation
        return max_left
    
    def prepare_batch(self, batch, max_length=None):
        """
        Prepare a batch with center padding for dual tower generation.
        Similar to parent but without labels.
        """
        # 1) Handle empty batch
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "last_img_idx": 0
            }
        
        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "last_img_idx": 0
            }
        
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        split_points = [
            int(pos.item()) + 1 if hasattr(pos, "item") else int(pos) + 1 
            for pos in batch["last_image_token_pos"]
        ]
        
        # 3) Discard samples that are too long
        max_len = self.max_length if max_length is None else max_length
        batch, split_points = self._discard_samples_that_are_too_long(batch, split_points, max_len)
        
        # 4) Center pad the batch
        if not batch["input_ids"]:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "last_img_idx": 0
            }
        
        max_left = self._center_pad_batch(batch, split_points)
        
        batch_size = len(batch["input_ids"])
        seq_length = batch["input_ids"][0].shape[0]
        
        # Calculate last_img_idx: it's max_left - 1
        last_img_idx = max_left - 1 if max_left > 0 else -1
        
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "img_region_mask": torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1) < max_left,
            "last_img_idx": last_img_idx,
            "images": batch["images"],
        }
    
    def _discard_samples_that_are_too_long(self, batch, split_points, max_length):
        """Filter out samples that exceed max_length (no labels version)."""
        filtered = []
        filtered_split_points = []
        for i in range(len(batch["input_ids"])):
            seq = batch["input_ids"][i]
            if len(seq) <= max_length:
                filtered.append({k: batch[k][i] for k in batch})
                filtered_split_points.append(split_points[i])
        if not filtered:
            return {k: [] for k in batch}, []
        out = {k: [d[k] for d in filtered] for k in batch}
        return out, filtered_split_points
    
    def __call__(self, batch):
        """Collate batch with center padding and preserve metadata."""
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "last_img_idx": 0,
                "gt_captions": [],
                "filenames": [],
                "image_ids": []
            }
        
        # Extract metadata before collating
        gt_captions = [item.pop("gt_caption") for item in batch]
        filenames = [item.pop("filename") for item in batch]
        image_ids = [item.pop("image_id") for item in batch]
        
        # Use our custom prepare_batch (not parent's)
        collated = self.prepare_batch(batch, max_length=self.max_length)
        
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
        
        # Try new dual tower format first
        dual_tower_weights = os.path.join(source, "dual_tower_model.pt")
        safetensors_weights = os.path.join(source, "model.safetensors")
        
        if os.path.exists(dual_tower_weights):
            weights_path = dual_tower_weights
            is_dual_tower_checkpoint = True
            print(f"Loading dual tower checkpoint: {dual_tower_weights}")
        elif os.path.exists(safetensors_weights):
            weights_path = safetensors_weights
            is_dual_tower_checkpoint = False
            print(f"Loading vanilla VLM checkpoint: {safetensors_weights}")
        else:
            raise FileNotFoundError(f"No model weights found in {source}. Expected 'dual_tower_model.pt' or 'model.safetensors'")
    else:
        config_path = hf_hub_download(repo_id=source, filename="config.json")
        
        # Try to download dual tower format first
        try:
            weights_path = hf_hub_download(repo_id=source, filename="dual_tower_model.pt")
            is_dual_tower_checkpoint = True
            print(f"Downloaded dual tower checkpoint from HF")
        except:
            # Fall back to safetensors format
            weights_path = hf_hub_download(repo_id=source, filename="model.safetensors")
            is_dual_tower_checkpoint = False
            print(f"Downloaded vanilla VLM checkpoint from HF")
    
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
    
    # Load weights based on checkpoint format
    if is_dual_tower_checkpoint:
        # Load dual tower checkpoint (full model)
        print("Loading full DualTowerVLM state dict...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load vanilla VLM checkpoint (only into left_tower)
        print("Loading vanilla VLM weights into left_tower...")
        from safetensors.torch import load_model as load_safetensors
        load_safetensors(model.left_tower, weights_path)
    
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

    # Load dataset - now using pre-filtered unique dataset
    from datasets import load_dataset
    print(f"Loading pre-filtered unique dataset from: {args.dataset}")
    hf_dataset = load_dataset(args.dataset, split="train", streaming=False)
    print(f"Dataset loaded with {len(hf_dataset)} unique samples")
    
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
    os.makedirs(os.path.join(os.getcwd(), FOLDER_PATH), exist_ok=True)
    output_file = os.path.join(os.getcwd(), FOLDER_PATH, f"result_{model_name}.jsonl")
    
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





