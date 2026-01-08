import torch
import os
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from data.collators import VQADualCollator
from data.processors import get_image_string
import re


class OCRVQAVanillaGenerationDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, total_samples):
        print("Preparing OCR-VQA evaluation dataset...")
        self.samples = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        
        max_iter = min(total_samples, len(dataset)) if total_samples > 0 else len(dataset)
        sample_count = 0
        
        for i in tqdm(range(len(dataset)), desc="Loading OCR-VQA samples"):
            if sample_count >= max_iter:
                break
                
            sample = dataset[i]

            # so here one sample has multiple instances of Q&A pairs
            questions = sample.get('questions', [])
            answers = sample.get('answers', [])
            image = sample.get('image')
            image_id = sample.get('image_id', str(i))
            
            if image is None or not questions or not answers:
                continue
            
            for qa_idx in range(min(len(questions), len(answers))):
                if sample_count >= max_iter:
                    break
                    
                question = questions[qa_idx]
                answer = answers[qa_idx]
                
                if not question or not answer:
                    continue
                
                self.samples.append({
                    "image": image,
                    "gt_answer": answer,
                    "question": question,
                    "filename": sample.get("image_id", f"{image_id}_{qa_idx}"),
                    "image_id": f"{image_id}_{qa_idx}" # custom unique id for each qa pair
                })
                sample_count += 1
        
        print(f"Loaded {len(self.samples)} Q&A pairs for OCR-VQA evaluation")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image, splitted_image_ratio = self.image_processor(image)
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        question = item.get("question", "")
        messages = [
            {"role": "user", "content": image_string + question}
        ]
        
        # Tokenize with add_generation_prompt=True to add assistant turn prefix
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True, # Critical: adds "<|im_start|>assistant\n" prefix
            return_dict=True,
        )
        
        input_ids = torch.tensor(conv_ids["input_ids"])
        attention_mask = torch.tensor(conv_ids["attention_mask"])
        
        return {
            "images": [processed_image],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gt_answer": item["gt_answer"],
            "question": item.get("question", ""),
            "filename": item["filename"],
            "image_id": item["image_id"],
        }


class OCRVQAVanillaGenerationCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # Filter None
        batch = [x for x in batch if x is not None]
        if not batch: 
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "gt_answers": [],
                "questions": [],
                "filenames": [],
                "image_ids": []
            }

        # Filter too long
        batch = [x for x in batch if len(x["input_ids"]) <= self.max_length]
        if not batch: 
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "gt_answers": [],
                "questions": [],
                "filenames": [],
                "image_ids": []
            }

        gt_answers = [item["gt_answer"] for item in batch]
        questions = [item["question"] for item in batch]
        filenames = [item["filename"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        
        batch_dict = {k: [item[k] for item in batch] for k in batch[0] if k not in ["gt_answer", "question", "filename", "image_id"]}
        
        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        images = batch_dict["images"]
        
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            # Left pad input_ids
            padded_input_ids.append(torch.nn.functional.pad(
                ids, (pad_len, 0), value=self.tokenizer.pad_token_id
            ))
            # Left pad attention_mask
            padded_attention_mask.append(torch.nn.functional.pad(
                mask, (pad_len, 0), value=0
            ))
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "images": images,
            "gt_answers": gt_answers,
            "questions": questions,
            "filenames": filenames,
            "image_ids": image_ids
        }


class OCRVQADualTowerGenerationDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, total_samples):
        print("Preparing OCR-VQA Dual Tower evaluation dataset...")
        self.samples = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        
        max_iter = min(total_samples, len(dataset)) if total_samples > 0 else len(dataset)
        sample_count = 0
        
        for i in tqdm(range(len(dataset)), desc="Loading OCR-VQA Dual Tower samples"):
            if sample_count >= max_iter:
                break
                
            sample = dataset[i]
            
            questions = sample.get('questions', [])
            answers = sample.get('answers', [])
            image = sample.get('image')
            image_id = sample.get('image_id', str(i))
            
            if image is None or not questions or not answers:
                continue
            
            # Create one evaluation sample per Q&A pair
            for qa_idx in range(min(len(questions), len(answers))):
                if sample_count >= max_iter:
                    break
                    
                question = questions[qa_idx]
                answer = answers[qa_idx]
                
                if not question or not answer:
                    continue
                
                self.samples.append({
                    "image": image,
                    "gt_answer": answer,
                    "question": question,
                    "filename": sample.get("image_id", f"{image_id}_{qa_idx}"),
                    "image_id": f"{image_id}_{qa_idx}"  # Unique ID for each Q&A pair
                })
                sample_count += 1
        
        print(f"Loaded {len(self.samples)} Q&A pairs for OCR-VQA Dual Tower evaluation")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image, splitted_image_ratio = self.image_processor(image)
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        question = item.get("question", "")
        messages = [
            {"role": "user", "content": image_string + question}
        ]
        
        # Tokenize with add_generation_prompt=True to add assistant turn prefix
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True, # Critical: adds "<|im_start|>assistant\n" prefix
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
            "gt_answer": item["gt_answer"],
            "question": item.get("question", ""),
            "filename": item["filename"],
            "image_id": item["image_id"],
        }


class OCRVQADualTowerGenerationCollator(VQADualCollator):
    def _center_pad_batch(self, batch, split_points):
        batch["input_ids"], max_left = self._split_and_center_pad(
            batch["input_ids"], split_points, self.tokenizer.pad_token_id
        )
        batch["attention_mask"], _ = self._split_and_center_pad(
            batch["attention_mask"], split_points, 0
        )
        # No labels for generation
        return max_left
    
    def prepare_batch(self, batch, max_length=None):
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
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "images": [],
                "last_img_idx": 0,
                "gt_answers": [],
                "questions": [],
                "filenames": [],
                "image_ids": []
            }
        
        # Extract metadata before collating
        gt_answers = [item.pop("gt_answer") for item in batch]
        questions = [item.pop("question") for item in batch]
        filenames = [item.pop("filename") for item in batch]
        image_ids = [item.pop("image_id") for item in batch]
        
        # Use our custom prepare_batch (not parent's)
        collated = self.prepare_batch(batch, max_length=self.max_length)
        
        # Add metadata back
        collated["gt_answers"] = gt_answers
        collated["questions"] = questions
        collated["filenames"] = filenames
        collated["image_ids"] = image_ids
        
        return collated


def normalize_text_for_accuracy(text):
    if not isinstance(text, str):
        text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = " ".join(text.split())
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def compute_ocrvqa_accuracy(model, ocrvqa_loader, device, tokenizer, max_new_tokens=50, max_samples=5000, log_samples_path=None, is_dual_tower=False):
    model.eval()
    correct = 0
    total = 0
    sample_count = 0
    samples_to_log = []
    
    with torch.no_grad():
        for batch in tqdm(ocrvqa_loader, desc="OCR-VQA Accuracy Eval", leave=False):
            if len(batch["images"]) == 0:
                continue
            
            if sample_count >= max_samples:
                break
            
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_gt_answers = batch["gt_answers"]
            batch_image_ids = batch["image_ids"]
            batch_questions = batch.get("questions", [""] * len(batch_image_ids))
            batch_filenames = batch.get("filenames", [""] * len(batch_image_ids))
            
            # Helper to check if model is DualTowerVLM or Vanilla VisionLanguageModel
            is_dual_tower = hasattr(model, "right_tower") or (hasattr(model, "module") and hasattr(model.module, "right_tower"))
            
            generate_kwargs = {
                "input_ids": input_ids,
                "images": images,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "greedy": True
            }

            if is_dual_tower:
                generate_kwargs["last_img_idx"] = batch["last_img_idx"]
            
            try:
                generated_ids = model.generate(**generate_kwargs)
                if generated_ids.numel() == 0 or generated_ids.shape[1] == 0:
                    print(f"Warning: generated_ids is empty! Shape: {generated_ids.shape}")
                    pred_answers = [""] * len(batch_image_ids)
                else:
                    pred_answers = tokenizer.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                pred_answers = [""] * len(batch_image_ids)
            
            for i, image_id in enumerate(batch_image_ids):
                gt_answer = batch_gt_answers[i]
                pred_answer = pred_answers[i]
                
                # Normalize both answers for comparison
                gt_normalized = normalize_text_for_accuracy(gt_answer)
                pred_normalized = normalize_text_for_accuracy(pred_answer)
                
                # String matching: exact match only
                # Empty predictions should never be correct
                is_correct = (len(pred_normalized) > 0 and 
                             len(gt_normalized) > 0 and
                             pred_normalized == gt_normalized)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Collect first 50 samples for logging
                if len(samples_to_log) < 50:
                    samples_to_log.append({
                        "image_id": image_id,
                        "filename": batch_filenames[i] if isinstance(batch_filenames, list) else str(i),
                        "question": batch_questions[i] if isinstance(batch_questions, list) else "",
                        "pred_answer": pred_answer,
                        "gt_answer": gt_answer,
                        "pred_normalized": pred_normalized,
                        "gt_normalized": gt_normalized,
                        "is_correct": is_correct
                    })
            
            sample_count += len(batch_image_ids)
    
    # Save samples to JSONL if path provided
    if log_samples_path and samples_to_log:
        os.makedirs(os.path.dirname(log_samples_path), exist_ok=True)
        with open(log_samples_path, 'w', encoding='utf-8') as f:
            for sample in samples_to_log:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples_to_log)} generation samples to {log_samples_path}")
    
    # Compute accuracy
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return accuracy

