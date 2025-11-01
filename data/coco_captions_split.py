import torch
from torch.utils.data import Dataset
from data.processors import get_image_string
from data.collators import BaseCollator

class COCOCollator(BaseCollator):
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)
    
    def _align_image_tokens(self, batch):
        """Align image tokens across batch without changing end positions"""
        split_positions = []
        for ids in batch["input_ids"]:
            image_token_mask = (ids == self.tokenizer.image_token_id)
            positions = torch.where(image_token_mask)[0]
            if len(positions) > 0:
                split_positions.append(positions[-1].item() + 1)
            else:
                split_positions.append(0)

        max_split_pos = max(split_positions)
        aligned_batch = {"input_ids": [], "labels": [], "attention_mask": [], "images": batch["images"]}

        for i, split_pos in enumerate(split_positions):
            pad_amount = max_split_pos - split_pos
            if pad_amount > 0:
                aligned_batch["input_ids"].append(
                    torch.nn.functional.pad(batch["input_ids"][i], (pad_amount, 0), value=self.tokenizer.pad_token_id)
                )
                aligned_batch["labels"].append(
                    torch.nn.functional.pad(batch["labels"][i], (pad_amount, 0), value=-100)
                )
                aligned_batch["attention_mask"].append(
                    torch.nn.functional.pad(batch["attention_mask"][i], (pad_amount, 0), value=0)
                )
            else:
                aligned_batch["input_ids"].append(batch["input_ids"][i])
                aligned_batch["labels"].append(batch["labels"][i])
                aligned_batch["attention_mask"].append(batch["attention_mask"][i])

        return aligned_batch
    
    def _pad_batch(self, batch, max_length):
        """
        For each sequence:
        - Keep the image region (0:split_pos) fixed (already aligned)
        - Left-pad only the text region (split_pos:) so text ends at right edge
        Final shape: [ ...pad... IMG IMG IMG | ...pad... TEXT TEXT TEXT ]
        """
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for ids, labels, attn in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
            # Find split (right after the last image token)
            image_token_mask = (ids == self.tokenizer.image_token_id)
            positions = torch.where(image_token_mask)[0]
            split_pos = positions[-1].item() + 1 if len(positions) > 0 else 0

            # Split sequence into image and text parts
            image_part = ids[:split_pos]
            text_part = ids[split_pos:]
            label_text = labels[split_pos:]
            attn_text = attn[split_pos:]

            # Compute how much we can pad before text so total length == max_length
            total_len = len(ids)
            pad_needed = max_length - total_len
            pad_needed = max(pad_needed, 0)

            # Left-pad only the text portion
            if pad_needed > 0:
                pad_ids = torch.full((pad_needed,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                pad_labels = torch.full((pad_needed,), -100, dtype=labels.dtype)
                pad_attn = torch.zeros((pad_needed,), dtype=attn.dtype)

                # Insert padding before text
                text_part = torch.cat([pad_ids, text_part])
                label_text = torch.cat([pad_labels, label_text])
                attn_text = torch.cat([pad_attn, attn_text])

            # Rebuild full sequence: [image_part | text_part]
            new_ids = torch.cat([image_part, text_part])
            new_labels = torch.cat([labels[:split_pos], label_text])
            new_attn = torch.cat([attn[:split_pos], attn_text])

            # If still short (can happen if no text region), right-pad to max_length
            # TODO: confirm if this is correct for no text region case! especially to _get_labels
            pad_tail = max_length - len(new_ids)
            if pad_tail > 0:
                new_ids = torch.nn.functional.pad(new_ids, (0, pad_tail), value=self.tokenizer.pad_token_id)
                new_labels = torch.nn.functional.pad(new_labels, (0, pad_tail), value=-100)
                new_attn = torch.nn.functional.pad(new_attn, (0, pad_tail), value=0)

            padded_input_ids.append(new_ids)
            padded_labels.append(new_labels)
            padded_attention_masks.append(new_attn)

        batch["input_ids"] = padded_input_ids
        batch["labels"] = padded_labels
        batch["attention_mask"] = padded_attention_masks
    
    def __call__(self, batch):
        # First, align image tokens across samples
        batch = self.prepare_batch(batch, max_length=None)  # Don't pad yet
        
        if len(batch["input_ids"]) == 0:
            return batch
        
        # Align image tokens by padding left side
        batch = self._align_image_tokens(batch)
        
        # Now pad right side to max_length
        if self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        
        self._pad_batch(batch, max_len)
        
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }
    
class COCODataset(Dataset):
    """Simple COCO Captions Dataset - converts image+caption to VQA format"""
    
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()
        
    def __len__(self):
        return len(self.dataset)
    
    def _get_prefix_len(self):
        """Calculate prefix length for label masking (same as BaseDataset)"""
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}], 
            tokenize=False, 
            add_special_tokens=False
        )
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)
    
    def _process_data(self, item):
        """Process COCO item: {image: PIL.Image, caption: str} -> VQA format"""
        
        # Process the image
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        # Handle global image token removal if tokenizer doesn't support it
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        # Create image string
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        # Create simple conversation: user asks "Describe the image.", assistant gives caption
        messages = [
            {"role": "user", "content": image_string + "Describe the image."},
            {"role": "assistant", "content": item['caption']}
        ]
        
        # Prepare inputs and labels
        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)
        
        return {
            "images": [processed_image],  # Wrap in list for consistency
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _prepare_inputs_and_loss_mask(self, messages):
        """Same logic as BaseDataset for input preparation"""
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return (
            torch.tensor(conv_ids["input_ids"]), 
            torch.tensor(mask).to(torch.bool), 
            torch.tensor(conv_ids["attention_mask"])
        )
    
    def _get_labels(self, input_ids, mask):
        """Same logic as BaseDataset for label creation"""
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target
        return labels