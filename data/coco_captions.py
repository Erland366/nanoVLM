import torch
from PIL import Image
from torch.utils.data import Dataset
from data.processors import get_image_string
import logging
from data.collators import BaseCollator

class COCOCollator(BaseCollator):
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)
    
    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"] = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]
    
    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
    
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
