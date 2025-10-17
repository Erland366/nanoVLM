import torch
from PIL import Image
from torch.utils.data import Dataset
from data.processors import get_image_string
import logging
from data.collators import BaseCollator

class AOKVQADataset(Dataset):
    """A-OKVQA Dataset - handles multiple choice visual question answering"""
    
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
        return 0

    def _format_question_with_choices(self, question, choices):
        """Format question and choices into the specified format"""
        formatted_text = f"Question: {question}\nChoices:\n"
        
        # Add choices with letters a, b, c, d
        choice_letters = ['a', 'b', 'c', 'd']
        for i, choice in enumerate(choices):
            if i < len(choice_letters):
                formatted_text += f"{choice_letters[i]}. {choice}\n"
        
        formatted_text += "\nAnswer:"
        return formatted_text

    def _get_ground_truth_letter(self, correct_choice_idx):
        """Convert correct choice index to letter (0->a, 1->b, 2->c, 3->d)"""
        choice_letters = ['a', 'b', 'c', 'd']
        if 0 <= correct_choice_idx < len(choice_letters):
            return choice_letters[correct_choice_idx]
        else:
            logging.warning(f"Invalid correct_choice_idx: {correct_choice_idx}")
            return 'a'  # Default fallback

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        """Process A-OKVQA item: {image, question, choices, correct_choice_idx}"""
        
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
        
        # Format the question with choices
        question = item['question']
        choices = item['choices']
        correct_choice_idx = item['correct_choice_idx']
        
        # Create the formatted question text
        formatted_question = self._format_question_with_choices(question, choices)
        
        # Get the ground truth answer (just the letter)
        ground_truth = self._get_ground_truth_letter(correct_choice_idx)

        # Create messages in the expected format
        messages = [
            {"role": "user", "content": image_string + formatted_question},
            {"role": "assistant", "content": ground_truth}
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
        
        mask_sum = sum(mask)
        if mask_sum == 0:
            logging.warning("No valid tokens in mask - this will cause NaN loss!")
        
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
        
        # Check if we have any valid labels
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            logging.warning("No valid labels found in A-OKVQA sample - all labels are -100")
        
        return labels

class AOKVQACollator(BaseCollator):
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