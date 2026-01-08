import torch
from PIL import Image
from torch.utils.data import Dataset
from data.processors import get_image_string
import logging


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, relevance_min_rating=1, image_correspondence_min_rating=1, visual_dependency_min_rating=1, formatting_min_rating=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, splitted_image_counts):
        messages = []
        for index, text in enumerate(item['texts']):
            try:
                if item.get('relevance_ratings') is not None and item['relevance_ratings'][index] is not None and item['relevance_ratings'][index] < self.relevance_min_rating:
                    continue
                if item.get('image_correspondence_ratings') is not None and item['image_correspondence_ratings'][index] is not None and item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                    continue
                if item.get('visual_dependency_ratings') is not None and item['visual_dependency_ratings'][index] is not None and item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                    continue
                if item.get('formatting_ratings') is not None and item['formatting_ratings'][index] is not None and item['formatting_ratings'][index] < self.formatting_min_rating:
                    continue
            except Exception as e:
                logging.warning(f"Error processing item: {item}, index: {index}: {e}")

            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        if len(messages) == 0:
            return messages

        # Safety check to ensure no image tokens are present in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning(f"Found and removed an image token in the {msg['role']} text before adding the image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if len(splitted_image_counts) > 0:
            image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, splitted_image_count = self.image_processor(image)
                if not hasattr(self.tokenizer, "global_image_token") and splitted_image_count[0]*splitted_image_count[1] == len(processed_image) - 1:
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts


    def _prepare_inputs_and_loss_mask(self, messages):
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
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def iter_for_worker(self, worker_id, num_workers):
        # dataset = split_dataset_by_node(self.dataset, rank=worker_id, world_size=num_workers)
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        # Handle images (should be a list)
        if item['images'] is None:
            images_data = []
        else:
            images_data = item['images']
            if not isinstance(images_data, list):
                images_data = [images_data]

        processed_images = []
        splitted_image_counts = []
        if images_data: # Only process if there are images
            processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels


class VQADualDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()
        
    def __len__(self):
        return len(self.dataset)
    
    def _get_prefix_len(self):
        """Calculate the prefix length for assistant responses in the chat template."""
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
        """Process a single data item: image + text to model inputs."""
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        # Handle global image token case
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        messages = [
            {"role": "user", "content": image_string + "Describe the image."},
            {"role": "assistant", "content": item['caption']}
        ]
        
        # Prepare the FULL sequence
        input_ids, loss_mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        
        # Find the last image token position
        image_token_id = self.tokenizer.encode(self.tokenizer.image_token, add_special_tokens=False)[0]
        last_image_token_pos = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == image_token_id:
                last_image_token_pos = i
                break
        
        # If no image token found, return None (will be filtered out)
        if last_image_token_pos == -1:
            return None
        
        # Get labels with proper shifting for causal LM
        labels = self._get_labels(input_ids, loss_mask)
        
        return {
            "images": processed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "last_image_token_pos": last_image_token_pos,
        }
    
    def _prepare_inputs_and_loss_mask(self, messages):
        """
        Prepare input_ids, loss mask, and attention mask from messages.
        Loss mask is 1 for assistant responses, 0 elsewhere.
        """
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
                mask[start:end] = [1] * (end - start)

            cursor += seg_len
        
        return (
            torch.tensor(conv_ids["input_ids"]), 
            torch.tensor(mask).to(torch.bool), 
            torch.tensor(conv_ids["attention_mask"])
        )
    
    def _get_labels(self, input_ids, mask):
        """Create labels with -100 for non-target positions, shifted for causal LM."""
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target
        return labels


class OCRVQADualDataset(Dataset):
    """
    Dataset for OCR-VQA training in dual tower format.
    Each sample in the dataset has:
    - image: PIL Image
    - questions: list of questions (2-7 questions per image)
    - answers: list of answers (2-7 answers per image)
    - image_id: identifier
    
    We create one training example per Q&A pair, so each image with N Q&A pairs
    becomes N training examples.
    """
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()
        
        # Expand dataset: create one entry per Q&A pair
        # self.samples is a list of (dataset_idx, qa_idx) tuples
        self.samples = []
        for dataset_idx in range(len(self.dataset)):
            item = self.dataset[dataset_idx]
            questions = item.get('questions', [])
            answers = item.get('answers', [])
            
            # Create one sample per Q&A pair
            num_pairs = min(len(questions), len(answers))
            for qa_idx in range(num_pairs):
                if questions[qa_idx] and answers[qa_idx]:  # Skip empty Q&A
                    self.samples.append((dataset_idx, qa_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def _get_prefix_len(self):
        """Calculate the prefix length for assistant responses in the chat template."""
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}], 
            tokenize=False, 
            add_special_tokens=False
        )
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))
    
    def __getitem__(self, idx):
        dataset_idx, qa_idx = self.samples[idx]
        item = self.dataset[dataset_idx]
        return self._process_data(item, qa_idx)
    
    def _process_data(self, item, qa_idx):
        """Process a single data item: image + question + answer to model inputs."""
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        processed_image, splitted_image_ratio = self.image_processor(image)
        
        # Handle global image token case
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]
        
        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        # Get questions and answers - they are sequences
        questions = item.get('questions', [])
        answers = item.get('answers', [])
        
        # Get the specific Q&A pair
        if qa_idx >= len(questions) or qa_idx >= len(answers):
            return None
        
        question = questions[qa_idx]
        answer = answers[qa_idx]
        
        # Skip if question or answer is empty
        if not question or not answer:
            return None
        
        messages = [
            {"role": "user", "content": image_string + question},
            {"role": "assistant", "content": answer}
        ]
        
        # Prepare the FULL sequence
        input_ids, loss_mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        
        # Find the last image token position
        image_token_id = self.tokenizer.encode(self.tokenizer.image_token, add_special_tokens=False)[0]
        last_image_token_pos = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == image_token_id:
                last_image_token_pos = i
                break
        
        # If no image token found, return None (will be filtered out)
        if last_image_token_pos == -1:
            return None
        
        # Get labels with proper shifting for causal LM
        labels = self._get_labels(input_ids, loss_mask)
        
        return {
            "images": processed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "last_image_token_pos": last_image_token_pos,
        }
    
    def _prepare_inputs_and_loss_mask(self, messages):
        """
        Prepare input_ids, loss mask, and attention mask from messages.
        Loss mask is 1 for assistant responses, 0 elsewhere.
        """
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
                mask[start:end] = [1] * (end - start)

            cursor += seg_len
        
        return (
            torch.tensor(conv_ids["input_ids"]), 
            torch.tensor(mask).to(torch.bool), 
            torch.tensor(conv_ids["attention_mask"])
        )
    
    def _get_labels(self, input_ids, mask):
        """Create labels with -100 for non-target positions, shifted for causal LM."""
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target
        return labels


class COCOCaptionsVanillaDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        mp_image_token_length,
        relevance_min_rating=None,
        image_correspondence_min_rating=None,
        visual_dependency_min_rating=None,
        formatting_min_rating=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()

        # these won't be used, just to maintain uniformity with the vanilla training code
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
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
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image, splitted_image_ratio = self.image_processor(image)

        # handle global imag etoken case
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]

        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        messages = [
            {"role": "user", "content": image_string + "Describe the image."},
            {"role": "assistant", "content": item['caption']}
        ]
        
        input_ids, loss_mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, loss_mask)
        
        return {
            "images": [processed_image],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # locate assistant token turn and flip its mask to 1 if its more than prefix len ("assistant\n" token)
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)

            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]), 
            torch.tensor(mask).to(torch.bool), 
            torch.tensor(conv_ids["attention_mask"])
        )
    
    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100
        return labels


class OCRVQAVanillaDataset(Dataset):
    """
    Dataset for OCR-VQA training.
    Each sample in the dataset has:
    - image: PIL Image
    - questions: list of questions (2-7 questions per image)
    - answers: list of answers (2-7 answers per image)
    - image_id: identifier
    
    We create one training example per Q&A pair, so each image with N Q&A pairs
    becomes N training examples.
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        mp_image_token_length,
        relevance_min_rating=None,
        image_correspondence_min_rating=None,
        visual_dependency_min_rating=None,
        formatting_min_rating=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()

        # these won't be used, just to maintain uniformity with the vanilla training code
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating

        # Expand dataset: create one entry per Q&A pair
        # self.samples is a list of (dataset_idx, qa_idx) tuples
        self.samples = []
        for dataset_idx in range(len(self.dataset)):
            item = self.dataset[dataset_idx]
            questions = item.get('questions', [])
            answers = item.get('answers', [])
            
            # Create one sample per Q&A pair
            num_pairs = min(len(questions), len(answers))
            for qa_idx in range(num_pairs):
                if questions[qa_idx] and answers[qa_idx]:  # Skip empty Q&A
                    self.samples.append((dataset_idx, qa_idx))

    def __len__(self):
        return len(self.samples)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}], 
            tokenize=False, 
            add_special_tokens=False
        )
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))
    
    def __getitem__(self, idx):
        dataset_idx, qa_idx = self.samples[idx]
        item = self.dataset[dataset_idx]
        return self._process_data(item, qa_idx)
    
    def _process_data(self, item, qa_idx):
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image, splitted_image_ratio = self.image_processor(image)

        # handle global image token case
        if (not hasattr(self.tokenizer, "global_image_token") and 
            splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1):
            processed_image = processed_image[1:]

        image_string = get_image_string(
            self.tokenizer, 
            [splitted_image_ratio], 
            self.mp_image_token_length
        )
        
        # Get questions and answers - they are sequences
        questions = item.get('questions', [])
        answers = item.get('answers', [])
        
        # Get the specific Q&A pair
        if qa_idx >= len(questions) or qa_idx >= len(answers):
            return None
        
        question = questions[qa_idx]
        answer = answers[qa_idx]
        
        # Skip if question or answer is empty
        if not question or not answer:
            return None
        
        messages = [
            {"role": "user", "content": image_string + question},
            {"role": "assistant", "content": answer}
        ]
        
        input_ids, loss_mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, loss_mask)
        
        return {
            "images": [processed_image],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # locate assistant token turn and flip its mask to 1 if its more than prefix len ("assistant\n" token)
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)

            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]), 
            torch.tensor(mask).to(torch.bool), 
            torch.tensor(conv_ids["attention_mask"])
        )
    
    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100
        return labels
