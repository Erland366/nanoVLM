"""
Synthetic dataset for torch.compile debugging.
Generates random tensors matching expected shapes - no network/disk I/O.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticVLMDataset(Dataset):
    """
    Generates synthetic VLM training data for debugging torch.compile.

    No network calls, no disk I/O - instant startup.
    """

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 256,
        vocab_size: int = 49218,
        image_size: int = 512,
        num_channels: int = 3,
        images_per_sample: int = 1,
        image_token_id: int = 49152,  # Default for SmolLM2 + extra tokens
        mp_image_token_length: int = 64,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.images_per_sample = images_per_sample
        self.image_token_id = image_token_id
        self.mp_image_token_length = mp_image_token_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input_ids
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))

        # Insert image tokens at the beginning (after potential BOS)
        num_image_tokens = self.images_per_sample * self.mp_image_token_length
        if num_image_tokens < self.seq_len - 10:
            input_ids[1:1 + num_image_tokens] = self.image_token_id

        # Labels: same as input_ids but shifted (for causal LM)
        # Use -100 for image token positions (don't compute loss there)
        labels = input_ids.clone()
        labels[input_ids == self.image_token_id] = -100

        # Attention mask: all 1s (no padding in synthetic data)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)

        # Generate random images
        images = [
            torch.randn(1, self.num_channels, self.image_size, self.image_size)
            for _ in range(self.images_per_sample)
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
        }


def get_synthetic_dataloader(
    batch_size: int = 2,
    num_samples: int = 100,
    seq_len: int = 256,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create a synthetic dataloader for debugging."""
    dataset = SyntheticVLMDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """Collate function for synthetic dataset."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Images: list of lists
    images = [item["images"] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": images,
    }
