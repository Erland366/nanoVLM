"""
CUDA-only integration checks for Mixture of Modality Heads (MoMH).

These are skipped by default. Enable them with:
  RUN_CUDA_MOMH_TESTS=1 pytest -q test_momh.py
"""

import os

import pytest
import torch

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CUDA_MOMH_TESTS") != "1",
    reason="Set RUN_CUDA_MOMH_TESTS=1 to run CUDA MoMH integration tests",
)


class _DummyTokenizer:
    def __init__(self, *, image_token_id: int, pad_token_id: int, eos_token_id: int):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def cfg():
    cfg = VLMConfig()
    cfg.momh_enabled = True
    cfg.momh_head_pct_vision = 0.4
    cfg.momh_head_pct_text = 0.4
    cfg.lm_max_length = 256
    return cfg


def _create_dummy_batch(cfg, batch_size, seq_len, num_images_per_sample, tokenizer, device):
    total_images = batch_size * num_images_per_sample
    images = torch.randn(total_images, 3, cfg.vit_img_size, cfg.vit_img_size, device=device)

    # Sample text tokens that cannot collide with image_token_id.
    input_ids = torch.randint(0, cfg.lm_vocab_size - 1, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    for i in range(batch_size):
        pad_len = int(torch.randint(0, seq_len // 4, (1,), device=device).item())
        input_ids[i, :pad_len] = tokenizer.pad_token_id
        attention_mask[i, :pad_len] = 0

        img_token_len = cfg.mp_image_token_length * num_images_per_sample
        input_ids[i, pad_len : pad_len + img_token_len] = tokenizer.image_token_id

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    images_list = [
        [images[i * num_images_per_sample : (i + 1) * num_images_per_sample]]
        for i in range(batch_size)
    ]

    return {
        "input_ids": input_ids,
        "images": images_list,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def test_forward_pass(cfg, device):
    torch.manual_seed(0)

    tokenizer = _DummyTokenizer(
        image_token_id=cfg.lm_vocab_size - 1, pad_token_id=0, eos_token_id=1
    )
    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer).to(device).eval()

    batch = _create_dummy_batch(
        cfg, batch_size=2, seq_len=128, num_images_per_sample=1, tokenizer=tokenizer, device=device
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(
            input_ids=batch["input_ids"],
            images=batch["images"],
            attention_mask=batch["attention_mask"],
            targets=batch["labels"],
        )

    assert logits.shape[0] == 2
    assert torch.isfinite(logits).all().item()
    assert loss is not None
    assert torch.isfinite(loss).item()


def test_backward_pass(cfg, device):
    torch.manual_seed(0)

    tokenizer = _DummyTokenizer(
        image_token_id=cfg.lm_vocab_size - 1, pad_token_id=0, eos_token_id=1
    )
    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer).to(device).train()

    batch = _create_dummy_batch(
        cfg, batch_size=2, seq_len=128, num_images_per_sample=1, tokenizer=tokenizer, device=device
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(
            input_ids=batch["input_ids"],
            images=batch["images"],
            attention_mask=batch["attention_mask"],
            targets=batch["labels"],
        )

    assert loss is not None
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Expected at least one gradient tensor"
    assert all(torch.isfinite(g).all().item() for g in grads)
