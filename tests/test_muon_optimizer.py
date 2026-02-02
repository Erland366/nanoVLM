from __future__ import annotations

import pytest
import torch

from models.config import TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel
from utils.optimizers import build_optimizer


class _StubTokenizer:
    def __init__(self, *, image_token_id: int = 0, pad_token_id: int = 0):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = None


def _tiny_vlm_config(*, vocab_size: int = 256, max_seq_len: int = 64) -> VLMConfig:
    cfg = VLMConfig()
    cfg.vlm_load_backbone_weights = False
    cfg.momh_enabled = False
    cfg.activation_checkpointing = False

    cfg.vit_img_size = 32
    cfg.vit_patch_size = 16
    cfg.vit_hidden_dim = 32
    cfg.vit_inter_dim = 4 * cfg.vit_hidden_dim
    cfg.vit_n_heads = 4
    cfg.vit_n_blocks = 1

    cfg.lm_hidden_dim = 32
    cfg.lm_inter_dim = 4 * cfg.lm_hidden_dim
    cfg.lm_n_heads = 4
    cfg.lm_n_kv_heads = 2
    cfg.lm_n_blocks = 1
    cfg.lm_max_length = max_seq_len
    cfg.lm_use_tokens = False
    cfg.lm_tie_weights = True

    cfg.extra_token_amount = 0
    cfg.lm_base_vocab_size = vocab_size
    cfg.lm_vocab_size = vocab_size

    cfg.mp_pixel_shuffle_factor = 2
    cfg.mp_image_token_length = 4
    return cfg


def _dion_available() -> bool:
    try:
        import dion  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _dion_available(), reason="requires dion (microsoft/dion) package")
def test_muon_groups_exclude_embeddings_and_biases():
    cfg = _tiny_vlm_config()
    tokenizer = _StubTokenizer(image_token_id=0, pad_token_id=0)
    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer).cpu()

    train_cfg = TrainConfig()
    train_cfg.optimizer = "muon"
    train_cfg.lr_mp = 0.0
    train_cfg.lr_vision_backbone = 0.0
    train_cfg.lr_language_backbone = 1e-3

    opt = build_optimizer(model, train_cfg)

    token_embedding_weight = model.decoder.token_embedding.weight
    linear_weight_params = {
        m.weight
        for m in model.decoder.modules()
        if isinstance(m, torch.nn.Linear) and m.weight is not None
    }
    linear_weight_params.discard(token_embedding_weight)

    muon_params = set()
    for group in opt.param_groups:
        if group.get("algorithm") != "muon":
            continue
        for p in group["params"]:
            muon_params.add(p)
            assert p.ndim == 2
            assert p is not token_embedding_weight
            assert p in linear_weight_params

    assert muon_params, "Expected at least one Muon parameter in optimizer groups."


@pytest.mark.skipif(not _dion_available(), reason="requires dion (microsoft/dion) package")
def test_muon_optimizer_step_runs():
    device = torch.device("cpu")
    cfg = _tiny_vlm_config(vocab_size=128, max_seq_len=32)
    tokenizer = _StubTokenizer(image_token_id=0, pad_token_id=0)
    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer).to(device)
    model.train()

    train_cfg = TrainConfig()
    train_cfg.optimizer = "muon"
    train_cfg.lr_mp = 0.0
    train_cfg.lr_vision_backbone = 0.0
    train_cfg.lr_language_backbone = 1e-3
    train_cfg.max_grad_norm = None

    opt = build_optimizer(model, train_cfg)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    targets = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    images = [[] for _ in range(batch_size)]

    _, loss = model(input_ids=input_ids, images=images, attention_mask=attention_mask, targets=targets)
    assert loss is not None
    assert torch.isfinite(loss).item()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

