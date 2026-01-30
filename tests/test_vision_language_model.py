import torch

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


class _DummyTokenizer:
    def __init__(self, *, vocab_size: int, image_token_id: int, eos_token_id: int, pad_token_id: int):
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


def test_forward_shapes_with_image_placeholders():
    torch.manual_seed(0)

    # Choose sizes so ModalityProjector outputs mp_image_token_length tokens:
    # vit_img_size=64, vit_patch_size=16 -> 4x4 patches => seq=16
    # mp_pixel_shuffle_factor=2 -> (4/2)^2 = 4 tokens
    cfg = VLMConfig(
        vit_model_type="testing",
        vit_patch_size=16,
        vit_hidden_dim=48,
        vit_inter_dim=96,
        vit_n_heads=3,
        vit_n_blocks=1,
        vit_img_size=64,
        vit_dropout=0.0,
        vit_cls_flag=False,
        lm_model_type="testing",
        lm_hidden_dim=64,
        lm_inter_dim=128,
        lm_rms_eps=1e-5,
        lm_re_base=10_000.0,
        lm_max_position_embeddings=256,
        lm_attn_scaling=1.0,
        lm_vocab_size=128,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_dropout=0.0,
        lm_n_blocks=2,
        lm_use_tokens=False,
        lm_tie_weights=True,
        mp_pixel_shuffle_factor=2,
        mp_image_token_length=4,
        momh_enabled=True,
    )
    tokenizer = _DummyTokenizer(vocab_size=cfg.lm_vocab_size, image_token_id=5, eos_token_id=1, pad_token_id=0)

    model = VisionLanguageModel(cfg, load_backbone=False, tokenizer=tokenizer).eval()

    batch_size = 1
    num_images = 2
    seq_len = 32

    # Two images -> expect 2 * mp_image_token_length placeholders in input_ids.
    num_placeholders = num_images * cfg.mp_image_token_length
    # Avoid accidentally sampling `image_token_id` outside the intended placeholder span.
    input_ids = torch.randint(0, cfg.lm_vocab_size - 1, (batch_size, seq_len))
    input_ids[:, :num_placeholders] = tokenizer.image_token_id

    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    images = torch.randn(num_images, 3, cfg.vit_img_size, cfg.vit_img_size)

    logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=input_ids.clone())
    assert logits.shape == (batch_size, seq_len, cfg.lm_vocab_size)
    assert loss is not None
    assert torch.isfinite(loss).item()
