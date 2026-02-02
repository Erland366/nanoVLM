import torch

from models.momh_attention import generate_momh_mask_mod_from_modality


def _eval(mask_mod, *, b: int, h: int, q_abs: list[int], kv_abs: list[int]) -> torch.Tensor:
    b_t = torch.tensor(b, dtype=torch.long)
    h_t = torch.tensor(h, dtype=torch.long)
    q_t = torch.tensor(q_abs, dtype=torch.long)
    kv_t = torch.tensor(kv_abs, dtype=torch.long)
    out = mask_mod(
        b_t.view(1, 1, 1, 1),
        h_t.view(1, 1, 1, 1),
        q_t.view(1, 1, -1, 1),
        kv_t.view(1, 1, 1, -1),
    )
    return out[0, 0]


def test_left_padding_blocks_attention():
    # seq: 0 1 2 3 4 5
    # pad: P P C C C C
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    is_vision = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.bool)

    mask_mod = generate_momh_mask_mod_from_modality(
        10, is_vision=is_vision, attention_mask=attention_mask, q_offset=0, pct_v=0.4, pct_t=0.4
    )

    # Any attention involving padding positions should be False.
    out = _eval(mask_mod, b=0, h=0, q_abs=[0, 2], kv_abs=[1, 2, 3])
    assert torch.equal(out[0], torch.zeros(3, dtype=torch.bool))  # q=0 is padding
    assert out[1, 0].item() is False  # kv=1 is padding


def test_scattered_vision_tokens():
    # Vision tokens at positions 1 and 4 (not a contiguous span).
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    is_vision = torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.bool)

    # 10 heads: H_V=4, H_T=4, H_VT=2
    mask_mod = generate_momh_mask_mod_from_modality(
        10, is_vision=is_vision, attention_mask=attention_mask, q_offset=0, pct_v=0.4, pct_t=0.4
    )

    # V-head: vision->vision only
    v_head = 0
    out_v = _eval(mask_mod, b=0, h=v_head, q_abs=[1], kv_abs=[0, 1, 4, 5])
    assert out_v[0, 0].item() is False  # kv text
    assert out_v[0, 1].item() is True   # kv vision (same token)
    assert out_v[0, 2].item() is True   # kv vision (other vision token)
    assert out_v[0, 3].item() is False  # kv text

    # T-head: text->text only (causal)
    t_head = 4
    out_t = _eval(mask_mod, b=0, h=t_head, q_abs=[5], kv_abs=[0, 2, 4, 5])
    assert out_t[0, 0].item() is True   # kv text, causal
    assert out_t[0, 1].item() is True   # kv text, causal
    assert out_t[0, 2].item() is False  # kv vision blocked for T-head
    assert out_t[0, 3].item() is True   # kv text, self

    out_t2 = _eval(mask_mod, b=0, h=t_head, q_abs=[2], kv_abs=[5])
    assert out_t2[0, 0].item() is False  # future key


def test_decode_q_offset_maps_to_absolute_position():
    # Simulate decode: Q_LEN=1 corresponds to absolute position 5 in a KV_LEN=6 context.
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    is_vision = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)

    # Query is last token (abs=5), which is text here.
    mask_mod = generate_momh_mask_mod_from_modality(
        10, is_vision=is_vision, attention_mask=attention_mask, q_offset=5, pct_v=0.4, pct_t=0.4
    )

    vt_head = 8  # first VT head (H_V=4, H_T=4)
    out = _eval(mask_mod, b=0, h=vt_head, q_abs=[0], kv_abs=[0, 1, 2, 3, 4, 5])
    assert out[0, 0].item() is True   # vision
    assert out[0, 1].item() is True   # vision
    assert out[0, 2].item() is True   # text (causal)
    assert out[0, 5].item() is True   # self


def test_multi_image_cross_attention_allowed():
    # Two "images" -> two clusters of vision tokens.
    attention_mask = torch.ones((1, 8), dtype=torch.bool)
    is_vision = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0]], dtype=torch.bool)

    mask_mod = generate_momh_mask_mod_from_modality(
        10, is_vision=is_vision, attention_mask=attention_mask, q_offset=0, pct_v=0.4, pct_t=0.4
    )

    v_head = 0
    out = _eval(mask_mod, b=0, h=v_head, q_abs=[1], kv_abs=[4])
    assert out[0, 0].item() is True  # vision->vision across images allowed


def test_document_ids_block_cross_doc_attention():
    # Two packed documents in a single sequence: positions 0..3 are doc0, 4..7 are doc1.
    attention_mask = torch.ones((1, 8), dtype=torch.bool)
    document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

    # Vision tokens exist in both documents.
    is_vision = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0]], dtype=torch.bool)

    mask_mod = generate_momh_mask_mod_from_modality(
        10,
        is_vision=is_vision,
        attention_mask=attention_mask,
        document_ids=document_ids,
        q_offset=0,
        pct_v=0.4,
        pct_t=0.4,
    )

    # V-head: vision->vision is allowed only within the same document.
    v_head = 0
    out_v = _eval(mask_mod, b=0, h=v_head, q_abs=[1], kv_abs=[1, 5])
    assert out_v[0, 0].item() is True   # same doc, same token
    assert out_v[0, 1].item() is False  # other doc blocked

    # VT-head: can attend to vision + causal text, but still must stay within doc boundary.
    vt_head = 8  # first VT head (H_V=4, H_T=4)
    out_vt = _eval(mask_mod, b=0, h=vt_head, q_abs=[6], kv_abs=[1, 5, 6])
    assert out_vt[0, 0].item() is False  # other doc vision blocked
    assert out_vt[0, 1].item() is True   # same doc vision allowed
    assert out_vt[0, 2].item() is True   # same doc self allowed
