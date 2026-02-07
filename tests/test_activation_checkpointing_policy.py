import torch
from torch.utils.checkpoint import CheckpointPolicy

from models.activation_checkpointing import get_default_sac_policy, get_sac_policy_fn


def test_default_policy_is_attention_only():
    assert get_default_sac_policy() == "attention_only"


def test_matmul_attention_alias_maps_to_attention_only():
    fn_new = get_sac_policy_fn("attention_only")
    fn_old = get_sac_policy_fn("matmul_attention")
    assert fn_new is fn_old


def test_attention_policy_prefers_recompute_for_matmul():
    fn = get_sac_policy_fn("attention_only")
    if hasattr(torch.ops.aten, "mm"):
        policy = fn(None, getattr(torch.ops.aten, "mm"))
        assert policy == CheckpointPolicy.PREFER_RECOMPUTE


def test_attention_policy_must_save_for_attention_ops_when_available():
    fn = get_sac_policy_fn("attention_only")
    for name in (
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_flash_attention_forward",
        "_efficient_attention_forward",
        "scaled_dot_product_attention",
    ):
        if hasattr(torch.ops.aten, name):
            policy = fn(None, getattr(torch.ops.aten, name))
            assert policy == CheckpointPolicy.MUST_SAVE
            return
