from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


def _maybe_add_op(ops: set, name: str) -> None:
    try:
        op = getattr(torch.ops.aten, name)
    except AttributeError:
        return
    ops.add(op)


@lru_cache(maxsize=None)
def _matmul_attention_ops() -> set:
    ops: set = set()
    for name in [
        "mm",
        "bmm",
        "addmm",
        "matmul",
        "_scaled_mm",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_flash_attention_forward",
        "_efficient_attention_forward",
        "scaled_dot_product_attention",
    ]:
        _maybe_add_op(ops, name)
    return ops


def _op_in_set(op, op_set: set) -> bool:
    if op in op_set:
        return True
    packet = getattr(op, "overloadpacket", None)
    return packet in op_set


def _policy_matmul_attention(ctx, op, *args, **kwargs) -> CheckpointPolicy:
    if _op_in_set(op, _matmul_attention_ops()):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


@lru_cache(maxsize=None)
def get_sac_policy_fn(name: str) -> Callable:
    if name == "matmul_attention":
        return _policy_matmul_attention
    raise ValueError(f"Unknown activation checkpointing policy: {name}")


def get_sac_context_fn(policy_name: str) -> Callable:
    policy_fn = get_sac_policy_fn(policy_name)
    return partial(create_selective_checkpoint_contexts, policy_fn)
