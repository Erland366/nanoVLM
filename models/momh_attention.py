"""
Mixture of Modality Heads (MoMH) Attention Module

Implements specialized attention patterns where different heads focus on different modalities:
- V-heads (40%): Vision -> Vision only (bidirectional)
- T-heads (40%): Text -> Text only (causal)
- VT-heads (20%): Full cross-modal attention

Uses PyTorch's flex_attention for efficient sparse attention computation.

MoMH masking is driven by an explicit per-token modality mask (`is_vision`), typically derived
from `<|image|>` placeholder token positions. This supports multi-image / multi-patch samples
where vision tokens are not a fixed contiguous span.
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile flex_attention for performance
# dynamic=False for prefill: fixed shapes, best performance
# dynamic=True for decode: KV length grows each step, avoid recompilation
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
flex_attention_compiled_dynamic = torch.compile(flex_attention, dynamic=True)

# Increase dynamo cache for multiple mask configurations
torch._dynamo.config.cache_size_limit = 1000

def generate_momh_mask_mod_from_modality(
    n_q_heads: int,
    *,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor | None,
    q_offset: int = 0,
    pct_v: float = 0.4,
    pct_t: float = 0.4,
):
    """
    Generate a MoMH mask_mod based on per-token modality (vision vs text).

    This avoids the flawed assumption that "vision tokens are a fixed span of length S_V".
    Instead, callers provide a boolean mask `is_vision` marking which KV positions are
    vision placeholder tokens (e.g. `<|image|>` positions).

    Args:
        n_q_heads: Total number of query heads.
        is_vision: Bool tensor [B, KV_LEN] marking vision tokens.
        attention_mask: Optional tensor [B, KV_LEN], where 1=content and 0=padding.
        q_offset: Absolute offset to map local q_idx (0..Q_LEN-1) into KV positions.
                  Use 0 for prefill (Q_LEN==KV_LEN) and (KV_LEN-Q_LEN) for decode.
        pct_v: Percentage of heads for V->V attention.
        pct_t: Percentage of heads for T->T attention.

    Returns:
        mask_mod function compatible with flex_attention's create_block_mask.
    """
    if is_vision.dtype is not torch.bool:
        is_vision = is_vision.to(torch.bool)
    if attention_mask is not None and attention_mask.dtype is not torch.bool:
        attention_mask = attention_mask.to(torch.bool)

    H_V = int(n_q_heads * pct_v)
    H_T = int(n_q_heads * pct_t)
    H_T_start = H_V
    H_VT_start = H_V + H_T

    def mask_mod(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_offset
        kv_abs = kv_idx

        if attention_mask is None:
            not_padding = torch.ones_like(q_abs, dtype=torch.bool) & torch.ones_like(
                kv_abs, dtype=torch.bool
            )
        else:
            q_is_content = attention_mask[b, q_abs]
            kv_is_content = attention_mask[b, kv_abs]
            not_padding = q_is_content & kv_is_content

        q_is_vision = is_vision[b, q_abs]
        kv_is_vision = is_vision[b, kv_abs]
        q_is_text = ~q_is_vision
        kv_is_text = ~kv_is_vision

        # V-heads: V->V only (bidirectional within vision tokens)
        head_V = (h < H_T_start) & q_is_vision & kv_is_vision & not_padding

        # T-heads: T->T only (causal within text tokens)
        head_T = (
            (h >= H_T_start)
            & (h < H_VT_start)
            & q_is_text
            & kv_is_text
            & (q_abs >= kv_abs)
            & not_padding
        )

        # VT-heads: cross-modal (full vision + causal non-vision)
        head_VT = (h >= H_VT_start) & not_padding & (kv_is_vision | (q_abs >= kv_abs))

        return head_V | head_T | head_VT

    return mask_mod


def create_momh_block_mask_from_modality(
    *,
    n_q_heads: int,
    q_len: int,
    kv_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pct_v: float,
    pct_t: float,
    device: str = "cuda",
):
    """
    Create a MoMH BlockMask for arbitrary (Q_LEN, KV_LEN) shapes.

    During decode with KV cache, Q_LEN is typically 1 while KV_LEN grows; we map
    q_idx into the absolute KV index space by using q_offset=(KV_LEN-Q_LEN).
    """
    if is_vision.ndim != 2:
        raise ValueError(f"is_vision must have shape [B, KV_LEN], got {tuple(is_vision.shape)}")
    if is_vision.shape[1] < kv_len:
        raise ValueError(
            f"is_vision second dim must be >= kv_len ({kv_len}), got {is_vision.shape[1]}"
        )
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape [B, KV_LEN], got {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[1] < kv_len:
            raise ValueError(
                f"attention_mask second dim must be >= kv_len ({kv_len}), got {attention_mask.shape[1]}"
            )

    q_offset = kv_len - q_len
    mask_mod = generate_momh_mask_mod_from_modality(
        n_q_heads,
        is_vision=is_vision[:, :kv_len],
        attention_mask=attention_mask[:, :kv_len] if attention_mask is not None else None,
        q_offset=q_offset,
        pct_v=pct_v,
        pct_t=pct_t,
    )

    if torch.compiler.is_compiling():
        # Avoid nested compilation / extra Dynamo frames inside an already-compiled model.
        return create_block_mask(
            mask_mod,
            B=is_vision.shape[0],
            H=n_q_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    # Eager mode: use PyTorch's recommended compilation path for create_block_mask.
    return create_block_mask(
        mask_mod,
        B=is_vision.shape[0],
        H=n_q_heads,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=True,
    )


def generate_momh_mask_mod(n_q_heads: int, S_V: int, content_starts: torch.Tensor,
                           pct_v: float = 0.4, pct_t: float = 0.4):
    """
    Generate mask_mod for Mixture of Modality Heads with left-padding support.

    Args:
        n_q_heads: Total number of query heads (e.g., 15)
        S_V: Number of vision tokens (fixed, e.g., 64)
        content_starts: 1D tensor [B] with content start position per batch item
                       (where padding ends and actual content begins)
        pct_v: Percentage of heads for V->V attention (default 0.4)
        pct_t: Percentage of heads for T->T attention (default 0.4)

    Returns:
        mask_mod function for flex_attention's create_block_mask
    """
    H_V = int(n_q_heads * pct_v)      # Number of V-heads (e.g., 6)
    H_T = int(n_q_heads * pct_t)      # Number of T-heads (e.g., 6)
    H_T_start = H_V                    # T-heads start index
    H_VT_start = H_V + H_T             # VT-heads start index

    def mask_mod(b, h, q_idx, kv_idx):
        # Get content start for this batch item (where padding ends)
        content_start = content_starts[b]

        # Vision tokens are at positions [content_start, content_start + S_V)
        q_is_vision = (q_idx >= content_start) & (q_idx < content_start + S_V)
        kv_is_vision = (kv_idx >= content_start) & (kv_idx < content_start + S_V)

        # Positions before content_start are padding (should have no attention)
        q_is_padding = q_idx < content_start
        kv_is_padding = kv_idx < content_start
        not_padding = ~q_is_padding & ~kv_is_padding

        # V-heads [0, H_T_start): V->V only (bidirectional within vision)
        head_V = (h < H_T_start) & q_is_vision & kv_is_vision & not_padding

        # T-heads [H_T_start, H_VT_start): T->T only (causal within text)
        q_is_text = (q_idx >= content_start + S_V)
        kv_is_text = (kv_idx >= content_start + S_V)
        head_T = (h >= H_T_start) & (h < H_VT_start) & \
                 q_is_text & kv_is_text & (q_idx >= kv_idx) & not_padding

        # VT-heads [H_VT_start, n_q_heads): full cross-modal
        # - Vision tokens can see all other vision tokens (bidirectional)
        # - Text tokens can see all vision tokens + causal text
        head_VT = (h >= H_VT_start) & not_padding & (kv_is_vision | (q_idx >= kv_idx))

        return head_V | head_T | head_VT

    return mask_mod


def create_momh_block_mask(n_q_heads: int, seq_len: int, S_V: int,
                           content_starts: torch.Tensor,
                           pct_v: float, pct_t: float, device: str = "cuda"):
    """
    Create block mask with per-batch content_start offsets for MoMH attention.

    Args:
        n_q_heads: Total number of query heads
        seq_len: Sequence length (should be fixed, e.g., lm_max_length)
        S_V: Number of vision tokens (should be fixed, e.g., mp_image_token_length)
        content_starts: 1D tensor [B] with padding offset per batch item
        pct_v: Percentage of heads for V->V
        pct_t: Percentage of heads for T->T
        device: Device string for mask creation

    Returns:
        BlockMask for use with flex_attention
    """
    mask_mod = generate_momh_mask_mod(n_q_heads, S_V, content_starts, pct_v, pct_t)
    return create_block_mask(
        mask_mod,
        B=content_starts.shape[0],
        H=n_q_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True
    )


def compute_content_starts(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute content start positions from attention mask.

    With left-padding, the attention_mask has 0s for padding tokens at the start
    and 1s for actual content. The content_start is the index of the first 1.

    Args:
        attention_mask: Tensor [B, seq_len] with 0 for padding, 1 for content

    Returns:
        Tensor [B] with content start position for each batch item
    """
    return attention_mask.argmax(dim=1)


def generate_momh_score_mod_with_offset(
    n_q_heads: int,
    S_V: int,
    content_starts: torch.Tensor,
    position_offset: torch.Tensor,
    pct_v: float = 0.4,
    pct_t: float = 0.4
):
    """
    Generate score_mod for MoMH with position offset support for decode phase.

    This function creates a score_mod that uses CAPTURED TENSORS for content_starts
    and position_offset. This is critical: changing the VALUES of these tensors
    does NOT trigger recompilation of flex_attention.

    During decode:
    - q_idx in tensor is 0 (single token query)
    - position_offset contains the actual position in the full sequence
    - content_starts tells us where vision/text tokens are located

    Args:
        n_q_heads: Total number of query heads
        S_V: Number of vision tokens (fixed)
        content_starts: Captured tensor [B] with content start positions (mutable values)
        position_offset: Captured scalar tensor with query position offset (mutable value)
        pct_v: Percentage of heads for V->V attention
        pct_t: Percentage of heads for T->T attention

    Returns:
        score_mod function for flex_attention
    """
    H_V = int(n_q_heads * pct_v)
    H_T = int(n_q_heads * pct_t)
    H_T_start = H_V
    H_VT_start = H_V + H_T

    def score_mod(score, b, h, q_idx, kv_idx):
        # Apply offset to get actual query position in the full sequence
        actual_q_idx = q_idx + position_offset

        # Get content start for this batch item (where padding ends)
        content_start = content_starts[b]

        # Vision tokens are at positions [content_start, content_start + S_V)
        q_is_vision = (actual_q_idx >= content_start) & (actual_q_idx < content_start + S_V)
        kv_is_vision = (kv_idx >= content_start) & (kv_idx < content_start + S_V)

        # Text tokens are at positions >= content_start + S_V
        q_is_text = actual_q_idx >= content_start + S_V
        kv_is_text = kv_idx >= content_start + S_V

        # Padding positions
        q_is_padding = actual_q_idx < content_start
        kv_is_padding = kv_idx < content_start
        not_padding = ~q_is_padding & ~kv_is_padding

        # V-heads [0, H_T_start): V->V only (bidirectional within vision)
        is_v_head = h < H_T_start
        v_head_valid = is_v_head & q_is_vision & kv_is_vision & not_padding

        # T-heads [H_T_start, H_VT_start): T->T only (causal within text)
        is_t_head = (h >= H_T_start) & (h < H_VT_start)
        t_head_valid = is_t_head & q_is_text & kv_is_text & (actual_q_idx >= kv_idx) & not_padding

        # VT-heads [H_VT_start, n_q_heads): full cross-modal
        # Vision tokens see all vision (bidirectional)
        # Text tokens see all vision + causal text
        is_vt_head = h >= H_VT_start
        vt_head_valid = is_vt_head & not_padding & (kv_is_vision | (actual_q_idx >= kv_idx))

        # Combine all valid attention patterns
        valid_mask = v_head_valid | t_head_valid | vt_head_valid

        # Return score if valid, else -inf to mask out
        return torch.where(valid_mask, score, torch.tensor(float('-inf'), device=score.device, dtype=score.dtype))

    return score_mod
