"""
Mixture of Modality Heads (MoMH) Attention Module

Implements specialized attention patterns where different heads focus on different modalities:
- V-heads (40%): Vision -> Vision only (bidirectional)
- T-heads (40%): Text -> Text only (causal)
- VT-heads (20%): Full cross-modal attention

Uses PyTorch's flex_attention for efficient sparse attention computation.
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile flex_attention for performance
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

# Increase dynamo cache for multiple mask configurations
torch._dynamo.config.cache_size_limit = 1000


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
