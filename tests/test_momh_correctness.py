"""
Correctness tests comparing MoMH flex_attention against vanilla manual implementation.

These tests verify that the flex_attention implementation produces the same results
as a straightforward manual implementation of the MoMH attention pattern.

Usage:
    cd /home/coder/edd/nanoVLM_root/nanoVLM_momh
    source .venv/bin/activate && pytest tests/test_momh_correctness.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.momh_attention import (
    flex_attention_compiled,
    create_momh_block_mask,
    generate_momh_score_mod_with_offset,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, flex_attention requires CUDA"
)


def manual_momh_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    S_V: int,
    content_start: int,
    pct_v: float,
    pct_t: float,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    Manual implementation of MoMH attention for correctness comparison.

    This is a straightforward (but slow) implementation that explicitly
    constructs the attention mask for each head type.

    Args:
        q: Query tensor [B, H, Q_LEN, D]
        k: Key tensor [B, H, KV_LEN, D]
        v: Value tensor [B, H, KV_LEN, D]
        n_heads: Total number of heads
        S_V: Number of vision tokens
        content_start: Position where content starts (after padding)
        pct_v: Percentage of V-heads
        pct_t: Percentage of T-heads
        position_offset: Offset to add to q_idx for decode phase

    Returns:
        Output tensor [B, H, Q_LEN, D]
    """
    B, H, Q_LEN, D = q.shape
    KV_LEN = k.shape[2]

    # Head boundaries
    H_V = int(n_heads * pct_v)
    H_T = int(n_heads * pct_t)
    H_T_start = H_V
    H_VT_start = H_V + H_T

    # Compute attention scores
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Q_LEN, KV_LEN]

    # Create position indices
    q_indices = torch.arange(Q_LEN, device=q.device).unsqueeze(1) + position_offset  # [Q_LEN, 1]
    kv_indices = torch.arange(KV_LEN, device=q.device).unsqueeze(0)  # [1, KV_LEN]

    # Determine token types based on position
    # Vision tokens: [content_start, content_start + S_V)
    # Text tokens: [content_start + S_V, ...)
    # Padding: [0, content_start)

    q_is_vision = (q_indices >= content_start) & (q_indices < content_start + S_V)
    q_is_text = q_indices >= content_start + S_V
    q_is_padding = q_indices < content_start

    kv_is_vision = (kv_indices >= content_start) & (kv_indices < content_start + S_V)
    kv_is_text = kv_indices >= content_start + S_V
    kv_is_padding = kv_indices < content_start

    not_padding = ~q_is_padding & ~kv_is_padding  # [Q_LEN, KV_LEN]

    # Create per-head masks
    masks = torch.zeros(H, Q_LEN, KV_LEN, dtype=torch.bool, device=q.device)

    for h in range(n_heads):
        if h < H_T_start:
            # V-heads: vision -> vision only (bidirectional)
            masks[h] = q_is_vision & kv_is_vision & not_padding
        elif h < H_VT_start:
            # T-heads: text -> text only (causal)
            causal = q_indices >= kv_indices
            masks[h] = q_is_text & kv_is_text & causal & not_padding
        else:
            # VT-heads: cross-modal
            # Vision can see all vision (bidirectional)
            # Text can see all vision + causal text
            causal = q_indices >= kv_indices
            masks[h] = not_padding & (kv_is_vision | causal)

    # Apply masks: set masked positions to -inf
    masks = masks.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, Q_LEN, KV_LEN]
    scores = scores.masked_fill(~masks, float('-inf'))

    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows (replace with 0)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, v)

    return output


class TestMoMHCorrectnessBlockMask:
    """Compare flex_attention with BlockMask against manual implementation."""

    @pytest.mark.parametrize("seq_len,content_start", [
        (128, 10),
        (128, 0),
        (256, 20),
    ])
    def test_prefill_matches_manual(self, seq_len, content_start):
        """Test that flex_attention with BlockMask matches manual implementation."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        pct_v = 0.2
        pct_t = 0.3

        torch.manual_seed(42)

        q = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)

        content_starts = torch.tensor([content_start], device="cuda")

        # Manual implementation
        out_manual = manual_momh_attention(
            q, k, v,
            n_heads=H,
            S_V=S_V,
            content_start=content_start,
            pct_v=pct_v,
            pct_t=pct_t,
            position_offset=0
        )

        # Flex attention with BlockMask
        block_mask = create_momh_block_mask(
            n_q_heads=H,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=pct_v,
            pct_t=pct_t,
            device="cuda"
        )
        out_flex = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # Compare outputs
        # Use relaxed tolerance due to numerical differences
        torch.testing.assert_close(
            out_flex, out_manual,
            atol=1e-4, rtol=1e-3,
            msg=f"BlockMask output doesn't match manual for seq_len={seq_len}, content_start={content_start}"
        )


class TestMoMHCorrectnessScoreMod:
    """Compare flex_attention with score_mod against manual implementation."""

    @pytest.mark.parametrize("kv_len,position_offset", [
        (100, 100),   # Decode at position 100
        (150, 150),   # Decode at position 150
        (200, 200),   # Decode at position 200
    ])
    def test_decode_matches_manual(self, kv_len, position_offset):
        """Test that flex_attention with score_mod matches manual implementation."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        Q_LEN = 1  # Single token decode
        content_start = 10
        pct_v = 0.2
        pct_t = 0.3

        torch.manual_seed(42)

        q = torch.randn(B, H, Q_LEN, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, kv_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, kv_len, D, device="cuda", dtype=torch.float32)

        # Manual implementation
        out_manual = manual_momh_attention(
            q, k, v,
            n_heads=H,
            S_V=S_V,
            content_start=content_start,
            pct_v=pct_v,
            pct_t=pct_t,
            position_offset=position_offset
        )

        # Flex attention with score_mod
        content_starts_buffer = torch.tensor([content_start], dtype=torch.int64, device="cuda")
        position_offset_buffer = torch.tensor(position_offset, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_q_heads=H,
            S_V=S_V,
            content_starts=content_starts_buffer,
            position_offset=position_offset_buffer,
            pct_v=pct_v,
            pct_t=pct_t
        )

        out_flex = flex_attention_compiled(q, k, v, score_mod=score_mod)

        # Compare outputs
        torch.testing.assert_close(
            out_flex, out_manual,
            atol=1e-4, rtol=1e-3,
            msg=f"score_mod output doesn't match manual for kv_len={kv_len}, position_offset={position_offset}"
        )


class TestMoMHHeadBehavior:
    """Test that each head type behaves correctly."""

    def test_v_heads_only_attend_to_vision(self):
        """Verify V-heads only attend to vision tokens, not text."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        seq_len = 128
        content_start = 10
        pct_v = 0.2
        pct_t = 0.3

        H_V = int(H * pct_v)  # 3 V-heads

        torch.manual_seed(42)

        # Create Q where only a TEXT token queries (position 80, which is text)
        q_text_pos = content_start + S_V + 6  # Position 80 = text token
        q = torch.zeros(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        q[:, :, q_text_pos, :] = torch.randn(B, H, D, device="cuda", dtype=torch.float32)

        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)

        content_starts = torch.tensor([content_start], device="cuda")

        block_mask = create_momh_block_mask(
            n_q_heads=H,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=pct_v,
            pct_t=pct_t,
            device="cuda"
        )
        out = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # V-heads (0, 1, 2) should output zeros for text query positions
        # because text tokens can't attend to anything in V-heads
        v_heads_output = out[:, :H_V, q_text_pos, :]  # [B, H_V, D]

        # The output should be zero (or very close) because softmax over all -inf is 0/NaN->0
        assert torch.allclose(v_heads_output, torch.zeros_like(v_heads_output), atol=1e-5), \
            "V-heads should produce zero output for text queries"

    def test_t_heads_only_attend_to_text(self):
        """Verify T-heads only attend to text tokens, not vision."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        seq_len = 128
        content_start = 10
        pct_v = 0.2
        pct_t = 0.3

        H_V = int(H * pct_v)  # 3 V-heads
        H_T = int(H * pct_t)  # 4 T-heads
        H_T_start = H_V
        H_T_end = H_V + H_T

        torch.manual_seed(42)

        # Create Q where only a VISION token queries (position 15, which is vision)
        q_vision_pos = content_start + 5  # Position 15 = vision token
        q = torch.zeros(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        q[:, :, q_vision_pos, :] = torch.randn(B, H, D, device="cuda", dtype=torch.float32)

        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)

        content_starts = torch.tensor([content_start], device="cuda")

        block_mask = create_momh_block_mask(
            n_q_heads=H,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=pct_v,
            pct_t=pct_t,
            device="cuda"
        )
        out = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # T-heads (3, 4, 5, 6) should output zeros for vision query positions
        t_heads_output = out[:, H_T_start:H_T_end, q_vision_pos, :]  # [B, H_T, D]

        assert torch.allclose(t_heads_output, torch.zeros_like(t_heads_output), atol=1e-5), \
            "T-heads should produce zero output for vision queries"

    def test_vt_heads_attend_to_all(self):
        """Verify VT-heads can attend to both vision and text tokens."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        seq_len = 128
        content_start = 10
        pct_v = 0.2
        pct_t = 0.3

        H_V = int(H * pct_v)
        H_T = int(H * pct_t)
        H_VT_start = H_V + H_T

        torch.manual_seed(42)

        # Query from a text token position (should see vision + causal text)
        q_text_pos = content_start + S_V + 10  # Position 84 = text token
        q = torch.zeros(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        q[:, :, q_text_pos, :] = torch.randn(B, H, D, device="cuda", dtype=torch.float32)

        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)

        content_starts = torch.tensor([content_start], device="cuda")

        block_mask = create_momh_block_mask(
            n_q_heads=H,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=pct_v,
            pct_t=pct_t,
            device="cuda"
        )
        out = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # VT-heads should produce non-zero output for text queries
        # (because they can attend to vision + text)
        vt_heads_output = out[:, H_VT_start:, q_text_pos, :]  # [B, H_VT, D]

        # Output should NOT be all zeros
        assert not torch.allclose(vt_heads_output, torch.zeros_like(vt_heads_output), atol=1e-5), \
            "VT-heads should produce non-zero output (can attend to vision + text)"


class TestMoMHConsistency:
    """Test consistency between BlockMask and score_mod approaches."""

    def test_blockmask_and_scoremod_same_result_for_prefill(self):
        """Verify BlockMask and score_mod produce same results for prefill-like scenario."""
        B = 1
        H = 15
        D = 64
        S_V = 64
        seq_len = 128
        content_start = 10
        pct_v = 0.2
        pct_t = 0.3

        torch.manual_seed(42)

        q = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float32)

        content_starts = torch.tensor([content_start], device="cuda")

        # BlockMask approach
        block_mask = create_momh_block_mask(
            n_q_heads=H,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=pct_v,
            pct_t=pct_t,
            device="cuda"
        )
        out_blockmask = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # score_mod approach (with offset=0 for prefill)
        content_starts_buffer = torch.tensor([content_start], dtype=torch.int64, device="cuda")
        position_offset_buffer = torch.tensor(0, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_q_heads=H,
            S_V=S_V,
            content_starts=content_starts_buffer,
            position_offset=position_offset_buffer,
            pct_v=pct_v,
            pct_t=pct_t
        )
        out_scoremod = flex_attention_compiled(q, k, v, score_mod=score_mod)

        # Should be very close (may have small numerical differences)
        torch.testing.assert_close(
            out_blockmask, out_scoremod,
            atol=1e-4, rtol=1e-3,
            msg="BlockMask and score_mod should produce same results"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
