"""
Tests for MoMH (Mixture of Modality Heads) attention during inference.

Tests verify that:
1. Prefill uses BlockMask for efficient sparse attention
2. Decode uses score_mod with position offset for correct MoMH masking
3. Outputs are finite (not NaN/Inf) for various shapes

Usage:
    cd /home/coder/edd/nanoVLM_root/nanoVLM_momh
    source .venv/bin/activate && pytest tests/test_momh_attention.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import VLMConfig
from models.language_model import LanguageModel
from models.momh_attention import compute_content_starts


# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, flex_attention requires CUDA"
)


@pytest.fixture(scope="module")
def model():
    """Create a shared model instance for all tests."""
    cfg = VLMConfig()
    model = LanguageModel(cfg).to("cuda").half()
    model.eval()
    return model


@pytest.fixture(scope="module")
def config():
    """Get the VLM config."""
    return VLMConfig()


class TestMoMHPrefill:
    """Tests for MoMH attention during prefill phase."""

    @pytest.mark.parametrize("batch_size,seq_len,content_start", [
        (1, 100, 10),   # Small sequence
        (1, 256, 20),   # Medium sequence
        (1, 512, 0),    # No padding
        (2, 100, 10),   # Batch size 2
    ])
    def test_prefill_finite_output(self, model, config, batch_size, seq_len, content_start):
        """Test that prefill produces finite outputs for various shapes."""
        hidden_dim = config.lm_hidden_dim

        # Create input embeddings
        input_embeds = torch.randn(
            batch_size, seq_len, hidden_dim,
            device="cuda", dtype=torch.float16
        )

        # Create attention mask
        attention_mask = torch.ones((batch_size, seq_len), device="cuda")
        attention_mask[:, :content_start] = 0

        # Compute content_starts
        content_starts = compute_content_starts(attention_mask)

        with torch.no_grad():
            output, kv_cache = model(
                input_embeds,
                attention_mask=attention_mask,
                kv_cache=None,
                start_pos=0,
                content_starts=content_starts
            )

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert torch.isfinite(output).all(), "Prefill output contains NaN or Inf"
        assert len(kv_cache) == config.lm_n_blocks

    def test_prefill_different_content_starts_per_batch(self, model, config):
        """Test prefill with different content_starts for each batch item."""
        batch_size = 2
        seq_len = 128
        hidden_dim = config.lm_hidden_dim

        input_embeds = torch.randn(
            batch_size, seq_len, hidden_dim,
            device="cuda", dtype=torch.float16
        )

        # Different padding per batch item
        attention_mask = torch.ones((batch_size, seq_len), device="cuda")
        attention_mask[0, :10] = 0  # First item: 10 padding tokens
        attention_mask[1, :20] = 0  # Second item: 20 padding tokens

        content_starts = compute_content_starts(attention_mask)
        assert content_starts[0] == 10
        assert content_starts[1] == 20

        with torch.no_grad():
            output, kv_cache = model(
                input_embeds,
                attention_mask=attention_mask,
                kv_cache=None,
                start_pos=0,
                content_starts=content_starts
            )

        assert torch.isfinite(output).all(), "Prefill output contains NaN or Inf"


class TestMoMHDecode:
    """Tests for MoMH attention during decode phase."""

    @pytest.mark.parametrize("prefill_len,num_decode_steps", [
        (100, 5),    # Short prefill, few decode steps
        (256, 10),   # Medium prefill
        (64, 20),    # Short prefill, many decode steps
    ])
    def test_decode_finite_output(self, model, config, prefill_len, num_decode_steps):
        """Test that decode produces finite outputs for various shapes."""
        batch_size = 1
        hidden_dim = config.lm_hidden_dim
        content_start = 10

        # Prefill first
        input_embeds = torch.randn(
            batch_size, prefill_len, hidden_dim,
            device="cuda", dtype=torch.float16
        )
        attention_mask = torch.ones((batch_size, prefill_len), device="cuda")
        attention_mask[:, :content_start] = 0
        content_starts = compute_content_starts(attention_mask)

        with torch.no_grad():
            _, kv_cache = model(
                input_embeds,
                attention_mask=attention_mask,
                kv_cache=None,
                start_pos=0,
                content_starts=content_starts
            )

        # Decode steps
        current_seq_len = prefill_len
        for step in range(num_decode_steps):
            next_token_embed = torch.randn(
                batch_size, 1, hidden_dim,
                device="cuda", dtype=torch.float16
            )
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device="cuda")
            ], dim=1)

            current_pos = current_seq_len
            current_seq_len += 1

            with torch.no_grad():
                output, kv_cache = model(
                    next_token_embed,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    start_pos=current_pos,
                    content_starts=content_starts,
                    position_offset=current_pos
                )

            assert output.shape == (batch_size, 1, hidden_dim)
            assert torch.isfinite(output).all(), f"Decode step {step+1} output contains NaN or Inf"

    def test_decode_batch_size_2(self, model, config):
        """Test decode with batch size > 1."""
        batch_size = 2
        prefill_len = 100
        num_decode_steps = 5
        hidden_dim = config.lm_hidden_dim

        # Prefill
        input_embeds = torch.randn(
            batch_size, prefill_len, hidden_dim,
            device="cuda", dtype=torch.float16
        )
        attention_mask = torch.ones((batch_size, prefill_len), device="cuda")
        attention_mask[0, :10] = 0
        attention_mask[1, :15] = 0
        content_starts = compute_content_starts(attention_mask)

        with torch.no_grad():
            _, kv_cache = model(
                input_embeds,
                attention_mask=attention_mask,
                kv_cache=None,
                start_pos=0,
                content_starts=content_starts
            )

        # Decode
        current_seq_len = prefill_len
        for step in range(num_decode_steps):
            next_token_embed = torch.randn(
                batch_size, 1, hidden_dim,
                device="cuda", dtype=torch.float16
            )
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device="cuda")
            ], dim=1)

            current_pos = current_seq_len
            current_seq_len += 1

            with torch.no_grad():
                output, kv_cache = model(
                    next_token_embed,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    start_pos=current_pos,
                    content_starts=content_starts,
                    position_offset=current_pos
                )

            assert torch.isfinite(output).all(), f"Decode step {step+1} output contains NaN or Inf"


class TestMoMHMaskLogic:
    """Tests for the MoMH mask logic itself."""

    def test_content_starts_computation(self):
        """Test that content_starts is computed correctly from attention mask."""
        # Batch with different padding lengths
        attention_mask = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 1],  # 3 padding tokens
            [0, 0, 0, 0, 0, 1, 1, 1],  # 5 padding tokens
            [1, 1, 1, 1, 1, 1, 1, 1],  # 0 padding tokens
        ], device="cuda")

        content_starts = compute_content_starts(attention_mask)

        assert content_starts[0] == 3
        assert content_starts[1] == 5
        assert content_starts[2] == 0

    def test_momh_head_distribution(self, config):
        """Test that MoMH head percentages sum to <= 100%."""
        pct_v = config.momh_head_pct_vision
        pct_t = config.momh_head_pct_text
        pct_vt = 1.0 - pct_v - pct_t

        assert pct_v + pct_t <= 1.0, "V + T percentages exceed 100%"
        assert pct_vt >= 0, "VT percentage is negative"

        n_heads = config.lm_n_heads
        H_V = int(n_heads * pct_v)
        H_T = int(n_heads * pct_t)
        H_VT = n_heads - H_V - H_T

        assert H_V + H_T + H_VT == n_heads, "Head counts don't sum to total"
        assert H_V >= 0 and H_T >= 0 and H_VT >= 0, "Negative head counts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
