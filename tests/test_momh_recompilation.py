"""
Diagnostic tests for MoMH flex_attention recompilation and graph breaks.

This script helps identify:
1. How many times torch.compile recompiles the attention function
2. When graph breaks occur during flex_attention execution
3. Impact of different input shapes on recompilation

Usage:
    # Run with recompilation logging enabled:
    cd /home/coder/edd/nanoVLM_root/nanoVLM_momh
    source .venv/bin/activate
    TORCH_LOGS="recompiles,graph_breaks" pytest tests/test_momh_recompilation.py -v -s

    # Run standalone with full diagnostics:
    TORCH_LOGS="recompiles,graph_breaks" python tests/test_momh_recompilation.py
"""

import sys
import os
import logging
from pathlib import Path
from collections import Counter
from io import StringIO

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import VLMConfig
from models.momh_attention import (
    generate_momh_mask_mod,
    generate_momh_score_mod_with_offset,
    create_momh_block_mask,
    compute_content_starts,
    flex_attention_compiled,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, flex_attention requires CUDA"
)


class RecompilationCounter:
    """Context manager to count recompilations."""

    def __init__(self):
        self.recompile_count = 0
        self.graph_break_count = 0
        self._log_handler = None
        self._log_stream = None

    def __enter__(self):
        # Reset dynamo before test
        torch._dynamo.reset()

        # Set up logging to capture recompilation messages
        self._log_stream = StringIO()
        self._log_handler = logging.StreamHandler(self._log_stream)
        self._log_handler.setLevel(logging.DEBUG)

        # Get torch._dynamo logger
        dynamo_logger = logging.getLogger("torch._dynamo")
        dynamo_logger.addHandler(self._log_handler)
        dynamo_logger.setLevel(logging.DEBUG)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Parse log for recompilation/graph break messages
        if self._log_stream:
            log_content = self._log_stream.getvalue()
            self.recompile_count = log_content.count("Recompiling")
            self.graph_break_count = log_content.count("Graph break")

            # Also check for common graph break indicators
            self.graph_break_count += log_content.count("graph_break")
            self.graph_break_count += log_content.count("GRAPH BREAK")

        # Clean up handler
        if self._log_handler:
            dynamo_logger = logging.getLogger("torch._dynamo")
            dynamo_logger.removeHandler(self._log_handler)

        return False


def count_compile_calls():
    """Count calls to compiled functions using torch profiler."""
    # Get compile stats from dynamo
    stats = torch._dynamo.utils.CompileProfiler()
    return stats


@pytest.fixture(scope="module")
def config():
    """Get the VLM config."""
    return VLMConfig()


class TestRecompilationPrefill:
    """Tests to detect recompilations during prefill."""

    def test_same_shape_no_recompile(self, config):
        """Running prefill twice with same shape should not recompile."""
        torch._dynamo.reset()

        batch_size = 1
        seq_len = 128
        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length

        # First run - will compile
        content_starts = torch.tensor([10], device="cuda")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        block_mask = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )

        # First call - compiles
        with torch.no_grad():
            out1 = flex_attention_compiled(q, k, v, block_mask=block_mask)

        # Second call with same shapes - should use cached compilation
        content_starts2 = torch.tensor([15], device="cuda")  # Different value, same shape
        block_mask2 = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts2,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )

        with torch.no_grad():
            out2 = flex_attention_compiled(q, k, v, block_mask=block_mask2)

        assert out1.shape == out2.shape
        print("Same shape prefill: outputs generated successfully")

    def test_different_seq_len_recompiles(self, config):
        """Different sequence lengths will trigger recompilation with dynamic=False."""
        torch._dynamo.reset()

        batch_size = 1
        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length

        seq_lens = [64, 128, 256]
        outputs = []

        for seq_len in seq_lens:
            content_starts = torch.tensor([10], device="cuda")
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

            block_mask = create_momh_block_mask(
                n_heads, seq_len, S_V, content_starts,
                config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
            )

            with torch.no_grad():
                out = flex_attention_compiled(q, k, v, block_mask=block_mask)
            outputs.append(out)
            print(f"  seq_len={seq_len}: shape={out.shape}")

        # Check dynamo cache
        print(f"Expected recompilations: {len(seq_lens)} (one per unique seq_len)")


class TestRecompilationDecode:
    """Tests to detect recompilations during decode phase."""

    def test_decode_score_mod_captured_tensors(self, config):
        """Test that changing captured tensor values doesn't trigger recompile."""
        torch._dynamo.reset()

        batch_size = 1
        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length
        kv_len = 100  # Fixed KV length (prefill length)

        # Create mutable captured tensors
        content_starts_buffer = torch.tensor([10], dtype=torch.int64, device="cuda")
        position_offset_buffer = torch.tensor(100, dtype=torch.int64, device="cuda")

        # Create score_mod with captured tensors
        score_mod = generate_momh_score_mod_with_offset(
            n_heads, S_V, content_starts_buffer, position_offset_buffer,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        # Use non-compiled flex_attention for decode (score_mod path)
        from torch.nn.attention.flex_attention import flex_attention
        flex_attention_decode = torch.compile(flex_attention, dynamic=False)

        # Decode step 1
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            out1 = flex_attention_decode(q, k, v, score_mod=score_mod)

        # Decode step 2 - change captured tensor values (should NOT recompile)
        position_offset_buffer.fill_(101)  # Update position
        k2 = torch.randn(batch_size, n_heads, kv_len + 1, head_dim, device="cuda", dtype=torch.float16)
        v2 = torch.randn(batch_size, n_heads, kv_len + 1, head_dim, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            out2 = flex_attention_decode(q, k2, v2, score_mod=score_mod)

        print(f"  Decode step 1 output shape: {out1.shape}")
        print(f"  Decode step 2 output shape: {out2.shape}")
        print("  Captured tensor value change: no recompilation expected")

    def test_decode_multiple_steps_growing_kv(self, config):
        """Test multiple decode steps with growing KV cache length."""
        torch._dynamo.reset()

        batch_size = 1
        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length

        # Create captured buffers
        content_starts_buffer = torch.tensor([10], dtype=torch.int64, device="cuda")
        position_offset_buffer = torch.tensor(0, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_heads, S_V, content_starts_buffer, position_offset_buffer,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        from torch.nn.attention.flex_attention import flex_attention
        flex_attention_decode = torch.compile(flex_attention, dynamic=False)

        # Simulate decode with growing KV
        initial_kv_len = 100
        num_steps = 10

        print(f"  Running {num_steps} decode steps with growing KV cache")
        print(f"  NOTE: Each unique KV length may trigger recompilation with dynamic=False")

        for step in range(num_steps):
            kv_len = initial_kv_len + step
            position_offset_buffer.fill_(kv_len)

            q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

            with torch.no_grad():
                out = flex_attention_decode(q, k, v, score_mod=score_mod)

            if step < 3 or step == num_steps - 1:
                print(f"    Step {step}: kv_len={kv_len}, out_shape={out.shape}")
            elif step == 3:
                print("    ...")


class TestGraphBreaks:
    """Tests to identify graph breaks in MoMH attention."""

    def test_mask_mod_graph_breaks(self, config):
        """Check if mask_mod causes graph breaks."""
        torch._dynamo.reset()

        n_heads = config.lm_n_heads
        S_V = config.mp_image_token_length
        seq_len = 128

        content_starts = torch.tensor([10], device="cuda")

        # Create mask_mod and trace it
        mask_mod = generate_momh_mask_mod(
            n_heads, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        # The mask_mod is used inside create_block_mask which is compiled
        # Graph breaks would show up during block mask creation
        print("  Creating block mask (check TORCH_LOGS for graph breaks)...")

        block_mask = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )

        print(f"  Block mask created successfully")
        print(f"  Mask shape info: Q_LEN={seq_len}, KV_LEN={seq_len}, B=1, H={n_heads}")

    def test_score_mod_graph_breaks(self, config):
        """Check if score_mod causes graph breaks."""
        torch._dynamo.reset()

        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length

        content_starts = torch.tensor([10], dtype=torch.int64, device="cuda")
        position_offset = torch.tensor(100, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_heads, S_V, content_starts, position_offset,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        from torch.nn.attention.flex_attention import flex_attention
        flex_attention_score = torch.compile(flex_attention, dynamic=False)

        print("  Running flex_attention with score_mod (check TORCH_LOGS for graph breaks)...")

        batch_size = 1
        kv_len = 100
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            out = flex_attention_score(q, k, v, score_mod=score_mod)

        print(f"  Output shape: {out.shape}")
        print(f"  Output finite: {torch.isfinite(out).all().item()}")


class TestDynamicShapeImpact:
    """Tests to measure impact of dynamic=True vs dynamic=False."""

    def test_dynamic_false_recompilation_count(self, config):
        """Measure recompilations with dynamic=False (current setting)."""
        torch._dynamo.reset()

        from torch.nn.attention.flex_attention import flex_attention
        flex_attn_static = torch.compile(flex_attention, dynamic=False)

        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length
        batch_size = 1

        # Different KV lengths (simulating decode steps)
        kv_lengths = [100, 101, 102, 103, 104]

        content_starts = torch.tensor([10], dtype=torch.int64, device="cuda")
        position_offset = torch.tensor(0, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_heads, S_V, content_starts, position_offset,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        print("  Testing dynamic=False with varying KV lengths...")
        print("  (Each unique shape typically triggers a recompilation)")

        for kv_len in kv_lengths:
            position_offset.fill_(kv_len)
            q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

            with torch.no_grad():
                out = flex_attn_static(q, k, v, score_mod=score_mod)

            print(f"    kv_len={kv_len}: computed")

        print(f"  Expected recompilations with dynamic=False: {len(kv_lengths)}")

    def test_dynamic_true_recompilation_count(self, config):
        """Measure recompilations with dynamic=True (alternative setting)."""
        torch._dynamo.reset()

        from torch.nn.attention.flex_attention import flex_attention
        flex_attn_dynamic = torch.compile(flex_attention, dynamic=True)

        n_heads = config.lm_n_heads
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        S_V = config.mp_image_token_length
        batch_size = 1

        # Different KV lengths (simulating decode steps)
        kv_lengths = [100, 101, 102, 103, 104]

        content_starts = torch.tensor([10], dtype=torch.int64, device="cuda")
        position_offset = torch.tensor(0, dtype=torch.int64, device="cuda")

        score_mod = generate_momh_score_mod_with_offset(
            n_heads, S_V, content_starts, position_offset,
            config.momh_head_pct_vision, config.momh_head_pct_text
        )

        print("  Testing dynamic=True with varying KV lengths...")
        print("  (Should use symbolic shapes and avoid recompilation)")

        for kv_len in kv_lengths:
            position_offset.fill_(kv_len)
            q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

            with torch.no_grad():
                out = flex_attn_dynamic(q, k, v, score_mod=score_mod)

            print(f"    kv_len={kv_len}: computed")

        print(f"  Expected recompilations with dynamic=True: 1 (initial compile)")


def run_full_diagnostics():
    """Run comprehensive diagnostics with verbose output."""
    print("\n" + "="*70)
    print("MoMH flex_attention Recompilation & Graph Break Diagnostics")
    print("="*70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. These diagnostics require CUDA.")
        return

    config = VLMConfig()
    print(f"\nConfiguration:")
    print(f"  lm_n_heads: {config.lm_n_heads}")
    print(f"  lm_hidden_dim: {config.lm_hidden_dim}")
    print(f"  mp_image_token_length: {config.mp_image_token_length}")
    print(f"  momh_head_pct_vision: {config.momh_head_pct_vision}")
    print(f"  momh_head_pct_text: {config.momh_head_pct_text}")

    print(f"\nTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"dynamo cache_size_limit: {torch._dynamo.config.cache_size_limit}")

    print("\n" + "-"*70)
    print("Test 1: Same Shape Prefill (should not recompile)")
    print("-"*70)
    torch._dynamo.reset()

    batch_size, seq_len = 1, 128
    n_heads = config.lm_n_heads
    head_dim = config.lm_hidden_dim // config.lm_n_heads
    S_V = config.mp_image_token_length

    # Run twice with same shape
    for i in range(2):
        content_starts = torch.tensor([10 + i], device="cuda")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        block_mask = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )
        with torch.no_grad():
            out = flex_attention_compiled(q, k, v, block_mask=block_mask)
        print(f"  Run {i+1}: shape={out.shape}, finite={torch.isfinite(out).all().item()}")

    print("\n" + "-"*70)
    print("Test 2: Different Seq Lengths (will recompile)")
    print("-"*70)
    torch._dynamo.reset()

    for seq_len in [64, 128, 256]:
        content_starts = torch.tensor([10], device="cuda")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        block_mask = create_momh_block_mask(
            n_heads, seq_len, S_V, content_starts,
            config.momh_head_pct_vision, config.momh_head_pct_text, "cuda"
        )
        with torch.no_grad():
            out = flex_attention_compiled(q, k, v, block_mask=block_mask)
        print(f"  seq_len={seq_len}: shape={out.shape}")

    print("\n" + "-"*70)
    print("Test 3: Decode with Growing KV (captured tensor update)")
    print("-"*70)
    torch._dynamo.reset()

    from torch.nn.attention.flex_attention import flex_attention
    flex_attn_decode = torch.compile(flex_attention, dynamic=False)

    content_starts_buffer = torch.tensor([10], dtype=torch.int64, device="cuda")
    position_offset_buffer = torch.tensor(0, dtype=torch.int64, device="cuda")

    score_mod = generate_momh_score_mod_with_offset(
        n_heads, S_V, content_starts_buffer, position_offset_buffer,
        config.momh_head_pct_vision, config.momh_head_pct_text
    )

    for kv_len in [100, 101, 102, 103]:
        position_offset_buffer.fill_(kv_len)
        q = torch.randn(batch_size, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            out = flex_attn_decode(q, k, v, score_mod=score_mod)
        print(f"  kv_len={kv_len}: computed (position_offset={position_offset_buffer.item()})")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Observations:
1. Same shape calls: Should NOT recompile (value changes in captured tensors OK)
2. Different shapes with dynamic=False: WILL recompile each unique shape
3. This means decode phase recompiles for each new KV length!

Recommendations:
- For training: Use fixed seq_len (lm_max_length) - no recompilation issues
- For inference decode: Consider dynamic=True OR pad KV to fixed sizes
- Current cache_size_limit=1000 provides buffer for many shapes
""")


if __name__ == "__main__":
    # Check if running with TORCH_LOGS
    torch_logs = os.environ.get("TORCH_LOGS", "")
    if "recompiles" not in torch_logs:
        print("TIP: Run with TORCH_LOGS='recompiles,graph_breaks' for detailed logging")
        print(f"     Current TORCH_LOGS='{torch_logs}'")
        print()

    run_full_diagnostics()
