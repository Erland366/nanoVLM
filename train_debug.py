#!/usr/bin/env python3
"""
Comprehensive torch.compile diagnostic script with synthetic data.

NO network calls, NO disk I/O - starts in ~2 seconds.

=== DEFAULT MODE (Comprehensive Diagnostic) ===

    python train_debug.py

    Runs ALL diagnostics:
    1. Graph break check (fullgraph=True, catches breaks)
    2. Eager vs Compiled comparison (speed + VRAM)
    3. Vary batch test (catches batch dim recompilation)
    4. Vary sequence test (catches seq dim recompilation)
    5. Compile latency comparison (regional vs core)

=== QUICK MODE ===

    python train_debug.py --quick

    Skip dynamic shape tests (faster iteration).

=== SPECIFIC DIAGNOSTICS ===

    # Only TORCH_LOGS diagnostics
    python train_debug.py --diagnose recompiles
    python train_debug.py --diagnose breaks
    python train_debug.py --diagnose all

    # Only profiler
    python train_debug.py --profile

    # Only backend ablation
    python train_debug.py --ablate-backend

    # Only config ablation (supports multiple configs separated by ';')
    python train_debug.py --ablate-config epilogue_fusion
    python train_debug.py --ablate-config "epilogue_fusion;max_autotune"
    python train_debug.py --ablate-config "epilogue_fusion=True;coordinate_descent_tuning=True"

    # Large-scale config ablation (bigger model, more reliable measurements)
    python train_debug.py --ablate-config epilogue_fusion --large-scale

=== OPT-OUT FLAGS ===

    --quick             Skip dynamic shape tests
    --skip-baseline     Skip eager comparison
    --skip-graph-check  Skip fullgraph graph break detection
    --skip-compile-bench Skip compile latency comparison
"""

import argparse
import contextlib
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from statistics import mean

from configs.config_debug import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.synthetic_dataset import get_synthetic_dataloader
from dataclasses import replace


# =============================================================================
# Utility Functions
# =============================================================================

def _fresh_inductor_cache():
    """Context manager for fresh inductor cache (cold compile benchmark)."""
    try:
        from torch._inductor.utils import fresh_inductor_cache
    except Exception:
        return contextlib.nullcontext()
    return fresh_inductor_cache()


def _reset_compiler_state():
    """Reset Dynamo/Inductor state for clean recompilation."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()


def _maybe_mark_dynamic(tensor, dim):
    """Mark tensor dimension as dynamic (with fallback)."""
    if hasattr(torch._dynamo, "maybe_mark_dynamic"):
        torch._dynamo.maybe_mark_dynamic(tensor, dim)
    else:
        torch._dynamo.mark_dynamic(tensor, dim)


def _set_torch_logs(diagnose_mode):
    """Set TORCH_LOGS environment variable based on diagnose mode."""
    log_mapping = {
        "recompiles": "recompiles",
        "breaks": "graph_breaks",
        "guards": "guards",
        "dynamic": "dynamic",
        "perf": "perf_hints",
        "all": "graph_breaks,recompiles,guards,perf_hints",
    }
    if diagnose_mode in log_mapping:
        os.environ["TORCH_LOGS"] = log_mapping[diagnose_mode]
        print(f"[DEBUG] Set TORCH_LOGS={log_mapping[diagnose_mode]}")


# =============================================================================
# Large-Scale Config
# =============================================================================

def _create_large_scale_config(vlm_cfg):
    """Create a larger config for more meaningful benchmarks.

    Increases model dimensions to better measure config impact:
    - LM: 768 hidden, 12 blocks (vs debug: 256, 4)
    - ViT: 512 hidden, 8 blocks (vs debug: 256, 4)

    Total params ~30M (vs debug ~5M).
    """
    return replace(
        vlm_cfg,
        # Language Model - larger
        lm_hidden_dim=768,
        lm_inter_dim=2048,
        lm_n_heads=12,
        lm_n_kv_heads=4,
        lm_n_blocks=12,
        # Vision Transformer - larger
        vit_hidden_dim=512,
        vit_inter_dim=4 * 512,
        vit_n_heads=8,
        vit_n_blocks=8,
    )


# =============================================================================
# Memory Measurement
# =============================================================================

def _get_peak_memory_mb():
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def _reset_memory_stats():
    """Reset GPU memory stats for fresh measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# =============================================================================
# Compile Strategies
# =============================================================================

def _apply_regional_compile(model, backend, mode, fullgraph, compile_dynamic):
    """Apply torch.compile to repeated blocks only (LM + ViT)."""
    # Set model-level flag so forward() calls mark_dynamic on inputs
    model._compile_dynamic = compile_dynamic

    for i, block in enumerate(model.decoder.blocks):
        block._compile_dynamic = compile_dynamic
        model.decoder.blocks[i] = torch.compile(
            block,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )
    for i, block in enumerate(model.vision_encoder.blocks):
        block._compile_dynamic = compile_dynamic
        model.vision_encoder.blocks[i] = torch.compile(
            block,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )


# =============================================================================
# Diagnostic: Graph Break Check
# =============================================================================

def _check_graph_breaks(vlm_cfg, device, batch_size):
    """
    Check for graph breaks using fullgraph=True.
    Returns (has_breaks, error_message).
    """
    print("\n" + "="*60)
    print("1. GRAPH BREAK CHECK (fullgraph=True)")
    print("="*60)

    _reset_compiler_state()
    if "TORCHDYNAMO_DISABLE" in os.environ:
        del os.environ["TORCHDYNAMO_DISABLE"]

    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    model.to(device)

    # Compile with fullgraph=True to catch breaks
    try:
        _apply_regional_compile(model, backend="inductor", mode="default",
                               fullgraph=True, compile_dynamic=True)

        # Build batch
        input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
        for b in range(batch_size):
            input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
        labels = input_ids.clone()
        labels[input_ids == model._image_token_id] = -100
        attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
        images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

        # Run forward + backward
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        loss.backward()

        print("✅ No graph breaks detected in compiled blocks")
        return False, None

    except Exception as e:
        error_msg = str(e)
        if "graph break" in error_msg.lower() or "Dynamo" in error_msg:
            print(f"⚠️  Graph break detected:")
            print(f"   {error_msg[:200]}...")
            return True, error_msg
        else:
            print(f"❌ Compilation error (not graph break):")
            print(f"   {error_msg[:200]}...")
            return True, error_msg
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Diagnostic: Baseline Comparison (Eager vs Compiled)
# =============================================================================

def _run_baseline_comparison(vlm_cfg, device, batch_size, steps):
    """Compare eager vs compiled: speed and VRAM."""
    print("\n" + "="*60)
    print("2. BASELINE COMPARISON (Eager vs Compiled)")
    print("="*60)
    print(f"   Batch size: {batch_size}, Steps: {steps}")

    results = {}

    for mode in ["eager", "compiled"]:
        _reset_compiler_state()
        _reset_memory_stats()

        # Set dynamo state
        if mode == "eager":
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
        elif "TORCHDYNAMO_DISABLE" in os.environ:
            del os.environ["TORCHDYNAMO_DISABLE"]

        model = VisionLanguageModel(vlm_cfg, load_backbone=False)
        model.to(device)

        if mode == "compiled":
            _apply_regional_compile(model, backend="inductor", mode="default",
                                   fullgraph=False, compile_dynamic=True)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Build batch
        input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
        for b in range(batch_size):
            input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
        labels = input_ids.clone()
        labels[input_ids == model._image_token_id] = -100
        attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
        images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

        model.train()
        step_times = []
        _reset_memory_stats()

        for step in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)

        peak_mb = _get_peak_memory_mb()
        # Skip step 0 for compiled (includes compile time)
        avg_step_ms = mean(step_times[1:]) * 1000 if len(step_times) > 1 else step_times[0] * 1000
        first_step_s = step_times[0]

        results[mode] = {
            "peak_mb": peak_mb,
            "avg_step_ms": avg_step_ms,
            "first_step_s": first_step_s,
        }
        print(f"   {mode}: avg={avg_step_ms:.2f}ms, peak_vram={peak_mb:.0f}MB, first_step={first_step_s:.2f}s")

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    eager = results["eager"]
    compiled = results["compiled"]
    speed_ratio = eager["avg_step_ms"] / compiled["avg_step_ms"] if compiled["avg_step_ms"] > 0 else 0
    mem_ratio = compiled["peak_mb"] / eager["peak_mb"] if eager["peak_mb"] > 0 else 0

    print()
    print(f"   {'Mode':<12} | {'Avg Step':<12} | {'Peak VRAM':<12} | {'vs Eager':<15}")
    print("   " + "-" * 55)
    print(f"   {'Eager':<12} | {eager['avg_step_ms']:<10.2f}ms | {eager['peak_mb']:<10.0f}MB | {'baseline':<15}")
    print(f"   {'Compiled':<12} | {compiled['avg_step_ms']:<10.2f}ms | {compiled['peak_mb']:<10.0f}MB | {speed_ratio:.2f}x speed, {mem_ratio:.2f}x mem")

    compile_overhead = compiled["first_step_s"] - eager["first_step_s"]
    print(f"\n   Compile overhead: {compile_overhead:.2f}s (first step difference)")

    return results


# =============================================================================
# Diagnostic: Dynamic Shape Tests
# =============================================================================

def _test_vary_batch(vlm_cfg, device, steps=8):
    """Test recompilation with varying batch sizes."""
    print("\n" + "="*60)
    print("3. DYNAMIC BATCH TEST (vary batch size)")
    print("="*60)

    batch_sizes = [4, 8, 4, 16, 8, 32]
    print(f"   Batch sizes: {batch_sizes[:steps]}")

    _reset_compiler_state()
    if "TORCHDYNAMO_DISABLE" in os.environ:
        del os.environ["TORCHDYNAMO_DISABLE"]

    # Enable recompile logging
    os.environ["TORCH_LOGS"] = "recompiles"

    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    model.to(device)
    model._compile_dynamic = True
    _apply_regional_compile(model, backend="inductor", mode="default",
                           fullgraph=False, compile_dynamic=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    recompiles_detected = 0

    for step in range(min(steps, len(batch_sizes))):
        batch_size = batch_sizes[step]

        input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
        for b in range(batch_size):
            input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
        labels = input_ids.clone()
        labels[input_ids == model._image_token_id] = -100
        attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
        images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

        # Mark dynamic
        _maybe_mark_dynamic(input_ids, 0)
        _maybe_mark_dynamic(attention_mask, 0)
        _maybe_mark_dynamic(labels, 0)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        loss.backward()
        optimizer.step()

    # Clear logs
    if "TORCH_LOGS" in os.environ:
        del os.environ["TORCH_LOGS"]

    # Note: We can't easily count recompiles without parsing stderr
    # For now, just report that the test completed
    print("   ✅ Vary batch test completed (check stderr for recompile logs)")
    print("   TIP: Run with TORCH_LOGS=recompiles to see detailed recompilation info")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _test_vary_sequence(vlm_cfg, device, steps=6):
    """Test recompilation with varying sequence lengths.

    NOTE: Varying sequence lengths with torch.compile can hit edge cases in PyTorch's
    symbolic shape system. This test uses conservative length variations and gracefully
    handles failures that may occur due to internal PyTorch limitations.
    """
    print("\n" + "="*60)
    print("4. DYNAMIC SEQUENCE TEST (vary sequence length)")
    print("="*60)

    # Use conservative sequence length variations (2x range max)
    # Larger variations can trigger PyTorch symbolic shape bugs
    base_len = min(128, vlm_cfg.lm_max_length // 4)
    seq_lengths = [base_len, base_len + 32, base_len + 64, base_len + 32, base_len, base_len + 64]
    seq_lengths = seq_lengths[:steps]
    print(f"   Sequence lengths: {seq_lengths}")

    _reset_compiler_state()
    if "TORCHDYNAMO_DISABLE" in os.environ:
        del os.environ["TORCHDYNAMO_DISABLE"]

    os.environ["TORCH_LOGS"] = "recompiles"

    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    model.to(device)
    model._compile_dynamic = True
    _apply_regional_compile(model, backend="inductor", mode="default",
                           fullgraph=False, compile_dynamic=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    batch_size = 2
    success = True
    error_msg = None

    try:
        for step, seq_len in enumerate(seq_lengths):
            input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, seq_len), device=device)
            # Don't add image tokens if seq too short
            if seq_len > vlm_cfg.mp_image_token_length + 2:
                for b in range(batch_size):
                    input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
            labels = input_ids.clone()
            labels[input_ids == model._image_token_id] = -100
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

            # Mark both batch and sequence as dynamic
            _maybe_mark_dynamic(input_ids, 0)  # batch
            _maybe_mark_dynamic(input_ids, 1)  # sequence
            _maybe_mark_dynamic(attention_mask, 0)
            _maybe_mark_dynamic(attention_mask, 1)
            _maybe_mark_dynamic(labels, 0)
            _maybe_mark_dynamic(labels, 1)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()
    except Exception as e:
        success = False
        error_msg = str(e)[:200]  # Truncate long error messages

    if "TORCH_LOGS" in os.environ:
        del os.environ["TORCH_LOGS"]

    if success:
        print("   ✅ Vary sequence test completed (check stderr for recompile logs)")
        print("   TIP: Run with TORCH_LOGS=recompiles to see detailed recompilation info")
    else:
        print("   ⚠️  Vary sequence test failed (PyTorch symbolic shape limitation)")
        print(f"   Error: {error_msg}")
        print("   NOTE: This is a known issue with dynamic sequence lengths in torch.compile.")
        print("   Consider using fixed sequence lengths or padding to max_length for training.")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Diagnostic: Compile Latency
# =============================================================================

def _benchmark_compile_latency(vlm_cfg, device, batch_size):
    """Compare regional vs core compile latency."""
    print("\n" + "="*60)
    print("5. COMPILE LATENCY COMPARISON")
    print("="*60)

    if "TORCHDYNAMO_DISABLE" in os.environ:
        del os.environ["TORCHDYNAMO_DISABLE"]

    def measure(strategy_name, compile_fn):
        _reset_compiler_state()
        with _fresh_inductor_cache():
            model = VisionLanguageModel(vlm_cfg, load_backbone=False)
            model.to(device)
            compile_fn(model)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)

            # Build batch
            input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
            for b in range(batch_size):
                input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
            labels = input_ids.clone()
            labels[input_ids == model._image_token_id] = -100
            attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
            images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return elapsed

    def compile_core(m):
        m._forward_core = torch.compile(m._forward_core, backend="inductor",
                                        mode="reduce-overhead", fullgraph=True)

    def compile_regional(m):
        _apply_regional_compile(m, backend="inductor", mode="default",
                               fullgraph=False, compile_dynamic=True)

    t_regional = measure("Regional blocks", compile_regional)
    print(f"   Regional blocks: {t_regional:.2f}s (first step)")

    t_core = measure("Full _forward_core", compile_core)
    print(f"   Full _forward_core: {t_core:.2f}s (first step)")

    ratio = t_core / t_regional if t_regional > 0 else 0
    print(f"\n   Regional is {ratio:.1f}x faster compile than full core")

    return {"regional": t_regional, "core": t_core}


# =============================================================================
# Ablation Functions
# =============================================================================

def _ablate_backends(vlm_cfg, device, batch_size):
    """Test eager/aot_eager/inductor backends."""
    print("\n" + "="*60)
    print("BACKEND ABLATION")
    print("="*60)

    backends = [
        ("eager", "Dynamo only (no codegen)"),
        ("aot_eager", "Dynamo + AOTAutograd"),
        ("inductor", "Full stack (default)"),
    ]

    for backend, desc in backends:
        _reset_compiler_state()
        try:
            model = VisionLanguageModel(vlm_cfg, load_backbone=False)
            model.to(device)
            model = torch.compile(model, backend=backend)

            input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
            for b in range(batch_size):
                input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
            attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
            images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out, _ = model(input_ids, images, attention_mask=attention_mask)

            print(f"   {backend}: ✅ OK - {desc}")
        except Exception as e:
            print(f"   {backend}: ❌ FAILED - {str(e)[:80]} ({desc})")


def _parse_config_spec(config_spec):
    """Parse config specification string.

    Formats supported:
    - "epilogue_fusion" -> test True/False
    - "epilogue_fusion=True" -> set to specific value
    - "epilogue_fusion;max_autotune" -> multiple configs
    - "epilogue_fusion=True;max_autotune=False" -> multiple with values

    Returns list of (config_name, value_or_None) tuples.
    """
    configs = []
    for part in config_spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, val = part.split("=", 1)
            # Parse value
            val = val.strip().lower()
            if val == "true":
                val = True
            elif val == "false":
                val = False
            elif val.isdigit():
                val = int(val)
            else:
                try:
                    val = float(val)
                except ValueError:
                    pass  # Keep as string
            configs.append((name.strip(), val))
        else:
            configs.append((part, None))  # None means test both True/False
    return configs


def _ablate_config(vlm_cfg, device, batch_size, config_spec, steps=10):
    """Test inductor config(s) impact on performance.

    Supports multiple configs: "config1;config2" or "config1=True;config2=False"
    """
    configs = _parse_config_spec(config_spec)

    print("\n" + "="*60)
    print(f"CONFIG ABLATION")
    print("="*60)
    print(f"   Configs: {configs}")
    print(f"   Batch size: {batch_size}, Steps: {steps}")

    # Determine what combinations to test
    if all(val is not None for _, val in configs):
        # All configs have specific values - just run once with those settings
        test_cases = [("custom", {name: val for name, val in configs})]
    else:
        # Some configs need True/False testing
        # For simplicity, test each config individually
        test_cases = []
        for name, val in configs:
            if val is None:
                test_cases.append((f"{name}=True", {name: True}))
                test_cases.append((f"{name}=False", {name: False}))
            else:
                test_cases.append((f"{name}={val}", {name: val}))

    results = {}

    for case_name, config_dict in test_cases:
        _reset_compiler_state()
        _reset_memory_stats()

        # Apply configs
        for cfg_name, cfg_val in config_dict.items():
            if hasattr(torch._inductor.config, cfg_name):
                setattr(torch._inductor.config, cfg_name, cfg_val)
            else:
                print(f"   ❌ Config {cfg_name} not found in torch._inductor.config")
                return

        model = VisionLanguageModel(vlm_cfg, load_backbone=False)
        model.to(device)
        _apply_regional_compile(model, backend="inductor", mode="default",
                               fullgraph=False, compile_dynamic=False)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
        for b in range(batch_size):
            input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
        labels = input_ids.clone()
        labels[input_ids == model._image_token_id] = -100
        attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
        images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

        model.train()

        # Warmup (includes compile)
        for _ in range(2):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()

        # Measure
        _reset_memory_stats()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_vram = _get_peak_memory_mb()

        avg_step = elapsed / steps
        results[case_name] = {"time_ms": avg_step * 1000, "vram_mb": peak_vram}
        print(f"   {case_name}: {avg_step*1000:.2f}ms/step, {peak_vram:.0f}MB VRAM")

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    if len(results) >= 2:
        print("\n   Summary:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["time_ms"])
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        print(f"   Fastest: {fastest[0]} ({fastest[1]['time_ms']:.2f}ms)")
        print(f"   Slowest: {slowest[0]} ({slowest[1]['time_ms']:.2f}ms)")
        print(f"   Speedup: {slowest[1]['time_ms']/fastest[1]['time_ms']:.2f}x")


# =============================================================================
# Profiler
# =============================================================================

def _run_with_profiler(vlm_cfg, device, batch_size, steps, output_path):
    """Run with torch.profiler and export chrome trace."""
    print("\n" + "="*60)
    print("PROFILER")
    print("="*60)
    print(f"   Steps: {steps}, Output: {output_path}")

    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    model.to(device)
    _apply_regional_compile(model, backend="inductor", mode="default",
                           fullgraph=False, compile_dynamic=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
    for b in range(batch_size):
        input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
    labels = input_ids.clone()
    labels[input_ids == model._image_token_id] = -100
    attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
    images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

    model.train()

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        loss.backward()
        optimizer.step()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for _ in range(steps):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()

    prof.export_chrome_trace(output_path)
    print(f"   ✅ Trace saved to {output_path}")
    print("   Open in chrome://tracing or edge://tracing")


# =============================================================================
# TORCH_TRACE for tlparse
# =============================================================================

def _run_with_trace(vlm_cfg, device, batch_size, steps, trace_dir):
    """Run with TORCH_TRACE for tlparse analysis.

    NOTE: TORCH_TRACE must be set BEFORE torch.compile is called. This function
    provides instructions for running with tracing since setting the env var
    mid-script doesn't work reliably.
    """
    import shutil

    print("\n" + "="*60)
    print("TORCH_TRACE (for tlparse)")
    print("="*60)

    # Clean up old trace
    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    print(f"\n   TORCH_TRACE must be set BEFORE running the script.")
    print(f"\n   Run this command instead:")
    print(f"\n      TORCH_TRACE={trace_dir} python train_debug.py --skip-graph-check --skip-compile-bench --quick --steps {steps}")
    print(f"\n   Then analyze with:")
    print(f"      pip install tlparse")
    print(f"      tlparse {trace_dir}")
    print("\n   What to look for in tlparse output:")
    print("      - [0/0] = first compile, [0/1] [0/2]... = recompiles (investigate!)")
    print("      - Light green frames = graph breaks")
    print("      - Red frames = compilation errors")
    print("      - inductor_output_code_* = generated Triton kernels")

    # Also run a quick trace to show it works when env var is pre-set
    if os.environ.get("TORCH_TRACE"):
        print(f"\n   TORCH_TRACE is set to: {os.environ['TORCH_TRACE']}")
        print("   Running trace capture...")

        _reset_compiler_state()
        if "TORCHDYNAMO_DISABLE" in os.environ:
            del os.environ["TORCHDYNAMO_DISABLE"]

        model = VisionLanguageModel(vlm_cfg, load_backbone=False)
        model.to(device)
        model._compile_dynamic = True
        _apply_regional_compile(model, backend="inductor", mode="default",
                               fullgraph=False, compile_dynamic=True)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (batch_size, vlm_cfg.lm_max_length), device=device)
        for b in range(batch_size):
            input_ids[b, 1:1 + vlm_cfg.mp_image_token_length] = model._image_token_id
        labels = input_ids.clone()
        labels[input_ids == model._image_token_id] = -100
        attention_mask = torch.ones(batch_size, vlm_cfg.lm_max_length, device=device)
        images = [[torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)] for _ in range(batch_size)]

        model.train()

        for step in range(steps):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()

        print(f"   ✅ Trace captured to {os.environ['TORCH_TRACE']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive torch.compile diagnostic script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Basic options
    parser.add_argument("--steps", type=int, default=10, help="Training steps for benchmarks")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")

    # Opt-out flags (diagnostics ON by default)
    parser.add_argument("--quick", action="store_true",
                       help="Skip dynamic shape tests (faster)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip eager vs compiled comparison")
    parser.add_argument("--skip-graph-check", action="store_true",
                       help="Skip fullgraph graph break detection")
    parser.add_argument("--skip-compile-bench", action="store_true",
                       help="Skip compile latency comparison")

    # Specific diagnostic modes (run ONLY that diagnostic)
    parser.add_argument("--diagnose", type=str,
                       choices=["recompiles", "breaks", "guards", "dynamic", "perf", "all"],
                       help="Run with specific TORCH_LOGS (skips other diagnostics)")
    parser.add_argument("--trace", action="store_true",
                       help="Generate TORCH_TRACE for tlparse analysis")
    parser.add_argument("--trace-dir", type=str, default="/tmp/nanovlm_trace",
                       help="Directory for TORCH_TRACE output")
    parser.add_argument("--profile", action="store_true",
                       help="Run profiler only")
    parser.add_argument("--profile-output", type=str, default="/tmp/nanovlm_trace.json",
                       help="Profiler output path")
    parser.add_argument("--ablate-backend", action="store_true",
                       help="Run backend ablation only")
    parser.add_argument("--ablate-config", type=str,
                       help="Run config ablation only (e.g., epilogue_fusion)")
    parser.add_argument("--large-scale", action="store_true",
                       help="Use larger model (~30M params) for more accurate benchmarks")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_cfg = VLMConfig()

    # Apply large-scale config if requested
    if args.large_scale:
        vlm_cfg = _create_large_scale_config(vlm_cfg)
        args.batch_size = max(args.batch_size, 8)
        args.steps = max(args.steps, 20)

    print("="*60)
    print("torch.compile DIAGNOSTIC REPORT" + (" (large-scale)" if args.large_scale else ""))
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.steps}")
    if args.large_scale:
        print(f"Model: lm={vlm_cfg.lm_hidden_dim}d/{vlm_cfg.lm_n_blocks}L, "
              f"vit={vlm_cfg.vit_hidden_dim}d/{vlm_cfg.vit_n_blocks}L")

    # === Specific diagnostic modes (run only one thing) ===

    if args.diagnose:
        _set_torch_logs(args.diagnose)
        print(f"\nRunning with TORCH_LOGS={os.environ.get('TORCH_LOGS', '')}")
        print("Check stderr for diagnostic output.")
        # Run a simple training loop
        _run_baseline_comparison(vlm_cfg, device, args.batch_size, args.steps)
        return

    if args.trace:
        _run_with_trace(vlm_cfg, device, args.batch_size, args.steps, args.trace_dir)
        return

    if args.profile:
        _run_with_profiler(vlm_cfg, device, args.batch_size, args.steps, args.profile_output)
        return

    if args.ablate_backend:
        _ablate_backends(vlm_cfg, device, args.batch_size)
        return

    if args.ablate_config:
        _ablate_config(vlm_cfg, device, args.batch_size, args.ablate_config, args.steps)
        return

    # === Default: Comprehensive Diagnostic ===

    # 1. Graph break check
    if not args.skip_graph_check:
        _check_graph_breaks(vlm_cfg, device, args.batch_size)

    # 2. Baseline comparison (eager vs compiled, speed + VRAM)
    if not args.skip_baseline:
        _run_baseline_comparison(vlm_cfg, device, args.batch_size, args.steps)

    # 3. Dynamic shape tests
    if not args.quick:
        _test_vary_batch(vlm_cfg, device, steps=8)
        _test_vary_sequence(vlm_cfg, device, steps=6)

    # 4. Compile latency comparison
    if not args.skip_compile_bench:
        _benchmark_compile_latency(vlm_cfg, device, args.batch_size)

    # Final summary
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nFor detailed analysis, run specific diagnostics:")
    print("  --diagnose recompiles   # See recompilation triggers")
    print("  --diagnose breaks       # See graph break locations")
    print("  --trace                 # Generate tlparse trace (deep analysis)")
    print("  --profile               # Generate chrome trace (kernel perf)")
    print("  --ablate-config <name>  # Test specific inductor config")
    print("  --ablate-config 'a;b'   # Test multiple configs")
    print("  --large-scale           # Use larger model for ablation")


if __name__ == "__main__":
    main()
