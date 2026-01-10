"""
Test script for Mixture of Modality Heads (MoMH) implementation.

Usage:
    # Test forward pass
    python test_momh.py --test-forward

    # Test backward pass (gradient flow)
    python test_momh.py --test-backward

    # Benchmark VRAM at different batch sizes
    python test_momh.py --benchmark-vram --batch_sizes "1 2 4 8 16 32 64 128"

    # Run all tests
    python test_momh.py --all
"""

import argparse
import gc
import torch
import torch.nn.functional as F

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from models.momh_attention import compute_content_starts, create_momh_block_mask


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        raise RuntimeError("CUDA required for MoMH testing (flex_attention needs CUDA)")


def create_dummy_batch(batch_size, seq_len, img_size, num_images_per_sample, tokenizer, device):
    """Create a dummy batch for testing."""
    # Create dummy images [B * num_images, 3, H, W]
    total_images = batch_size * num_images_per_sample
    images = torch.randn(total_images, 3, img_size, img_size, device=device)

    # Create dummy input_ids with image tokens at the start
    # Simulate left-padding: [PAD...PAD, IMG_TOKENS, TEXT_TOKENS]
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Simulate variable padding per sample
    for i in range(batch_size):
        pad_len = torch.randint(0, seq_len // 4, (1,)).item()
        input_ids[i, :pad_len] = tokenizer.pad_token_id
        attention_mask[i, :pad_len] = 0
        # Place image tokens after padding
        img_token_len = 64  # mp_image_token_length
        input_ids[i, pad_len:pad_len + img_token_len] = tokenizer.image_token_id

    # Create labels (shifted input_ids with -100 for non-answer positions)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Convert images to list format expected by VLM
    images_list = [[images[i * num_images_per_sample:(i + 1) * num_images_per_sample]] for i in range(batch_size)]

    return {
        'input_ids': input_ids,
        'images': images_list,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def test_forward_pass(cfg, device):
    """Test that forward pass works with MoMH enabled."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass with MoMH")
    print("=" * 60)

    # Create model
    print("Creating model...")
    model = VisionLanguageModel(cfg, load_backbone=False)
    model.to(device)
    model.eval()

    # Use the model's tokenizer to get correct token ids
    tokenizer = model.tokenizer

    # Create dummy batch
    batch_size = 2
    seq_len = 256  # Smaller for testing
    img_size = cfg.vit_img_size
    num_images = 1

    print(f"Creating dummy batch: batch_size={batch_size}, seq_len={seq_len}")
    batch = create_dummy_batch(batch_size, seq_len, img_size, num_images, tokenizer, device)

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(
                input_ids=batch['input_ids'],
                images=batch['images'],
                attention_mask=batch['attention_mask'],
                targets=batch['labels']
            )

    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}" if loss is not None else "  Loss: None")
    print(f"  Logits finite: {torch.isfinite(logits).all().item()}")

    # Verify content_starts calculation
    content_starts = compute_content_starts(batch['attention_mask'])
    print(f"  Content starts: {content_starts.tolist()}")

    print("\n[PASS] Forward pass completed successfully!")
    return True


def test_backward_pass(cfg, device):
    """Test that gradients flow correctly with MoMH enabled."""
    print("\n" + "=" * 60)
    print("Testing Backward Pass (Gradient Flow) with MoMH")
    print("=" * 60)

    # Create model
    print("Creating model...")
    model = VisionLanguageModel(cfg, load_backbone=False)
    model.to(device)
    model.train()

    # Use the model's tokenizer to get correct token ids
    tokenizer = model.tokenizer

    # Create dummy batch
    batch_size = 2
    seq_len = 256
    img_size = cfg.vit_img_size
    num_images = 1

    print(f"Creating dummy batch: batch_size={batch_size}, seq_len={seq_len}")
    batch = create_dummy_batch(batch_size, seq_len, img_size, num_images, tokenizer, device)

    # Forward pass
    print("Running forward pass...")
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(
            input_ids=batch['input_ids'],
            images=batch['images'],
            attention_mask=batch['attention_mask'],
            targets=batch['labels']
        )

    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    print("Running backward pass...")
    loss.backward()

    # Check gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_finite = torch.isfinite(param.grad).all().item()
            if not grad_finite:
                print(f"  WARNING: Non-finite gradient in {name}")
            grad_stats[name] = {'norm': grad_norm, 'finite': grad_finite}

    total_params_with_grad = len([p for p in model.parameters() if p.grad is not None])
    total_params = len(list(model.parameters()))
    print(f"  Parameters with gradients: {total_params_with_grad}/{total_params}")

    all_finite = all(s['finite'] for s in grad_stats.values())
    print(f"  All gradients finite: {all_finite}")

    if all_finite:
        print("\n[PASS] Backward pass completed successfully!")
        return True
    else:
        print("\n[FAIL] Some gradients are not finite!")
        return False


def benchmark_vram(cfg, device, batch_sizes):
    """Benchmark VRAM usage at different batch sizes."""
    print("\n" + "=" * 60)
    print("Benchmarking VRAM Usage with MoMH")
    print("=" * 60)

    seq_len = cfg.lm_max_length
    img_size = cfg.vit_img_size
    num_images = 1

    results = []

    for batch_size in batch_sizes:
        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create model fresh for each test
            model = VisionLanguageModel(cfg, load_backbone=False)
            model.to(device)
            model.train()

            # Use model's tokenizer for correct token ids
            tokenizer = model.tokenizer

            # Create batch
            batch = create_dummy_batch(batch_size, seq_len, img_size, num_images, tokenizer, device)

            # Forward + backward pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(
                    input_ids=batch['input_ids'],
                    images=batch['images'],
                    attention_mask=batch['attention_mask'],
                    targets=batch['labels']
                )

            loss.backward()

            # Record VRAM
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            results.append({
                'batch_size': batch_size,
                'peak_vram_mb': peak_vram_mb,
                'status': 'OK'
            })
            print(f"  Batch {batch_size:3d}: {peak_vram_mb:,.0f} MB")

            # Cleanup
            del model, batch, logits, loss

        except torch.cuda.OutOfMemoryError:
            results.append({
                'batch_size': batch_size,
                'peak_vram_mb': None,
                'status': 'OOM'
            })
            print(f"  Batch {batch_size:3d}: OOM")

        except Exception as e:
            results.append({
                'batch_size': batch_size,
                'peak_vram_mb': None,
                'status': f'ERROR: {str(e)[:50]}'
            })
            print(f"  Batch {batch_size:3d}: ERROR - {str(e)[:50]}")

    # Summary
    print("\n--- VRAM Usage Summary ---")
    max_batch = 0
    for r in results:
        if r['status'] == 'OK':
            max_batch = max(max_batch, r['batch_size'])

    print(f"Maximum batch size without OOM: {max_batch}")

    return results


def test_momh_mask_creation(cfg, device):
    """Test that MoMH block mask is created correctly."""
    print("\n" + "=" * 60)
    print("Testing MoMH Block Mask Creation")
    print("=" * 60)

    batch_size = 4
    seq_len = 128
    n_heads = cfg.lm_n_heads
    S_V = cfg.mp_image_token_length

    # Create sample content_starts (simulating different padding amounts)
    content_starts = torch.tensor([10, 20, 5, 15], device=device)

    print(f"Creating block mask: B={batch_size}, H={n_heads}, seq_len={seq_len}, S_V={S_V}")
    print(f"Content starts: {content_starts.tolist()}")

    try:
        block_mask = create_momh_block_mask(
            n_q_heads=n_heads,
            seq_len=seq_len,
            S_V=S_V,
            content_starts=content_starts,
            pct_v=cfg.momh_head_pct_vision,
            pct_t=cfg.momh_head_pct_text,
            device=str(device)
        )
        print(f"  Block mask created successfully")
        print(f"  Block mask type: {type(block_mask)}")
        print("\n[PASS] Block mask creation successful!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Block mask creation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test MoMH implementation')
    parser.add_argument('--test-forward', action='store_true', help='Test forward pass')
    parser.add_argument('--test-backward', action='store_true', help='Test backward pass')
    parser.add_argument('--test-mask', action='store_true', help='Test block mask creation')
    parser.add_argument('--benchmark-vram', action='store_true', help='Benchmark VRAM usage')
    parser.add_argument('--batch_sizes', type=str, default="1 2 4 8 16 32 64 128",
                        help='Space-separated batch sizes for VRAM benchmark')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    # If no specific test selected, run all
    if not any([args.test_forward, args.test_backward, args.test_mask, args.benchmark_vram, args.all]):
        args.all = True

    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create config with MoMH enabled
    cfg = VLMConfig()
    cfg.momh_enabled = True
    cfg.momh_head_pct_vision = 0.4
    cfg.momh_head_pct_text = 0.4

    # Use smaller model for faster testing
    cfg.lm_max_length = 512  # Reduced for testing
    # Note: Keep vit_img_size=512 (default) as it must work with pixel_shuffle_factor=4
    # 512/16 = 32 patches per side, which is divisible by 4

    print(f"\nConfig: MoMH enabled={cfg.momh_enabled}, V%={cfg.momh_head_pct_vision}, T%={cfg.momh_head_pct_text}")
    print(f"Heads: Q={cfg.lm_n_heads}, KV={cfg.lm_n_kv_heads}")
    print(f"S_V (image tokens): {cfg.mp_image_token_length}")

    results = {}

    if args.all or args.test_mask:
        results['mask'] = test_momh_mask_creation(cfg, device)

    if args.all or args.test_forward:
        results['forward'] = test_forward_pass(cfg, device)

    if args.all or args.test_backward:
        results['backward'] = test_backward_pass(cfg, device)

    if args.all or args.benchmark_vram:
        batch_sizes = [int(x) for x in args.batch_sizes.split()]
        results['vram'] = benchmark_vram(cfg, device, batch_sizes)

    # Final summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}")
        elif isinstance(result, list):
            max_batch = max([r['batch_size'] for r in result if r['status'] == 'OK'], default=0)
            print(f"  {test_name}: Max batch size = {max_batch}")


if __name__ == '__main__':
    main()
