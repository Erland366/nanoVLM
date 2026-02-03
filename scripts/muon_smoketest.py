from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

import torch
import wandb
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import GlobalConfig, TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel
from utils.optimizers import build_optimizer


class _StubTokenizer:
    def __init__(self, *, image_token_id: int = 0, pad_token_id: int = 0):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = None


def _tiny_vlm_config(*, vocab_size: int, max_seq_len: int) -> VLMConfig:
    cfg = VLMConfig()

    cfg.vlm_load_backbone_weights = False
    cfg.momh_enabled = False
    cfg.activation_checkpointing = False

    cfg.vit_img_size = 32
    cfg.vit_patch_size = 16
    cfg.vit_hidden_dim = 64
    cfg.vit_inter_dim = 4 * cfg.vit_hidden_dim
    cfg.vit_n_heads = 4
    cfg.vit_n_blocks = 2

    cfg.lm_hidden_dim = 64
    cfg.lm_inter_dim = 4 * cfg.lm_hidden_dim
    cfg.lm_n_heads = 4
    cfg.lm_n_kv_heads = 2
    cfg.lm_n_blocks = 2
    cfg.lm_max_length = max_seq_len
    cfg.lm_use_tokens = False
    cfg.lm_tie_weights = True

    cfg.extra_token_amount = 0
    cfg.lm_base_vocab_size = vocab_size
    cfg.lm_vocab_size = vocab_size

    cfg.mp_pixel_shuffle_factor = 2
    cfg.mp_image_token_length = 4

    return cfg


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    def _cuda_usable() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            _ = torch.ones(1, device="cuda")
        except Exception:
            return False
        return True

    if args.device == "cuda":
        if not _cuda_usable():
            raise RuntimeError(
                "Requested --device cuda, but CUDA is not usable in this environment. "
                "If you see warnings about unsupported GPU compute capability, install a newer PyTorch build."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if _cuda_usable() else "cpu")

    vlm_cfg = _tiny_vlm_config(vocab_size=int(args.vocab_size), max_seq_len=int(args.seq_len))
    train_cfg = TrainConfig()
    global_cfg = GlobalConfig()

    train_cfg.optimizer = "muon"
    train_cfg.prefix_run_name = "muon-smoketest"
    train_cfg.log_wandb = True
    train_cfg.lr_mp = 1e-3
    train_cfg.lr_vision_backbone = 1e-3
    train_cfg.lr_language_backbone = 1e-3
    train_cfg.max_grad_norm = None

    tokenizer = _StubTokenizer(image_token_id=0, pad_token_id=0)
    model = VisionLanguageModel(vlm_cfg, load_backbone=False, tokenizer=tokenizer).to(device)
    model.train()

    optimizer = build_optimizer(model, train_cfg)

    run = wandb.init(
        entity=train_cfg.wandb_entity,
        project=train_cfg.wandb_project,
        name=f"{train_cfg.prefix_run_name}-{time.strftime('%m%d-%H%M%S')}",
        config={
            "VLMConfig": asdict(vlm_cfg),
            "TrainConfig": asdict(train_cfg),
            "GlobalConfig": asdict(global_cfg),
            "smoketest": {
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "vocab_size": int(args.vocab_size),
                "device": str(device),
            },
        },
    )

    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    vocab_size = int(args.vocab_size)

    images = [[] for _ in range(batch_size)]
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    for step in range(int(args.steps)):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

        t0 = time.perf_counter()
        _, loss = model(input_ids=input_ids, images=images, attention_mask=attention_mask, targets=targets)
        if loss is None:
            raise RuntimeError("Expected non-None loss from VisionLanguageModel forward.")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        run.log(
            {
                "smoketest/loss": float(loss.detach().cpu().item()),
                "smoketest/step_time_ms": float(dt_ms),
            },
            step=step,
        )

    run.finish()


if __name__ == "__main__":
    main()
