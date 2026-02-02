import inspect
import os
import random
from typing import Any, Dict, Optional

import numpy
import torch
import torch.distributed as dist

try:
    import torch.distributed.checkpoint as dist_cp
except Exception:  # pragma: no cover - older torch may not provide DCP
    dist_cp = None


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_trainer_state(checkpoint_dir: str, trainer_state: Dict[str, Any]) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    rank = get_rank()
    rank_path = os.path.join(checkpoint_dir, f"trainer_state_rank{rank}.pt")
    torch.save(trainer_state, rank_path)
    if rank == 0:
        torch.save(trainer_state, os.path.join(checkpoint_dir, "trainer_state.pt"))


def load_trainer_state(checkpoint_dir: str, *, strict: bool) -> Optional[Dict[str, Any]]:
    rank = get_rank()
    rank_path = os.path.join(checkpoint_dir, f"trainer_state_rank{rank}.pt")
    fallback_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

    if os.path.exists(rank_path):
        return torch.load(rank_path, map_location="cpu", weights_only=False)
    if os.path.exists(fallback_path):
        return torch.load(fallback_path, map_location="cpu", weights_only=False)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trainer = state.get("trainer")
        if trainer is None and strict:
            raise KeyError("checkpoint.pt missing 'trainer' state")
        return trainer

    if strict:
        raise FileNotFoundError(f"No trainer state found in {checkpoint_dir}")
    return None


def _supports_no_dist(func) -> bool:
    try:
        return "no_dist" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _save_with_dcp(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if dist_cp is None:
        raise RuntimeError("torch.distributed.checkpoint is unavailable")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if _supports_no_dist(dist_cp.save):
        dist_cp.save(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(checkpoint_dir),
            no_dist=True,
        )
        return

    if is_dist():
        raise RuntimeError("DCP save without no_dist requires all ranks; use checkpoint_format='torch' or upgrade PyTorch.")
    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(checkpoint_dir),
    )


def _load_with_dcp(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, strict: bool) -> None:
    if dist_cp is None:
        raise RuntimeError("torch.distributed.checkpoint is unavailable")
    model_state = unwrap_model(model).state_dict()
    optimizer_state = optimizer.state_dict()
    state_dict = {
        "model": model_state,
        "optimizer": optimizer_state,
    }
    if _supports_no_dist(dist_cp.load):
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
            no_dist=True,
        )
    else:
        if is_dist():
            raise RuntimeError("DCP load without no_dist requires all ranks; use checkpoint_format='torch' or upgrade PyTorch.")
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        )

    unwrap_model(model).load_state_dict(model_state, strict=strict)
    optimizer.load_state_dict(optimizer_state)


def save_model_optimizer_state(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    use_dcp: bool,
    strict: bool,
) -> str:
    rank = get_rank()
    if use_dcp:
        if dist_cp is None:
            if strict:
                raise RuntimeError("checkpoint_format='dcp' but torch.distributed.checkpoint is unavailable")
        else:
            if rank == 0:
                _save_with_dcp(checkpoint_dir, model, optimizer)
            if is_dist():
                dist.barrier()
            return "dcp"

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(checkpoint_dir, "checkpoint.pt"),
        )
    if is_dist():
        dist.barrier()
    return "torch"


def load_model_optimizer_state(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    use_dcp: bool,
    strict: bool,
) -> None:
    if use_dcp and dist_cp is not None:
        _load_with_dcp(checkpoint_dir, model, optimizer, strict=strict)
        if is_dist():
            dist.barrier()
        return

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        if use_dcp and strict:
            raise FileNotFoundError(f"Expected DCP checkpoint at {checkpoint_dir} but none found")
        raise FileNotFoundError(f"checkpoint.pt not found in {checkpoint_dir}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(state["model"], strict=strict)
    optimizer.load_state_dict(state["optimizer"])
