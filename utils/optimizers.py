from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

from models.config import TrainConfig


def _params_requiring_grad(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [p for p in params if getattr(p, "requires_grad", False)]


def _collect_linear_weights(module: nn.Module) -> set[nn.Parameter]:
    weights: set[nn.Parameter] = set()
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            weights.add(submodule.weight)
    return weights


def _append_param_group(
    param_groups: list[dict],
    *,
    params: Iterable[nn.Parameter],
    lr: float,
    lr_key: str,
    algorithm: str | None = None,
) -> None:
    params_list = _params_requiring_grad(params)
    if not params_list:
        return
    group = {"params": params_list, "lr": float(lr), "lr_key": lr_key}
    if algorithm is not None:
        group["algorithm"] = algorithm
    param_groups.append(group)


def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    optimizer_name = getattr(train_cfg, "optimizer", "adamw")
    if optimizer_name == "adamw":
        return _build_adamw(model, train_cfg)
    if optimizer_name == "muon":
        return _build_muon(model, train_cfg)
    raise ValueError(f"Unknown optimizer: {optimizer_name}. Expected 'adamw' or 'muon'.")


def _build_adamw(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    param_groups: list[dict] = []

    if train_cfg.lr_mp > 0:
        _append_param_group(
            param_groups,
            params=model.MP.parameters(),
            lr=train_cfg.lr_mp,
            lr_key="mp",
        )
    if train_cfg.lr_vision_backbone > 0:
        _append_param_group(
            param_groups,
            params=model.vision_encoder.parameters(),
            lr=train_cfg.lr_vision_backbone,
            lr_key="vision",
        )
    if train_cfg.lr_language_backbone > 0:
        _append_param_group(
            param_groups,
            params=model.decoder.parameters(),
            lr=train_cfg.lr_language_backbone,
            lr_key="language",
        )

    if not param_groups:
        raise ValueError("No trainable parameters: all learning rates are <= 0.")

    return torch.optim.AdamW(
        param_groups,
        betas=tuple(train_cfg.adamw_betas),
        weight_decay=float(train_cfg.weight_decay),
    )


def _build_muon(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    try:
        from dion import Muon  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Muon requested but the 'dion' package is not installed. "
            "Install it with: uv pip install git+https://github.com/microsoft/dion.git"
        ) from exc

    scalar_algo = getattr(train_cfg, "muon_scalar_algorithm", "adamw")
    if scalar_algo not in ("adamw", "lion"):
        raise ValueError(
            f"Invalid muon_scalar_algorithm: {scalar_algo}. Expected 'adamw' or 'lion'."
        )

    distributed_mesh = dist.group.WORLD if dist.is_available() and dist.is_initialized() else None

    param_groups: list[dict] = []
    excluded_from_muon: set[nn.Parameter] = set()
    if hasattr(model, "decoder") and hasattr(model.decoder, "token_embedding"):
        excluded_from_muon.add(model.decoder.token_embedding.weight)

    def add_module_groups(module: nn.Module, *, lr: float, lr_key: str) -> None:
        if lr <= 0:
            return

        muon_weights = _collect_linear_weights(module)
        muon_weights = {p for p in muon_weights if p.requires_grad and p not in excluded_from_muon}

        scalar_params = [
            p for p in module.parameters() if p.requires_grad and p not in muon_weights
        ]

        _append_param_group(
            param_groups,
            params=sorted(muon_weights, key=id),
            lr=lr,
            lr_key=lr_key,
            algorithm="muon",
        )
        _append_param_group(
            param_groups,
            params=scalar_params,
            lr=lr,
            lr_key=lr_key,
            algorithm=scalar_algo,
        )

    add_module_groups(model.MP, lr=train_cfg.lr_mp, lr_key="mp")
    add_module_groups(model.vision_encoder, lr=train_cfg.lr_vision_backbone, lr_key="vision")
    add_module_groups(model.decoder, lr=train_cfg.lr_language_backbone, lr_key="language")

    if not param_groups:
        raise ValueError("No trainable parameters: all learning rates are <= 0.")

    return Muon(
        param_groups,
        distributed_mesh=distributed_mesh,
        lr=1.0,
        mu=float(train_cfg.muon_mu),
        betas=tuple(train_cfg.adamw_betas),
        weight_decay=float(train_cfg.weight_decay),
        cautious_wd=bool(train_cfg.muon_cautious_wd),
        epsilon=float(train_cfg.muon_epsilon),
        nesterov=bool(train_cfg.muon_nesterov),
        adjust_lr=train_cfg.muon_adjust_lr,
        flatten=False,
        use_triton=bool(train_cfg.muon_use_triton),
    )

