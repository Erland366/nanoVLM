from __future__ import annotations

import re

import torch


_SM_ARCH_RE = re.compile(r"^sm_(\d+)")


def _parse_arch_capability(arch: str) -> tuple[int, int] | None:
    match = _SM_ARCH_RE.match(arch)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) < 2:
        return None
    if len(digits) == 2:
        return int(digits[0]), int(digits[1])
    return int(digits[:-1]), int(digits[-1])


def _supported_cuda_capabilities() -> set[tuple[int, int]]:
    supported: set[tuple[int, int]] = set()
    for arch in torch.cuda.get_arch_list():
        parsed = _parse_arch_capability(arch)
        if parsed is not None:
            supported.add(parsed)
    return supported


def _format_sm(capability: tuple[int, int]) -> str:
    major, minor = capability
    return f"sm_{major}{minor}"


def ensure_cuda_device_compatibility(device: torch.device | str) -> None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is not available.")

    supported = _supported_cuda_capabilities()
    if not supported:
        return

    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability in supported:
        return

    supported_text = ", ".join(sorted(_format_sm(cap) for cap in supported))
    device_name = torch.cuda.get_device_name(device_index)
    capability_text = _format_sm(capability)
    raise RuntimeError(
        "CUDA device compatibility check failed: "
        f"device {device_index} ({device_name}) reports {capability_text}, "
        f"but this PyTorch build ({torch.__version__}) supports [{supported_text}]. "
        "Install a PyTorch build that supports this GPU architecture "
        "(for Blackwell sm_120, use a CUDA 12.8+ build such as torch 2.10+cu128)."
    )
