import torch
import pytest

from utils.cuda_compat import ensure_cuda_device_compatibility


def test_noop_for_cpu_device():
    ensure_cuda_device_compatibility(torch.device("cpu"))


def test_cuda_missing_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        ensure_cuda_device_compatibility(torch.device("cuda"))


def test_unsupported_capability_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA RTX PRO 6000 Blackwell")

    with pytest.raises(RuntimeError, match="sm_120"):
        ensure_cuda_device_compatibility(torch.device("cuda"))


def test_supported_capability_passes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA H100")

    ensure_cuda_device_compatibility(torch.device("cuda"))
