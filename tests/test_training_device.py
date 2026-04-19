from __future__ import annotations

from types import SimpleNamespace

import pytest

from library_escape.train.common import _device_runtime_info_from_torch


def _fake_torch(*, version: str, cuda_available: bool, cuda_version: str | None, device_name: str = "RTX 4090"):
    class _FakeCuda:
        def is_available(self) -> bool:
            return cuda_available

        def get_device_name(self, index: int) -> str:
            assert index == 0
            return device_name

    return SimpleNamespace(
        __version__=version,
        cuda=_FakeCuda(),
        version=SimpleNamespace(cuda=cuda_version),
    )


def test_auto_device_falls_back_to_cpu_for_cpu_only_torch():
    info = _device_runtime_info_from_torch(
        "auto",
        _fake_torch(version="2.11.0+cpu", cuda_available=False, cuda_version=None),
    )

    assert info.requested == "auto"
    assert info.selected == "cpu"
    assert info.cuda_available is False
    assert "CPU-only" in info.reason


def test_explicit_cuda_raises_clear_error_when_cuda_is_unavailable():
    with pytest.raises(RuntimeError, match="Requested device 'cuda'"):
        _device_runtime_info_from_torch(
            "cuda",
            _fake_torch(version="2.11.0+cpu", cuda_available=False, cuda_version=None),
        )


def test_explicit_cuda_uses_gpu_when_available():
    info = _device_runtime_info_from_torch(
        "cuda",
        _fake_torch(version="2.11.0", cuda_available=True, cuda_version="12.8"),
    )

    assert info.selected == "cuda"
    assert info.cuda_available is True
    assert info.cuda_version == "12.8"
    assert info.device_name == "RTX 4090"
