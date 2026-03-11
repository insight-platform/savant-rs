"""Tests for Jetson detection and NVENC utilities."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "jetson_model"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


class TestIsJetsonKernel:
    """is_jetson_kernel() works on any platform (no CUDA required)."""

    def test_returns_bool(self) -> None:
        result = ds.is_jetson_kernel()
        assert isinstance(result, bool)


class TestJetsonModel:
    """jetson_model() requires CUDA on Jetson; returns None on non-Jetson."""

    def test_returns_optional_str_or_none(self) -> None:
        try:
            result = ds.jetson_model(0)
        except Exception:
            pytest.skip("CUDA not available")
        assert result is None or isinstance(result, str)

    def test_orin_nano_contains_substring(self) -> None:
        """If we detect Orin Nano, the string contains 'Orin Nano'."""
        try:
            result = ds.jetson_model(0)
        except Exception:
            pytest.skip("CUDA not available")
        if result and "Orin Nano" in result:
            assert "8GB" in result or "4GB" in result


class TestHasNvenc:
    """has_nvenc() returns bool; requires CUDA/NVML."""

    def test_returns_bool(self) -> None:
        try:
            result = ds.has_nvenc(0)
        except Exception:
            pytest.skip("CUDA/NVML not available")
        assert isinstance(result, bool)

    def test_consistent_with_jetson_model(self) -> None:
        """On Orin Nano, has_nvenc must be False."""
        try:
            model = ds.jetson_model(0)
            nvenc = ds.has_nvenc(0)
        except Exception:
            pytest.skip("CUDA/NVML not available")
        if model and "Orin Nano" in model:
            assert nvenc is False
