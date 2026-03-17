"""Shared pytest fixtures and skip markers for savant_rs tests."""

from __future__ import annotations

import pytest


# ─── DeepStream feature detection ──────────────────────────────────────────

def _deepstream_feature_compiled() -> bool:
    """Return True if the savant_rs wheel was built with ``deepstream`` feature."""
    try:
        import savant_rs.deepstream as ds

        return hasattr(ds, "DsNvSurfaceBufferGenerator")
    except ImportError:
        return False


def _deepstream_runtime_available() -> bool:
    """Return True if CUDA + DeepStream runtime are operational.

    Performs a full smoke-test: init CUDA, create a tiny generator, and
    acquire one surface.  Cached at module level so it runs only once per
    session.
    """
    if not _deepstream_feature_compiled():
        return False
    try:
        import savant_rs.deepstream as ds

        ds.init_cuda(0)
        gen = ds.DsNvSurfaceBufferGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


HAS_DS_FEATURE: bool = _deepstream_feature_compiled()
HAS_DS_RUNTIME: bool = _deepstream_runtime_available()

skip_no_ds_feature = pytest.mark.skipif(
    not HAS_DS_FEATURE,
    reason="savant_rs built without deepstream feature",
)
skip_no_ds_runtime = pytest.mark.skipif(
    not HAS_DS_RUNTIME,
    reason="CUDA/DeepStream runtime not available",
)


# ─── Optional library detection ────────────────────────────────────────────

def _has_cv2_cuda() -> bool:
    try:
        import cv2

        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception as e:
        print(f"Error importing cupy: {e}")
        return False


HAS_CV2_CUDA: bool = _has_cv2_cuda()
HAS_CUPY: bool = _has_cupy()

skip_no_cv2_cuda = pytest.mark.skipif(
    not HAS_CV2_CUDA,
    reason="OpenCV CUDA not available",
)
skip_no_cupy = pytest.mark.skipif(
    not HAS_CUPY,
    reason="CuPy not available",
)
