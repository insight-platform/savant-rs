"""Shared fixtures for deepstream_nvbufsurface Python tests."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import NvBufSurfaceGenerator, init_cuda


@pytest.fixture(scope="session", autouse=True)
def _cuda_init():
    """Initialize CUDA once for the entire test session."""
    init_cuda(0)


@pytest.fixture()
def rgba_gen() -> NvBufSurfaceGenerator:
    """640×480 RGBA generator with pool_size=4."""
    return NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)


@pytest.fixture()
def nv12_gen() -> NvBufSurfaceGenerator:
    """640×480 NV12 generator with pool_size=4."""
    return NvBufSurfaceGenerator("NV12", 640, 480, pool_size=4)
