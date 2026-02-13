"""Shared fixtures for deepstream_encoders Python tests."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import init_cuda

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _gst_and_cuda_init():
    """Initialize GStreamer and CUDA once for the entire test session."""
    Gst.init(None)
    init_cuda(0)
