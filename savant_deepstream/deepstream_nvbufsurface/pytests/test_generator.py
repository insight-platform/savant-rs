"""Tests for NvBufSurfaceGenerator Python bindings."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import NvBufSurfaceGenerator


class TestGeneratorConstruction:
    def test_rgba_640x480(self):
        gen = NvBufSurfaceGenerator("RGBA", 640, 480)
        assert gen is not None

    def test_nv12_1920x1080(self):
        gen = NvBufSurfaceGenerator("NV12", 1920, 1080)
        assert gen is not None

    def test_custom_fps(self):
        gen = NvBufSurfaceGenerator("RGBA", 320, 240, fps_num=60, fps_den=1)
        assert gen is not None

    def test_custom_pool_size(self):
        gen = NvBufSurfaceGenerator("RGBA", 320, 240, pool_size=8)
        assert gen is not None


class TestNvmmCaps:
    def test_caps_string_format(self, rgba_gen: NvBufSurfaceGenerator):
        caps = rgba_gen.nvmm_caps_str()
        assert "memory:NVMM" in caps
        assert "RGBA" in caps
        assert "640" in caps
        assert "480" in caps

    def test_caps_nv12(self, nv12_gen: NvBufSurfaceGenerator):
        caps = nv12_gen.nvmm_caps_str()
        assert "NV12" in caps


class TestAcquireSurface:
    def test_acquire_returns_nonzero(self, rgba_gen: NvBufSurfaceGenerator):
        buf_ptr = rgba_gen.acquire_surface()
        assert isinstance(buf_ptr, int)
        assert buf_ptr != 0

    def test_acquire_with_id(self, rgba_gen: NvBufSurfaceGenerator):
        buf_ptr = rgba_gen.acquire_surface(id=42)
        assert buf_ptr != 0

    def test_acquire_without_id(self, rgba_gen: NvBufSurfaceGenerator):
        buf_ptr = rgba_gen.acquire_surface(id=None)
        assert buf_ptr != 0


class TestAcquireSurfaceWithPtr:
    def test_returns_triple(self, rgba_gen: NvBufSurfaceGenerator):
        result = rgba_gen.acquire_surface_with_ptr()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_nonzero(self, rgba_gen: NvBufSurfaceGenerator):
        buf_ptr, data_ptr, pitch = rgba_gen.acquire_surface_with_ptr()
        assert buf_ptr != 0
        assert data_ptr != 0
        assert pitch > 0

    def test_with_id(self, rgba_gen: NvBufSurfaceGenerator):
        buf_ptr, data_ptr, pitch = rgba_gen.acquire_surface_with_ptr(id=99)
        assert buf_ptr != 0

    def test_pitch_at_least_width_times_4(self, rgba_gen: NvBufSurfaceGenerator):
        """RGBA pitch must be >= width * 4 bytes."""
        _, _, pitch = rgba_gen.acquire_surface_with_ptr()
        assert pitch >= 640 * 4
