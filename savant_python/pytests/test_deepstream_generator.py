"""Tests for NvBufSurfaceGenerator — require CUDA/DeepStream runtime."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "NvBufSurfaceGenerator"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


def _ds_runtime_available() -> bool:
    """Check if DeepStream + CUDA runtime is actually available."""
    try:
        ds.init_cuda(0)
        gen = ds.NvBufSurfaceGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


_has_runtime = _ds_runtime_available()
skip_no_runtime = pytest.mark.skipif(
    not _has_runtime, reason="CUDA/DeepStream not available"
)


@skip_no_runtime
class TestGeneratorConstruction:
    def test_rgba_640x480(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480)
        assert gen is not None

    def test_nv12_1920x1080(self):
        gen = ds.NvBufSurfaceGenerator("NV12", 1920, 1080)
        assert gen is not None

    def test_custom_fps(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 320, 240, fps_num=60, fps_den=1)
        assert gen is not None


@skip_no_runtime
class TestNvmmCaps:
    def test_caps_string_format(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        caps = gen.nvmm_caps_str()
        assert "memory:NVMM" in caps
        assert "RGBA" in caps
        assert "640" in caps
        assert "480" in caps


@skip_no_runtime
class TestAcquireSurface:
    def test_acquire_returns_ds_nvbufsurface_gstbuffer(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        buf = gen.acquire_surface()
        assert isinstance(buf, ds.DsNvBufSurfaceGstBuffer)
        assert buf.ptr != 0

    def test_acquire_with_id(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        buf = gen.acquire_surface(id=42)
        assert buf.ptr != 0


@skip_no_runtime
class TestAcquireSurfaceWithPtr:
    def test_returns_triple(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        result = gen.acquire_surface_with_ptr()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_nonzero(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        buf, data_ptr, pitch = gen.acquire_surface_with_ptr()
        assert isinstance(buf, ds.DsNvBufSurfaceGstBuffer)
        assert buf.ptr != 0
        assert data_ptr != 0
        assert pitch > 0

    def test_pitch_at_least_width_times_4(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=4)
        _, _, pitch = gen.acquire_surface_with_ptr()
        assert pitch >= 640 * 4


@skip_no_runtime
class TestGeneratorProperties:
    def test_width(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.width == 640

    def test_height(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.height == 480

    def test_format(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.format == ds.VideoFormat.RGBA


@skip_no_runtime
class TestGetSavantIdMeta:
    def test_no_meta_returns_empty(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 320, 240, pool_size=2)
        buf = gen.acquire_surface()
        meta = ds.get_savant_id_meta(buf)
        assert meta == []

    def test_frame_meta_present(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 320, 240, pool_size=2)
        buf = gen.acquire_surface(id=42)
        meta = ds.get_savant_id_meta(buf)
        assert len(meta) >= 1
        ids = [v for kind, v in meta if kind == "frame"]
        assert 42 in ids

    def test_null_ptr_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.get_savant_id_meta(0)


@skip_no_runtime
class TestGetNvBufSurfaceInfo:
    def test_returns_tuple(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        buf = gen.acquire_surface()
        info = ds.get_nvbufsurface_info(buf)
        assert isinstance(info, tuple)
        assert len(info) == 4

    def test_correct_dimensions(self):
        gen = ds.NvBufSurfaceGenerator("RGBA", 640, 480, pool_size=2)
        buf = gen.acquire_surface()
        data_ptr, pitch, width, height = ds.get_nvbufsurface_info(buf)
        assert width == 640
        assert height == 480
        assert data_ptr != 0
        assert pitch >= 640 * 4

    def test_null_ptr_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.get_nvbufsurface_info(0)
