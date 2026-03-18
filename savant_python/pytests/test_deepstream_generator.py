"""Tests for BufferGenerator — require CUDA/DeepStream runtime."""

from __future__ import annotations

import pytest

from conftest import HAS_DS_FEATURE, skip_no_ds_runtime

if not HAS_DS_FEATURE:
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)

import savant_rs.deepstream as _ds

ds = _ds


@skip_no_ds_runtime
class TestGeneratorConstruction:
    def test_rgba_640x480(self):
        gen = ds.BufferGenerator("RGBA", 640, 480)
        assert gen is not None

    def test_nv12_1920x1080(self):
        gen = ds.BufferGenerator("NV12", 1920, 1080)
        assert gen is not None

    def test_custom_fps(self):
        gen = ds.BufferGenerator("RGBA", 320, 240, fps_num=60, fps_den=1)
        assert gen is not None


@skip_no_ds_runtime
class TestNvmmCaps:
    def test_caps_string_format(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=4)
        caps = gen.nvmm_caps_str()
        assert "memory:NVMM" in caps
        assert "RGBA" in caps
        assert "640" in caps
        assert "480" in caps


@skip_no_ds_runtime
class TestAcquireSurface:
    def test_acquire_returns_shared_buffer(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=4)
        buf = gen.acquire()
        assert isinstance(buf, ds.SharedBuffer)
        assert buf

    def test_acquire_with_id(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=4)
        buf = gen.acquire(id=42)
        assert buf


@skip_no_ds_runtime
class TestGeneratorProperties:
    def test_width(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.width == 640

    def test_height(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.height == 480

    def test_format(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=2)
        assert gen.format == ds.VideoFormat.RGBA


@skip_no_ds_runtime
class TestGetSavantIdMeta:
    def test_no_meta_returns_empty(self):
        gen = ds.BufferGenerator("RGBA", 320, 240, pool_size=2)
        buf = gen.acquire()
        meta = ds.get_savant_id_meta(buf)
        assert meta == []

    def test_frame_meta_present(self):
        gen = ds.BufferGenerator("RGBA", 320, 240, pool_size=2)
        buf = gen.acquire(id=42)
        meta = ds.get_savant_id_meta(buf)
        assert len(meta) >= 1
        ids = [v for kind, v in meta if kind == ds.SavantIdMetaKind.FRAME]
        assert 42 in ids

    def test_null_ptr_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.get_savant_id_meta(0)


@skip_no_ds_runtime
class TestGetNvBufSurfaceInfo:
    def test_returns_tuple(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=2)
        buf = gen.acquire()
        info = ds.get_nvbufsurface_info(buf)
        assert isinstance(info, tuple)
        assert len(info) == 4

    def test_correct_dimensions(self):
        gen = ds.BufferGenerator("RGBA", 640, 480, pool_size=2)
        buf = gen.acquire()
        data_ptr, pitch, width, height = ds.get_nvbufsurface_info(buf)
        assert width == 640
        assert height == 480
        assert data_ptr != 0
        assert pitch >= 640 * 4

    def test_null_ptr_raises(self):
        with pytest.raises(ValueError, match="null"):
            ds.get_nvbufsurface_info(0)
