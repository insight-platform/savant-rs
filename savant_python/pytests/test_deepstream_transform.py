"""Tests for DeepStream transform (scale + letterbox) — require CUDA/DeepStream runtime."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "NvBufSurfaceGenerator"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


def _ds_runtime_available() -> bool:
    try:
        ds.init_cuda(0)
        gen = ds.NvBufSurfaceGenerator("RGBA", 64, 64, pool_size=1)
        _ = gen.acquire_surface()
        return True
    except Exception:
        return False


_has_runtime = _ds_runtime_available()
skip_no_runtime = pytest.mark.skipif(not _has_runtime, reason="CUDA/DeepStream not available")


def _make_gen(fmt: str, w: int, h: int) -> "ds.NvBufSurfaceGenerator":
    return ds.NvBufSurfaceGenerator(fmt, w, h, pool_size=2)


@skip_no_runtime
class TestTransform:
    def test_same_size(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_downscale_symmetric(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig(padding=ds.Padding.SYMMETRIC)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_no_padding(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig(padding=ds.Padding.NONE)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_upscale(self):
        src_gen = _make_gen("RGBA", 320, 240)
        dst_gen = _make_gen("RGBA", 1920, 1080)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


@skip_no_runtime
class TestTransformWithId:
    def test_id_propagated(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface(id=123)
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg, id=456)
        meta = ds.get_savant_id_meta(dst)
        ids = [v for kind, v in meta if kind == "frame"]
        assert 456 in ids


@skip_no_runtime
class TestTransformWithPtr:
    def test_returns_triple(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        result = dst_gen.transform_with_ptr(src, cfg)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_nonzero_values(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        buf_ptr, data_ptr, pitch = dst_gen.transform_with_ptr(src, cfg)
        assert buf_ptr != 0
        assert data_ptr != 0
        assert pitch > 0


@skip_no_runtime
class TestTransformErrors:
    def test_null_src_raises(self):
        dst_gen = _make_gen("RGBA", 640, 480)
        cfg = ds.TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform(0, cfg)

    def test_null_src_with_ptr_raises(self):
        dst_gen = _make_gen("RGBA", 640, 480)
        cfg = ds.TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform_with_ptr(0, cfg)
