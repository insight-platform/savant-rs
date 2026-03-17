"""Tests for DeepStream transform (scale + letterbox) — require CUDA/DeepStream runtime."""

from __future__ import annotations

import pytest

from conftest import HAS_DS_FEATURE, skip_no_ds_runtime
from deepstream_helpers import make_gen

if not HAS_DS_FEATURE:
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)

import savant_rs.deepstream as _ds

ds = _ds


@skip_no_ds_runtime
class TestTransform:
    def test_same_size(self):
        src_gen = make_gen("RGBA", 640, 480, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_downscale_symmetric(self):
        src_gen = make_gen("RGBA", 1920, 1080, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig(padding=ds.Padding.SYMMETRIC)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_no_padding(self):
        src_gen = make_gen("RGBA", 1920, 1080, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig(padding=ds.Padding.NONE)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_upscale(self):
        src_gen = make_gen("RGBA", 320, 240, pool_size=2)
        dst_gen = make_gen("RGBA", 1920, 1080, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


@skip_no_ds_runtime
class TestTransformWithId:
    def test_id_propagated(self):
        src_gen = make_gen("RGBA", 640, 480, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface(id=123)
        cfg = ds.TransformConfig()
        dst = dst_gen.transform(src, cfg, id=456)
        meta = ds.get_savant_id_meta(dst)
        ids = [v for kind, v in meta if kind == "frame"]
        assert 456 in ids


@skip_no_ds_runtime
class TestTransformWithPtr:
    def test_returns_triple(self):
        src_gen = make_gen("RGBA", 640, 480, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        result = dst_gen.transform_with_ptr(src, cfg)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_nonzero_values(self):
        src_gen = make_gen("RGBA", 640, 480, pool_size=2)
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        src = src_gen.acquire_surface()
        cfg = ds.TransformConfig()
        buf_ptr, data_ptr, pitch = dst_gen.transform_with_ptr(src, cfg)
        assert buf_ptr != 0
        assert data_ptr != 0
        assert pitch > 0


@skip_no_ds_runtime
class TestTransformErrors:
    def test_null_src_raises(self):
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        cfg = ds.TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform(0, cfg)

    def test_null_src_with_ptr_raises(self):
        dst_gen = make_gen("RGBA", 640, 480, pool_size=2)
        cfg = ds.TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform_with_ptr(0, cfg)
