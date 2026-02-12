"""Tests for transform (scale + letterbox) Python API."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import (
    ComputeMode,
    Interpolation,
    NvBufSurfaceGenerator,
    Padding,
    TransformConfig,
)


def _make_gen(fmt: str, w: int, h: int) -> NvBufSurfaceGenerator:
    return NvBufSurfaceGenerator(fmt, w, h, pool_size=2)


# ── Basic transforms ─────────────────────────────────────────────────────


class TestTransform:
    def test_same_size(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_downscale_symmetric(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(padding=Padding.SYMMETRIC)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_downscale_right_bottom(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(padding=Padding.RIGHT_BOTTOM)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_no_padding(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(padding=Padding.NONE)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_upscale(self):
        src_gen = _make_gen("RGBA", 320, 240)
        dst_gen = _make_gen("RGBA", 1920, 1080)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


# ── With ID ──────────────────────────────────────────────────────────────


class TestTransformWithId:
    def test_id_propagated(self):
        from deepstream_nvbufsurface import get_savant_id_meta

        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface(id=123)
        cfg = TransformConfig()
        dst = dst_gen.transform(src, cfg, id=456)
        meta = get_savant_id_meta(dst)
        ids = [v for kind, v in meta if kind == "frame"]
        assert 456 in ids


# ── Interpolation methods ────────────────────────────────────────────────


class TestTransformInterpolation:
    @pytest.mark.parametrize(
        "interp",
        [
            Interpolation.NEAREST,
            Interpolation.BILINEAR,
            Interpolation.ALGO1,
            Interpolation.ALGO2,
            Interpolation.ALGO3,
            Interpolation.ALGO4,
            Interpolation.DEFAULT,
        ],
    )
    def test_all_interpolations(self, interp):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 320, 240)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(interpolation=interp)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


# ── Compute modes ────────────────────────────────────────────────────────


class TestTransformComputeMode:
    def test_gpu_compute(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(compute_mode=ComputeMode.GPU)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_default_compute(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(compute_mode=ComputeMode.DEFAULT)
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


# ── transform_with_ptr ───────────────────────────────────────────────────


class TestTransformWithPtr:
    def test_returns_triple(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        result = dst_gen.transform_with_ptr(src, cfg)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_nonzero_values(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        buf_ptr, data_ptr, pitch = dst_gen.transform_with_ptr(src, cfg)
        assert buf_ptr != 0
        assert data_ptr != 0
        assert pitch > 0


# ── Source crop ──────────────────────────────────────────────────────────


class TestTransformCrop:
    def test_with_src_rect(self):
        src_gen = _make_gen("RGBA", 1920, 1080)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig(src_rect=(100, 100, 800, 600))
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


# ── Color format conversion ─────────────────────────────────────────────


class TestTransformColorConvert:
    def test_rgba_to_nv12(self):
        src_gen = _make_gen("RGBA", 640, 480)
        dst_gen = _make_gen("NV12", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0

    def test_nv12_to_rgba(self):
        src_gen = _make_gen("NV12", 640, 480)
        dst_gen = _make_gen("RGBA", 640, 480)
        src = src_gen.acquire_surface()
        cfg = TransformConfig()
        dst = dst_gen.transform(src, cfg)
        assert dst != 0


# ── Error handling ───────────────────────────────────────────────────────


class TestTransformErrors:
    def test_null_src_raises(self):
        dst_gen = _make_gen("RGBA", 640, 480)
        cfg = TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform(0, cfg)

    def test_null_src_with_ptr_raises(self):
        dst_gen = _make_gen("RGBA", 640, 480)
        cfg = TransformConfig()
        with pytest.raises(ValueError, match="null"):
            dst_gen.transform_with_ptr(0, cfg)
