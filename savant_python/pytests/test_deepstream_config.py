"""Tests for TransformConfig construction and field access."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "TransformConfig"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


class TestTransformConfigDefaults:
    def test_default_padding(self):
        cfg = ds.TransformConfig()
        assert cfg.padding == ds.Padding.SYMMETRIC

    def test_default_interpolation(self):
        cfg = ds.TransformConfig()
        assert cfg.interpolation == ds.Interpolation.BILINEAR

    def test_default_compute_mode(self):
        cfg = ds.TransformConfig()
        assert cfg.compute_mode == ds.ComputeMode.DEFAULT

    def test_default_src_rect(self):
        cfg = ds.TransformConfig()
        assert cfg.src_rect is None


class TestTransformConfigCustom:
    def test_set_padding(self):
        cfg = ds.TransformConfig(padding=ds.Padding.RIGHT_BOTTOM)
        assert cfg.padding == ds.Padding.RIGHT_BOTTOM

    def test_set_interpolation(self):
        cfg = ds.TransformConfig(interpolation=ds.Interpolation.NEAREST)
        assert cfg.interpolation == ds.Interpolation.NEAREST

    def test_set_compute_mode(self):
        cfg = ds.TransformConfig(compute_mode=ds.ComputeMode.GPU)
        assert cfg.compute_mode == ds.ComputeMode.GPU

    def test_set_src_rect(self):
        rect = (10, 20, 100, 200)
        cfg = ds.TransformConfig(src_rect=rect)
        assert cfg.src_rect == rect

    def test_all_fields(self):
        cfg = ds.TransformConfig(
            padding=ds.Padding.NONE,
            interpolation=ds.Interpolation.ALGO3,
            src_rect=(0, 0, 320, 240),
            compute_mode=ds.ComputeMode.GPU,
        )
        assert cfg.padding == ds.Padding.NONE
        assert cfg.interpolation == ds.Interpolation.ALGO3
        assert cfg.src_rect == (0, 0, 320, 240)
        assert cfg.compute_mode == ds.ComputeMode.GPU


class TestTransformConfigMutable:
    def test_set_padding_after_init(self):
        cfg = ds.TransformConfig()
        cfg.padding = ds.Padding.NONE
        assert cfg.padding == ds.Padding.NONE

    def test_set_src_rect_after_init(self):
        cfg = ds.TransformConfig()
        cfg.src_rect = (5, 10, 50, 100)
        assert cfg.src_rect == (5, 10, 50, 100)

    def test_clear_src_rect(self):
        cfg = ds.TransformConfig(src_rect=(1, 2, 3, 4))
        cfg.src_rect = None
        assert cfg.src_rect is None


class TestTransformConfigRepr:
    def test_repr_contains_class_name(self):
        cfg = ds.TransformConfig()
        assert "TransformConfig" in repr(cfg)
