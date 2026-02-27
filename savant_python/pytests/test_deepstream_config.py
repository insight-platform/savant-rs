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


class TestRect:
    def test_rect_construction(self):
        r = ds.Rect(10, 20, 100, 200)
        assert r.top == 10
        assert r.left == 20
        assert r.width == 100
        assert r.height == 200

    def test_rect_repr(self):
        r = ds.Rect(1, 2, 3, 4)
        assert "Rect" in repr(r)


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

    def test_all_fields(self):
        cfg = ds.TransformConfig(
            padding=ds.Padding.NONE,
            interpolation=ds.Interpolation.ALGO3,
            compute_mode=ds.ComputeMode.GPU,
        )
        assert cfg.padding == ds.Padding.NONE
        assert cfg.interpolation == ds.Interpolation.ALGO3
        assert cfg.compute_mode == ds.ComputeMode.GPU


class TestTransformConfigMutable:
    def test_set_padding_after_init(self):
        cfg = ds.TransformConfig()
        cfg.padding = ds.Padding.NONE
        assert cfg.padding == ds.Padding.NONE


class TestTransformConfigRepr:
    def test_repr_contains_class_name(self):
        cfg = ds.TransformConfig()
        assert "TransformConfig" in repr(cfg)
