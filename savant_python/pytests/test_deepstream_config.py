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

    def test_default_dst_padding(self):
        cfg = ds.TransformConfig()
        assert cfg.dst_padding is None

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


class TestDstPadding:
    def test_dst_padding_construction(self):
        p = ds.DstPadding(10, 20, 30, 40)
        assert p.left == 10
        assert p.top == 20
        assert p.right == 30
        assert p.bottom == 40

    def test_dst_padding_defaults(self):
        p = ds.DstPadding()
        assert p.left == 0
        assert p.top == 0
        assert p.right == 0
        assert p.bottom == 0

    def test_dst_padding_repr(self):
        p = ds.DstPadding(1, 2, 3, 4)
        assert "DstPadding" in repr(p)


class TestTransformConfigCustom:
    def test_set_padding(self):
        cfg = ds.TransformConfig(padding=ds.Padding.RIGHT_BOTTOM)
        assert cfg.padding == ds.Padding.RIGHT_BOTTOM

    def test_set_dst_padding(self):
        p = ds.DstPadding(5, 10, 5, 10)
        cfg = ds.TransformConfig(dst_padding=p)
        assert cfg.dst_padding is not None
        assert cfg.dst_padding.left == 5
        assert cfg.dst_padding.top == 10
        assert cfg.dst_padding.right == 5
        assert cfg.dst_padding.bottom == 10

    def test_set_interpolation(self):
        cfg = ds.TransformConfig(interpolation=ds.Interpolation.NEAREST)
        assert cfg.interpolation == ds.Interpolation.NEAREST

    def test_set_compute_mode(self):
        cfg = ds.TransformConfig(compute_mode=ds.ComputeMode.GPU)
        assert cfg.compute_mode == ds.ComputeMode.GPU

    def test_all_fields(self):
        p = ds.DstPadding(1, 2, 3, 4)
        cfg = ds.TransformConfig(
            padding=ds.Padding.NONE,
            dst_padding=p,
            interpolation=ds.Interpolation.GPU_LANCZOS_VIC_SMART,
            compute_mode=ds.ComputeMode.GPU,
        )
        assert cfg.padding == ds.Padding.NONE
        assert cfg.dst_padding == p
        assert cfg.interpolation == ds.Interpolation.GPU_LANCZOS_VIC_SMART
        assert cfg.compute_mode == ds.ComputeMode.GPU


class TestTransformConfigMutable:
    def test_set_padding_after_init(self):
        cfg = ds.TransformConfig()
        cfg.padding = ds.Padding.NONE
        assert cfg.padding == ds.Padding.NONE

    def test_set_dst_padding_after_init(self):
        cfg = ds.TransformConfig()
        cfg.dst_padding = ds.DstPadding(1, 2, 3, 4)
        assert cfg.dst_padding is not None
        assert cfg.dst_padding.left == 1


class TestTransformConfigRepr:
    def test_repr_contains_class_name(self):
        cfg = ds.TransformConfig()
        assert "TransformConfig" in repr(cfg)
