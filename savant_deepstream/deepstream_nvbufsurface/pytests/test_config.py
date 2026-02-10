"""Tests for TransformConfig construction and field access."""

from __future__ import annotations

import pytest

from deepstream_nvbufsurface import (
    ComputeMode,
    Interpolation,
    Padding,
    TransformConfig,
)


class TestTransformConfigDefaults:
    def test_default_padding(self):
        cfg = TransformConfig()
        assert cfg.padding == Padding.SYMMETRIC

    def test_default_interpolation(self):
        cfg = TransformConfig()
        assert cfg.interpolation == Interpolation.BILINEAR

    def test_default_compute_mode(self):
        cfg = TransformConfig()
        assert cfg.compute_mode == ComputeMode.DEFAULT

    def test_default_src_rect(self):
        cfg = TransformConfig()
        assert cfg.src_rect is None


class TestTransformConfigCustom:
    def test_set_padding(self):
        cfg = TransformConfig(padding=Padding.RIGHT_BOTTOM)
        assert cfg.padding == Padding.RIGHT_BOTTOM

    def test_set_interpolation(self):
        cfg = TransformConfig(interpolation=Interpolation.NEAREST)
        assert cfg.interpolation == Interpolation.NEAREST

    def test_set_compute_mode(self):
        cfg = TransformConfig(compute_mode=ComputeMode.GPU)
        assert cfg.compute_mode == ComputeMode.GPU

    def test_set_src_rect(self):
        rect = (10, 20, 100, 200)
        cfg = TransformConfig(src_rect=rect)
        assert cfg.src_rect == rect

    def test_all_fields(self):
        cfg = TransformConfig(
            padding=Padding.NONE,
            interpolation=Interpolation.ALGO3,
            src_rect=(0, 0, 320, 240),
            compute_mode=ComputeMode.GPU,
        )
        assert cfg.padding == Padding.NONE
        assert cfg.interpolation == Interpolation.ALGO3
        assert cfg.src_rect == (0, 0, 320, 240)
        assert cfg.compute_mode == ComputeMode.GPU


class TestTransformConfigMutable:
    def test_set_padding_after_init(self):
        cfg = TransformConfig()
        cfg.padding = Padding.NONE
        assert cfg.padding == Padding.NONE

    def test_set_interpolation_after_init(self):
        cfg = TransformConfig()
        cfg.interpolation = Interpolation.ALGO2
        assert cfg.interpolation == Interpolation.ALGO2

    def test_set_compute_mode_after_init(self):
        cfg = TransformConfig()
        cfg.compute_mode = ComputeMode.GPU
        assert cfg.compute_mode == ComputeMode.GPU

    def test_set_src_rect_after_init(self):
        cfg = TransformConfig()
        cfg.src_rect = (5, 10, 50, 100)
        assert cfg.src_rect == (5, 10, 50, 100)

    def test_clear_src_rect(self):
        cfg = TransformConfig(src_rect=(1, 2, 3, 4))
        cfg.src_rect = None
        assert cfg.src_rect is None


class TestTransformConfigRepr:
    def test_repr_contains_class_name(self):
        cfg = TransformConfig()
        assert "TransformConfig" in repr(cfg)
