"""Tests for DeepStream Padding, Interpolation, ComputeMode, VideoFormat, MemType enums."""

from __future__ import annotations

import pytest

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "Padding"):
    pytest.skip("savant_rs built without deepstream feature", allow_module_level=True)
ds = _ds


class TestPadding:
    def test_variants_exist(self):
        assert ds.Padding.NONE is not None
        assert ds.Padding.RIGHT_BOTTOM is not None
        assert ds.Padding.SYMMETRIC is not None

    def test_equality(self):
        assert ds.Padding.SYMMETRIC == ds.Padding.SYMMETRIC
        assert ds.Padding.NONE != ds.Padding.SYMMETRIC

    def test_int_conversion(self):
        vals = {int(ds.Padding.NONE), int(ds.Padding.RIGHT_BOTTOM), int(ds.Padding.SYMMETRIC)}
        assert len(vals) == 3


class TestInterpolation:
    ALL = None

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.__class__.ALL = [
            ds.Interpolation.NEAREST,
            ds.Interpolation.BILINEAR,
            ds.Interpolation.ALGO1,
            ds.Interpolation.ALGO2,
            ds.Interpolation.ALGO3,
            ds.Interpolation.ALGO4,
            ds.Interpolation.DEFAULT,
        ]

    def test_all_variants_exist(self):
        for v in self.ALL:
            assert v is not None

    def test_all_distinct(self):
        assert len({int(v) for v in self.ALL}) == len(self.ALL)

    def test_equality(self):
        assert ds.Interpolation.BILINEAR == ds.Interpolation.BILINEAR
        assert ds.Interpolation.NEAREST != ds.Interpolation.BILINEAR


class TestComputeMode:
    def test_variants_exist(self):
        assert ds.ComputeMode.DEFAULT is not None
        assert ds.ComputeMode.GPU is not None
        assert ds.ComputeMode.VIC is not None

    def test_all_distinct(self):
        vals = {int(ds.ComputeMode.DEFAULT), int(ds.ComputeMode.GPU), int(ds.ComputeMode.VIC)}
        assert len(vals) == 3

    def test_equality(self):
        assert ds.ComputeMode.GPU == ds.ComputeMode.GPU
        assert ds.ComputeMode.GPU != ds.ComputeMode.VIC


class TestVideoFormat:
    def test_from_name(self):
        assert ds.VideoFormat.from_name("RGBA") == ds.VideoFormat.RGBA
        assert ds.VideoFormat.from_name("NV12") == ds.VideoFormat.NV12

    def test_name(self):
        assert ds.VideoFormat.RGBA.name() == "RGBA"
        assert ds.VideoFormat.NV12.name() == "NV12"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ds.VideoFormat.from_name("UNKNOWN")


class TestMemType:
    def test_variants_exist(self):
        assert ds.MemType.DEFAULT is not None
        assert ds.MemType.CUDA_PINNED is not None
        assert ds.MemType.SYSTEM is not None

    def test_name(self):
        assert ds.MemType.DEFAULT.name() == "default"
        assert ds.MemType.CUDA_DEVICE.name() == "cuda_device"
