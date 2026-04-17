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
        vals = {
            int(ds.Padding.NONE),
            int(ds.Padding.RIGHT_BOTTOM),
            int(ds.Padding.SYMMETRIC),
        }
        assert len(vals) == 3

    def test_from_name(self):
        assert ds.Padding.from_name("none") == ds.Padding.NONE
        assert ds.Padding.from_name("right_bottom") == ds.Padding.RIGHT_BOTTOM
        assert ds.Padding.from_name("rightbottom") == ds.Padding.RIGHT_BOTTOM
        assert ds.Padding.from_name("symmetric") == ds.Padding.SYMMETRIC

    def test_from_name_case_insensitive(self):
        assert ds.Padding.from_name("SYMMETRIC") == ds.Padding.SYMMETRIC

    def test_from_name_unknown_raises(self):
        with pytest.raises(ValueError):
            ds.Padding.from_name("unknown")

    def test_repr(self):
        assert repr(ds.Padding.SYMMETRIC) == "Padding.SYMMETRIC"
        assert repr(ds.Padding.NONE) == "Padding.NONE"


class TestInterpolation:
    ALL = None

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.__class__.ALL = [
            ds.Interpolation.NEAREST,
            ds.Interpolation.BILINEAR,
            ds.Interpolation.GPU_CUBIC_VIC_5TAP,
            ds.Interpolation.GPU_SUPER_VIC_10TAP,
            ds.Interpolation.GPU_LANCZOS_VIC_SMART,
            ds.Interpolation.GPU_IGNORED_VIC_NICEST,
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

    def test_from_name_canonical(self):
        assert (
            ds.Interpolation.from_name("gpu_cubic_vic_5tap")
            == ds.Interpolation.GPU_CUBIC_VIC_5TAP
        )
        assert (
            ds.Interpolation.from_name("gpu_super_vic_10tap")
            == ds.Interpolation.GPU_SUPER_VIC_10TAP
        )
        assert (
            ds.Interpolation.from_name("gpu_lanczos_vic_smart")
            == ds.Interpolation.GPU_LANCZOS_VIC_SMART
        )
        assert (
            ds.Interpolation.from_name("gpu_ignored_vic_nicest")
            == ds.Interpolation.GPU_IGNORED_VIC_NICEST
        )
        assert ds.Interpolation.from_name("bilinear") == ds.Interpolation.BILINEAR
        assert ds.Interpolation.from_name("nearest") == ds.Interpolation.NEAREST
        assert ds.Interpolation.from_name("default") == ds.Interpolation.DEFAULT

    def test_from_name_legacy_short(self):
        assert (
            ds.Interpolation.from_name("cubic") == ds.Interpolation.GPU_CUBIC_VIC_5TAP
        )
        assert (
            ds.Interpolation.from_name("super") == ds.Interpolation.GPU_SUPER_VIC_10TAP
        )
        assert (
            ds.Interpolation.from_name("lanczos")
            == ds.Interpolation.GPU_LANCZOS_VIC_SMART
        )
        assert (
            ds.Interpolation.from_name("nicest")
            == ds.Interpolation.GPU_IGNORED_VIC_NICEST
        )

    def test_from_name_legacy_algo(self):
        assert (
            ds.Interpolation.from_name("algo1") == ds.Interpolation.GPU_CUBIC_VIC_5TAP
        )
        assert (
            ds.Interpolation.from_name("algo2") == ds.Interpolation.GPU_SUPER_VIC_10TAP
        )
        assert (
            ds.Interpolation.from_name("algo3")
            == ds.Interpolation.GPU_LANCZOS_VIC_SMART
        )
        assert (
            ds.Interpolation.from_name("algo4")
            == ds.Interpolation.GPU_IGNORED_VIC_NICEST
        )

    def test_from_name_case_insensitive(self):
        assert (
            ds.Interpolation.from_name("GPU_CUBIC_VIC_5TAP")
            == ds.Interpolation.GPU_CUBIC_VIC_5TAP
        )
        assert (
            ds.Interpolation.from_name("GpuLanczosVicSmart")
            == ds.Interpolation.GPU_LANCZOS_VIC_SMART
        )

    def test_from_name_unknown_raises(self):
        with pytest.raises(ValueError):
            ds.Interpolation.from_name("unknown")

    def test_repr(self):
        assert (
            repr(ds.Interpolation.GPU_CUBIC_VIC_5TAP)
            == "Interpolation.GPU_CUBIC_VIC_5TAP"
        )
        assert repr(ds.Interpolation.BILINEAR) == "Interpolation.BILINEAR"


class TestComputeMode:
    def test_variants_exist(self):
        assert ds.ComputeMode.DEFAULT is not None
        assert ds.ComputeMode.GPU is not None
        assert ds.ComputeMode.VIC is not None

    def test_all_distinct(self):
        vals = {
            int(ds.ComputeMode.DEFAULT),
            int(ds.ComputeMode.GPU),
            int(ds.ComputeMode.VIC),
        }
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
