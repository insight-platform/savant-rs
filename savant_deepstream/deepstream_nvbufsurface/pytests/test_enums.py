"""Tests for Padding, Interpolation, and ComputeMode enum types."""

from __future__ import annotations


from deepstream_nvbufsurface import ComputeMode, Interpolation, Padding


# ── Padding ──────────────────────────────────────────────────────────────


class TestPadding:
    def test_variants_exist(self):
        assert Padding.NONE is not None
        assert Padding.RIGHT_BOTTOM is not None
        assert Padding.SYMMETRIC is not None

    def test_equality(self):
        assert Padding.SYMMETRIC == Padding.SYMMETRIC
        assert Padding.NONE != Padding.SYMMETRIC

    def test_int_conversion(self):
        # Variants should have distinct int values
        vals = {int(Padding.NONE), int(Padding.RIGHT_BOTTOM), int(Padding.SYMMETRIC)}
        assert len(vals) == 3

    def test_repr(self):
        r = repr(Padding.SYMMETRIC)
        assert "Padding" in r or "SYMMETRIC" in r


# ── Interpolation ────────────────────────────────────────────────────────


class TestInterpolation:
    ALL = [
        Interpolation.NEAREST,
        Interpolation.BILINEAR,
        Interpolation.ALGO1,
        Interpolation.ALGO2,
        Interpolation.ALGO3,
        Interpolation.ALGO4,
        Interpolation.DEFAULT,
    ]

    def test_all_variants_exist(self):
        for v in self.ALL:
            assert v is not None

    def test_all_distinct(self):
        assert len({int(v) for v in self.ALL}) == len(self.ALL)

    def test_equality(self):
        assert Interpolation.BILINEAR == Interpolation.BILINEAR
        assert Interpolation.NEAREST != Interpolation.BILINEAR


# ── ComputeMode ──────────────────────────────────────────────────────────


class TestComputeMode:
    def test_variants_exist(self):
        assert ComputeMode.DEFAULT is not None
        assert ComputeMode.GPU is not None
        assert ComputeMode.VIC is not None

    def test_all_distinct(self):
        vals = {int(ComputeMode.DEFAULT), int(ComputeMode.GPU), int(ComputeMode.VIC)}
        assert len(vals) == 3

    def test_equality(self):
        assert ComputeMode.GPU == ComputeMode.GPU
        assert ComputeMode.GPU != ComputeMode.VIC
