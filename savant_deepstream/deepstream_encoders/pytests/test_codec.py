"""Tests for the Codec enum."""

from __future__ import annotations

import pytest

from deepstream_encoders import Codec


class TestCodecValues:
    def test_h264(self):
        assert Codec.H264 is not None

    def test_hevc(self):
        assert Codec.HEVC is not None

    def test_jpeg(self):
        assert Codec.JPEG is not None

    def test_av1(self):
        assert Codec.AV1 is not None


class TestCodecFromName:
    def test_h264(self):
        assert Codec.from_name("h264") == Codec.H264

    def test_h264_uppercase(self):
        assert Codec.from_name("H264") == Codec.H264

    def test_hevc(self):
        assert Codec.from_name("hevc") == Codec.HEVC

    def test_h265_alias(self):
        assert Codec.from_name("h265") == Codec.HEVC

    def test_jpeg(self):
        assert Codec.from_name("jpeg") == Codec.JPEG

    def test_av1(self):
        assert Codec.from_name("av1") == Codec.AV1

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown codec"):
            Codec.from_name("vp9")


class TestCodecName:
    def test_h264_name(self):
        assert Codec.H264.name() == "h264"

    def test_hevc_name(self):
        assert Codec.HEVC.name() == "hevc"

    def test_jpeg_name(self):
        assert Codec.JPEG.name() == "jpeg"

    def test_av1_name(self):
        assert Codec.AV1.name() == "av1"


class TestCodecRepr:
    def test_repr_h264(self):
        assert "H264" in repr(Codec.H264)

    def test_repr_hevc(self):
        assert "HEVC" in repr(Codec.HEVC)
