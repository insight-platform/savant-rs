"""Tests for EncoderConfig."""

from __future__ import annotations


from deepstream_nvbufsurface import VideoFormat
from savant_gstreamer import Codec

from deepstream_encoders import EncoderConfig


class TestEncoderConfigDefaults:
    def test_basic_creation(self):
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        assert config is not None

    def test_default_codec(self):
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        assert config.codec.name() == Codec.HEVC.name()

    def test_default_dimensions(self):
        config = EncoderConfig(Codec.H264, 1280, 720)
        assert config.width == 1280
        assert config.height == 720

    def test_default_format(self):
        config = EncoderConfig(Codec.HEVC, 640, 480)
        assert config.format.name() == VideoFormat.NV12.name()


class TestEncoderConfigCustom:
    def test_custom_format(self):
        config = EncoderConfig(Codec.H264, 640, 480, format="RGBA")
        assert config.format.name() == VideoFormat.RGBA.name()

    def test_custom_fps(self):
        config = EncoderConfig(Codec.HEVC, 640, 480, fps_num=60, fps_den=1)
        assert config is not None

    def test_with_properties(self):
        from deepstream_encoders import HevcDgpuProps

        props = HevcDgpuProps(bitrate=4_000_000)
        config = EncoderConfig(Codec.HEVC, 640, 480, properties=props)
        assert config is not None


class TestEncoderConfigRepr:
    def test_repr_contains_codec(self):
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        r = repr(config)
        assert "Hevc" in r or "hevc" in r or "HEVC" in r

    def test_repr_contains_dimensions(self):
        config = EncoderConfig(Codec.H264, 1280, 720)
        r = repr(config)
        assert "1280" in r
        assert "720" in r
