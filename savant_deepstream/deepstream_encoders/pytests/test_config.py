"""Tests for EncoderConfig."""

from __future__ import annotations

import pytest

from deepstream_encoders import Codec, EncoderConfig


class TestEncoderConfigDefaults:
    def test_basic_creation(self):
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        assert config is not None

    def test_default_codec(self):
        config = EncoderConfig(Codec.HEVC, 1920, 1080)
        assert config.codec == Codec.HEVC

    def test_default_dimensions(self):
        config = EncoderConfig(Codec.H264, 1280, 720)
        assert config.width == 1280
        assert config.height == 720

    def test_default_format(self):
        config = EncoderConfig(Codec.HEVC, 640, 480)
        assert config.format == "NV12"


class TestEncoderConfigCustom:
    def test_custom_format(self):
        config = EncoderConfig(Codec.H264, 640, 480, format="RGBA")
        assert config.format == "RGBA"

    def test_custom_fps(self):
        config = EncoderConfig(Codec.HEVC, 640, 480, fps_num=60, fps_den=1)
        assert config is not None

    def test_custom_pool_size(self):
        config = EncoderConfig(Codec.HEVC, 640, 480, pool_size=8)
        assert config is not None

    def test_encoder_properties(self):
        config = EncoderConfig(
            Codec.HEVC, 640, 480,
            encoder_properties={"bitrate": "4000000"},
        )
        assert config is not None


class TestEncoderConfigBFrameRejection:
    def test_rejects_b_frames_property(self):
        with pytest.raises(ValueError, match="B-frames"):
            EncoderConfig(
                Codec.HEVC, 640, 480,
                encoder_properties={"num-B-Frames": "2"},
            )

    def test_rejects_b_frames_lowercase(self):
        with pytest.raises(ValueError, match="B-frames"):
            EncoderConfig(
                Codec.H264, 640, 480,
                encoder_properties={"b-frames": "1"},
            )

    def test_rejects_bframes_variant(self):
        with pytest.raises(ValueError, match="B-frames"):
            EncoderConfig(
                Codec.HEVC, 640, 480,
                encoder_properties={"bframes": "3"},
            )


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
