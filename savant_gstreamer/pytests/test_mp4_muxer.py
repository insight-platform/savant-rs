"""Tests for savant_gstreamer.Mp4Muxer and Codec."""

from __future__ import annotations

import os
import tempfile

import pytest

from savant_gstreamer import Codec, Mp4Muxer


# ── Codec enum tests ──────────────────────────────────────────────────────


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


# ── Mp4Muxer construction ────────────────────────────────────────────────


class TestConstruction:
    def test_create_with_codec_enum_h264(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            assert not muxer.is_finished
            muxer.finish()
            assert muxer.is_finished
        finally:
            os.unlink(path)

    def test_create_with_codec_enum_hevc(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.HEVC, path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_create_with_string_h264(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer("h264", path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_create_with_string_hevc(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer("hevc", path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_create_with_string_h265_alias(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer("h265", path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_create_with_string_jpeg(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer("jpeg", path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_unsupported_string_raises(self):
        with pytest.raises(ValueError, match="Unknown codec"):
            Mp4Muxer("vp9", "/tmp/bad.mp4")

    def test_case_insensitive_string(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer("H264", path)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_custom_fps(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.HEVC, path, fps_num=60, fps_den=1)
            muxer.finish()
        finally:
            os.unlink(path)


# ── Push and finish ──────────────────────────────────────────────────────


class TestPushAndFinish:
    def test_push_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            data = bytes(
                [
                    0x00,
                    0x00,
                    0x00,
                    0x01,
                    0x67,
                    0x42,
                    0x00,
                    0x0A,
                    0xE9,
                    0x40,
                    0x40,
                    0x04,
                    0x00,
                    0x00,
                    0x00,
                    0x02,
                ]
            )
            muxer.push(data, pts_ns=0, duration_ns=33_333_333)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_push_multiple_frames(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            data = bytes(64)
            for i in range(5):
                muxer.push(data, pts_ns=i * 33_333_333, duration_ns=33_333_333)
            muxer.finish()
        finally:
            os.unlink(path)

    def test_push_with_dts(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            data = bytes(64)
            # Simulate B-frame reordering: PTS and DTS differ
            muxer.push(data, pts_ns=66_666_666, dts_ns=0, duration_ns=33_333_333)
            muxer.push(data, pts_ns=0, dts_ns=33_333_333, duration_ns=33_333_333)
            muxer.push(
                data, pts_ns=33_333_333, dts_ns=66_666_666, duration_ns=33_333_333
            )
            muxer.finish()
        finally:
            os.unlink(path)

    def test_push_without_duration(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.HEVC, path)
            muxer.push(bytes(32), pts_ns=0)
            muxer.finish()
        finally:
            os.unlink(path)


# ── Finalization ─────────────────────────────────────────────────────────


class TestFinalization:
    def test_double_finish_is_safe(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            muxer.finish()
            muxer.finish()  # should not raise
        finally:
            os.unlink(path)

    def test_push_after_finish_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.H264, path)
            muxer.finish()
            with pytest.raises(RuntimeError, match="finished"):
                muxer.push(bytes(16), pts_ns=0)
        finally:
            os.unlink(path)

    def test_is_finished_property(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            muxer = Mp4Muxer(Codec.HEVC, path)
            assert not muxer.is_finished
            muxer.finish()
            assert muxer.is_finished
        finally:
            os.unlink(path)
