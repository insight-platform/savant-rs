"""Tests for the NvEncoder Python API."""

from __future__ import annotations

import pytest

from deepstream_encoders import Codec, EncoderConfig, NvEncoder


# ─── Encoder creation ─────────────────────────────────────────────────────


class TestEncoderCreation:
    def test_create_hevc_encoder(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)
        assert encoder is not None

    def test_create_h264_encoder(self):
        config = EncoderConfig(Codec.H264, 320, 240)
        encoder = NvEncoder(config)
        assert encoder is not None

    def test_create_jpeg_encoder(self):
        config = EncoderConfig(Codec.JPEG, 320, 240, format="I420")
        encoder = NvEncoder(config)
        assert encoder is not None

    def test_codec_getter(self):
        config = EncoderConfig(Codec.H264, 320, 240)
        encoder = NvEncoder(config)
        assert encoder.codec == Codec.H264


# ─── Buffer acquisition ───────────────────────────────────────────────────


class TestBufferAcquisition:
    def test_acquire_surface(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        buf_ptr = encoder.acquire_surface(id=0)
        assert isinstance(buf_ptr, int)
        assert buf_ptr != 0

    def test_acquire_surface_no_id(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        buf_ptr = encoder.acquire_surface()
        assert buf_ptr != 0

    def test_nvmm_caps_str(self):
        config = EncoderConfig(Codec.HEVC, 640, 480)
        encoder = NvEncoder(config)
        caps = encoder.nvmm_caps_str()
        assert "memory:NVMM" in caps
        assert "NV12" in caps


# ─── Frame submission and encoding ────────────────────────────────────────


class TestFrameSubmission:
    def test_submit_single_frame(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        buf = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf, frame_id=0, pts_ns=0, duration_ns=33_333_333)

    def test_submit_multiple_frames(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        for i in range(5):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(
                buf,
                frame_id=i,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )

    def test_submit_null_buffer_raises(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)
        with pytest.raises(ValueError, match="null"):
            encoder.submit_frame(0, frame_id=0, pts_ns=0)


# ─── PTS validation ──────────────────────────────────────────────────────


class TestPtsValidation:
    def test_reordered_pts_raises(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        buf1 = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf1, frame_id=0, pts_ns=100)

        buf2 = encoder.acquire_surface(id=1)
        with pytest.raises(ValueError, match="PTS reordering"):
            encoder.submit_frame(buf2, frame_id=1, pts_ns=50)

    def test_equal_pts_raises(self):
        config = EncoderConfig(Codec.H264, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        buf1 = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf1, frame_id=0, pts_ns=100)

        buf2 = encoder.acquire_surface(id=1)
        with pytest.raises(ValueError, match="PTS reordering"):
            encoder.submit_frame(buf2, frame_id=1, pts_ns=100)


# ─── Encoding and finish ─────────────────────────────────────────────────


class TestEncoding:
    def test_finish_returns_encoded_frames(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        for i in range(5):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(
                buf,
                frame_id=i,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )

        remaining = encoder.finish()
        assert isinstance(remaining, list)
        assert len(remaining) > 0, "Expected at least one encoded frame"

        for frame in remaining:
            assert frame.size > 0, "Encoded frame data should not be empty"
            assert frame.codec == Codec.HEVC
            assert isinstance(frame.data, bytes)
            assert isinstance(frame.frame_id, int)
            assert isinstance(frame.pts_ns, int)

    def test_finish_h264(self):
        config = EncoderConfig(Codec.H264, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        for i in range(5):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(
                buf,
                frame_id=i,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )

        remaining = encoder.finish()
        assert len(remaining) > 0
        for frame in remaining:
            assert frame.codec == Codec.H264

    def test_finish_with_rgba_format(self):
        config = EncoderConfig(
            Codec.HEVC, 320, 240,
            format="RGBA",
            pool_size=4,
        )
        encoder = NvEncoder(config)

        for i in range(3):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(
                buf,
                frame_id=i,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )

        remaining = encoder.finish()
        assert len(remaining) > 0

    def test_pull_encoded_nonblocking(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        # Pull without submitting anything — should return None.
        frame = encoder.pull_encoded()
        assert frame is None

    def test_pull_encoded_timeout(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        frame = encoder.pull_encoded_timeout(timeout_ms=50)
        assert frame is None


# ─── Finalization ─────────────────────────────────────────────────────────


class TestFinalization:
    def test_double_finish_returns_empty(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        buf = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf, frame_id=0, pts_ns=0, duration_ns=33_333_333)

        _first = encoder.finish()
        second = encoder.finish()
        assert second == []

    def test_submit_after_finish_raises(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        encoder.finish()

        buf = encoder.acquire_surface(id=0)
        with pytest.raises(RuntimeError, match="finalized"):
            encoder.submit_frame(buf, frame_id=0, pts_ns=0)


# ─── Frame ID tracking ───────────────────────────────────────────────────


class TestFrameIdTracking:
    def test_frame_ids_preserved(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)

        expected_ids = [100, 200, 300, 400, 500]
        for i, fid in enumerate(expected_ids):
            buf = encoder.acquire_surface(id=fid)
            encoder.submit_frame(
                buf,
                frame_id=fid,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )

        remaining = encoder.finish()
        returned_ids = {f.frame_id for f in remaining}

        for fid in returned_ids:
            assert fid in expected_ids, f"Unexpected frame_id {fid}"


# ─── EncodedFrame attributes ─────────────────────────────────────────────


class TestEncodedFrame:
    def _get_one_frame(self):
        config = EncoderConfig(Codec.HEVC, 320, 240, pool_size=4)
        encoder = NvEncoder(config)
        for i in range(3):
            buf = encoder.acquire_surface(id=i)
            encoder.submit_frame(
                buf,
                frame_id=i,
                pts_ns=i * 33_333_333,
                duration_ns=33_333_333,
            )
        remaining = encoder.finish()
        assert len(remaining) > 0
        return remaining[0]

    def test_has_frame_id(self):
        frame = self._get_one_frame()
        assert isinstance(frame.frame_id, int)

    def test_has_pts_ns(self):
        frame = self._get_one_frame()
        assert isinstance(frame.pts_ns, int)

    def test_has_data(self):
        frame = self._get_one_frame()
        assert isinstance(frame.data, bytes)
        assert len(frame.data) > 0

    def test_has_size(self):
        frame = self._get_one_frame()
        assert frame.size == len(frame.data)

    def test_has_codec(self):
        frame = self._get_one_frame()
        assert frame.codec == Codec.HEVC

    def test_repr(self):
        frame = self._get_one_frame()
        r = repr(frame)
        assert "EncodedFrame" in r
        assert "bytes" in r
