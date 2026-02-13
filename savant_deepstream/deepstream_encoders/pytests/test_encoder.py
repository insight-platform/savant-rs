"""Tests for the NvEncoder Python API."""

from __future__ import annotations

import pytest

from savant_gstreamer import Codec

from deepstream_encoders import (
    EncoderConfig,
    NvEncoder,
    H264DgpuProps,
    HevcDgpuProps,
    H264Profile,
    HevcProfile,
    RateControl,
    DgpuPreset,
    TuningPreset,
    Platform,
    JetsonPresetLevel,
    H264JetsonProps,
    JpegProps,
)


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
        assert encoder.codec.name() == Codec.H264.name()


# ─── Buffer acquisition ───────────────────────────────────────────────────


class TestBufferAcquisition:
    def test_acquire_surface(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)
        buf_ptr = encoder.acquire_surface(id=0)
        assert isinstance(buf_ptr, int)
        assert buf_ptr != 0

    def test_acquire_surface_no_id(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
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
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)
        buf = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf, frame_id=0, pts_ns=0, duration_ns=33_333_333)

    def test_submit_multiple_frames(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
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
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)

        buf1 = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf1, frame_id=0, pts_ns=100)

        buf2 = encoder.acquire_surface(id=1)
        with pytest.raises(ValueError, match="PTS reordering"):
            encoder.submit_frame(buf2, frame_id=1, pts_ns=50)

    def test_equal_pts_raises(self):
        config = EncoderConfig(Codec.H264, 320, 240)
        encoder = NvEncoder(config)

        buf1 = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf1, frame_id=0, pts_ns=100)

        buf2 = encoder.acquire_surface(id=1)
        with pytest.raises(ValueError, match="PTS reordering"):
            encoder.submit_frame(buf2, frame_id=1, pts_ns=100)


# ─── Encoding and finish ─────────────────────────────────────────────────


class TestEncoding:
    def test_finish_returns_encoded_frames(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
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
            assert frame.codec.name() == Codec.HEVC.name()
            assert isinstance(frame.data, bytes)
            assert isinstance(frame.frame_id, int)
            assert isinstance(frame.pts_ns, int)

    def test_finish_h264(self):
        config = EncoderConfig(Codec.H264, 320, 240)
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
            assert frame.codec.name() == Codec.H264.name()

    def test_finish_with_rgba_format(self):
        config = EncoderConfig(
            Codec.HEVC,
            320,
            240,
            format="RGBA",
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
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)

        # Pull without submitting anything — should return None.
        frame = encoder.pull_encoded()
        assert frame is None

    def test_pull_encoded_timeout(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)

        frame = encoder.pull_encoded_timeout(timeout_ms=50)
        assert frame is None


# ─── Finalization ─────────────────────────────────────────────────────────


class TestFinalization:
    def test_double_finish_returns_empty(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)

        buf = encoder.acquire_surface(id=0)
        encoder.submit_frame(buf, frame_id=0, pts_ns=0, duration_ns=33_333_333)

        _first = encoder.finish()
        second = encoder.finish()
        assert second == []

    def test_submit_after_finish_raises(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
        encoder = NvEncoder(config)
        encoder.finish()

        buf = encoder.acquire_surface(id=0)
        with pytest.raises(RuntimeError, match="finalized"):
            encoder.submit_frame(buf, frame_id=0, pts_ns=0)


# ─── Frame ID tracking ───────────────────────────────────────────────────


class TestFrameIdTracking:
    def test_frame_ids_preserved(self):
        config = EncoderConfig(Codec.HEVC, 320, 240)
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
        config = EncoderConfig(Codec.HEVC, 320, 240)
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
        assert frame.codec.name() == Codec.HEVC.name()

    def test_repr(self):
        frame = self._get_one_frame()
        r = repr(frame)
        assert "EncodedFrame" in r
        assert "bytes" in r


# ─── Typed properties ─────────────────────────────────────────────────


class TestPropertyEnums:
    def test_rate_control(self):
        assert RateControl.VBR == RateControl.from_name("vbr")
        assert RateControl.CBR == RateControl.from_name("cbr")
        assert RateControl.CQP == RateControl.from_name("cqp")

    def test_h264_profile(self):
        assert H264Profile.BASELINE == H264Profile.from_name("baseline")
        assert H264Profile.HIGH == H264Profile.from_name("high")
        assert H264Profile.HIGH444 == H264Profile.from_name("high444")

    def test_hevc_profile(self):
        assert HevcProfile.MAIN == HevcProfile.from_name("main")
        assert HevcProfile.MAIN10 == HevcProfile.from_name("main10")

    def test_dgpu_preset(self):
        assert DgpuPreset.P1 == DgpuPreset.from_name("P1")
        assert DgpuPreset.P7 == DgpuPreset.from_name("7")

    def test_tuning_preset(self):
        assert TuningPreset.LOW_LATENCY == TuningPreset.from_name("low_latency")
        assert TuningPreset.HIGH_QUALITY == TuningPreset.from_name("high_quality")

    def test_jetson_preset_level(self):
        assert JetsonPresetLevel.FAST == JetsonPresetLevel.from_name("fast")
        assert JetsonPresetLevel.SLOW == JetsonPresetLevel.from_name("slow")

    def test_platform(self):
        assert Platform.DGPU == Platform.from_name("dgpu")
        assert Platform.JETSON == Platform.from_name("jetson")


class TestPropertyStructs:
    def test_h264_dgpu_props(self):
        props = H264DgpuProps(
            bitrate=6_000_000,
            profile=H264Profile.HIGH,
            preset=DgpuPreset.P5,
        )
        assert props.bitrate == 6_000_000
        assert props.profile == H264Profile.HIGH
        assert props.preset == DgpuPreset.P5
        assert props.cq is None  # unset fields are None

    def test_hevc_dgpu_props(self):
        props = HevcDgpuProps(
            bitrate=8_000_000,
            profile=HevcProfile.MAIN,
            control_rate=RateControl.VBR,
            cq=28,
        )
        assert props.bitrate == 8_000_000
        assert props.profile == HevcProfile.MAIN
        assert props.control_rate == RateControl.VBR
        assert props.cq == 28

    def test_h264_jetson_props(self):
        props = H264JetsonProps(
            bitrate=4_000_000,
            preset_level=JetsonPresetLevel.FAST,
            insert_sps_pps=True,
        )
        assert props.bitrate == 4_000_000
        assert props.preset_level == JetsonPresetLevel.FAST
        assert props.insert_sps_pps is True

    def test_jpeg_props(self):
        props = JpegProps(quality=95)
        assert props.quality == 95

    def test_from_pairs(self):
        props = H264DgpuProps.from_pairs(
            {
                "bitrate": "6000000",
                "profile": "high",
                "control-rate": "vbr",
            }
        )
        assert props.bitrate == 6_000_000
        assert props.profile == H264Profile.HIGH
        assert props.control_rate == RateControl.VBR

    def test_from_pairs_rejects_b_frame_key_as_unknown(self):
        """B-frame keys are not recognized fields and are rejected as unknown."""
        with pytest.raises(ValueError, match="unknown"):
            H264DgpuProps.from_pairs({"num-B-Frames": "2"})

    def test_from_pairs_rejects_unknown(self):
        with pytest.raises(ValueError):
            H264DgpuProps.from_pairs({"magic-beans": "42"})


class TestEncoderWithProperties:
    def test_create_with_hevc_dgpu_props(self):
        props = HevcDgpuProps(bitrate=6_000_000)
        config = EncoderConfig(Codec.HEVC, 320, 240, properties=props)
        encoder = NvEncoder(config)
        assert encoder.codec.name() == Codec.HEVC.name()

    def test_create_with_h264_dgpu_props(self):
        props = H264DgpuProps(bitrate=4_000_000)
        config = EncoderConfig(Codec.H264, 320, 240, properties=props)
        encoder = NvEncoder(config)
        assert encoder.codec.name() == Codec.H264.name()

    def test_encode_with_properties(self):
        props = HevcDgpuProps(bitrate=6_000_000)
        config = EncoderConfig(Codec.HEVC, 320, 240, properties=props)
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
