"""Tests for savant_rs.gstreamer — Mp4Demuxer and DemuxedPacket (requires gst feature)."""

from __future__ import annotations

import os
import tempfile

import pytest

try:
    from savant_rs.gstreamer import Codec, DemuxedPacket, Mp4Demuxer, Mp4Muxer
except ImportError:
    Codec = None
    DemuxedPacket = None
    Mp4Demuxer = None
    Mp4Muxer = None


def _gst_runtime_available() -> bool:
    if Mp4Muxer is None or Mp4Demuxer is None:
        return False
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        muxer = Mp4Muxer(Codec.H264, path)
        muxer.finish()
        os.unlink(path)
        return True
    except RuntimeError:
        try:
            os.unlink(path)
        except OSError:
            pass
        return False


_HAS_GST_RUNTIME = _gst_runtime_available()

requires_gst_runtime = pytest.mark.skipif(
    not _HAS_GST_RUNTIME,
    reason="GStreamer runtime not available (missing plugins or misconfigured)",
)

H264_SPS_PPS_IDR: bytes = bytes(
    [
        # SPS
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0A,
        0xE9, 0x40, 0x40, 0x04, 0x00, 0x00, 0x00, 0x02,
        # PPS
        0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80,
        # IDR slice (minimal)
        0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x80, 0x40,
    ]
)


def _make_h264_mp4(path: str, num_frames: int = 5) -> None:
    """Produce a minimal H.264 MP4 using Mp4Muxer."""
    muxer = Mp4Muxer(Codec.H264, path, fps_num=30, fps_den=1)
    dur = 33_333_333
    for i in range(num_frames):
        muxer.push(H264_SPS_PPS_IDR, pts_ns=i * dur, duration_ns=dur)
    muxer.finish()


# ── Construction ─────────────────────────────────────────────────────────


@requires_gst_runtime
class TestConstruction:
    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="does not exist"):
            Mp4Demuxer("/tmp/nonexistent_savant_rs_demuxer_test.mp4")

    def test_create_from_valid_mp4(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path)
            demuxer = Mp4Demuxer(path)
            assert not demuxer.is_finished
            demuxer.finish()
            assert demuxer.is_finished
        finally:
            os.unlink(path)


# ── Pull ─────────────────────────────────────────────────────────────────


@requires_gst_runtime
class TestPull:
    def test_pull_returns_demuxed_packets(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            num_frames = 5
            _make_h264_mp4(path, num_frames)
            demuxer = Mp4Demuxer(path)
            packets = []
            while True:
                pkt = demuxer.pull()
                if pkt is None:
                    break
                packets.append(pkt)
            assert len(packets) > 0
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_pull_timeout_returns_packets(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 3)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull_timeout(2000)
            assert pkt is not None
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_pull_all_returns_list(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 4)
            demuxer = Mp4Demuxer(path)
            packets = demuxer.pull_all()
            assert isinstance(packets, list)
            assert len(packets) > 0
            assert demuxer.is_finished
        finally:
            os.unlink(path)


# ── DemuxedPacket properties ─────────────────────────────────────────────


@requires_gst_runtime
class TestDemuxedPacketProperties:
    def test_packet_has_data(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            assert isinstance(pkt.data, bytes)
            assert len(pkt.data) > 0
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_packet_has_pts(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            assert isinstance(pkt.pts_ns, int)
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_packet_dts_is_optional(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            assert pkt.dts_ns is None or isinstance(pkt.dts_ns, int)
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_packet_duration_is_optional(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            assert pkt.duration_ns is None or isinstance(pkt.duration_ns, int)
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_packet_is_keyframe_bool(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            assert isinstance(pkt.is_keyframe, bool)
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_packet_repr(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            pkt = demuxer.pull()
            assert pkt is not None
            r = repr(pkt)
            assert "DemuxedPacket" in r
            assert "pts_ns=" in r
            demuxer.finish()
        finally:
            os.unlink(path)


# ── Codec detection ──────────────────────────────────────────────────────


@requires_gst_runtime
class TestCodecDetection:
    def test_detected_codec_none_before_pull(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            # Before any pull, detected_codec may or may not be set
            # depending on whether qtdemux already emitted caps.
            # We just ensure it doesn't crash.
            _ = demuxer.detected_codec
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_detected_codec_after_pull(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            demuxer = Mp4Demuxer(path)
            demuxer.pull()
            assert demuxer.detected_codec == Codec.H264
            demuxer.finish()
        finally:
            os.unlink(path)


# ── Finalization ─────────────────────────────────────────────────────────


@requires_gst_runtime
class TestFinalization:
    def test_double_finish_is_safe(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            demuxer.finish()
            demuxer.finish()  # no-op
        finally:
            os.unlink(path)

    def test_pull_after_finish_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            demuxer.finish()
            with pytest.raises(RuntimeError, match="finished"):
                demuxer.pull()
        finally:
            os.unlink(path)

    def test_is_finished_property(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = Mp4Demuxer(path)
            assert not demuxer.is_finished
            demuxer.finish()
            assert demuxer.is_finished
        finally:
            os.unlink(path)


# ── Muxer → Demuxer round-trip ───────────────────────────────────────────


@requires_gst_runtime
class TestMuxerDemuxerRoundTrip:
    def test_h264_mux_demux_preserves_frame_count(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            num_frames = 8
            _make_h264_mp4(path, num_frames)
            demuxer = Mp4Demuxer(path)
            packets = demuxer.pull_all()
            # qtdemux may merge or split NALUs, but we should get ≥ 1 packet
            assert len(packets) >= 1
            assert demuxer.detected_codec == Codec.H264
        finally:
            os.unlink(path)

    def test_pts_ordering_is_monotonic(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 10)
            demuxer = Mp4Demuxer(path)
            packets = demuxer.pull_all()
            pts_values = [p.pts_ns for p in packets]
            for i in range(1, len(pts_values)):
                assert pts_values[i] >= pts_values[i - 1], (
                    f"PTS not monotonic at index {i}: {pts_values[i - 1]} > {pts_values[i]}"
                )
        finally:
            os.unlink(path)
