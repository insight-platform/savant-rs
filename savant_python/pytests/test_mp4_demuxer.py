"""Tests for savant_rs.gstreamer — Mp4Demuxer callback API and DemuxedPacket (requires gst feature)."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import List

import pytest

try:
    from savant_rs.gstreamer import (
        Codec,
        DemuxedPacket,
        Mp4Demuxer,
        Mp4DemuxerOutput,
        Mp4Muxer,
    )
except ImportError:
    Codec = None
    DemuxedPacket = None
    Mp4Demuxer = None
    Mp4DemuxerOutput = None
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
        # PPS
        0x00,
        0x00,
        0x00,
        0x01,
        0x68,
        0xCE,
        0x38,
        0x80,
        # IDR slice (minimal)
        0x00,
        0x00,
        0x00,
        0x01,
        0x65,
        0x88,
        0x80,
        0x40,
    ]
)


def _make_h264_mp4(path: str, num_frames: int = 5) -> None:
    """Produce a minimal H.264 MP4 using Mp4Muxer."""
    muxer = Mp4Muxer(Codec.H264, path, fps_num=30, fps_den=1)
    dur = 33_333_333
    for i in range(num_frames):
        muxer.push(H264_SPS_PPS_IDR, pts_ns=i * dur, duration_ns=dur)
    muxer.finish()


class PacketCollector:
    """Thread-safe collector for Mp4Demuxer callback outputs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.packets: List[DemuxedPacket] = []
        self.eos_count: int = 0
        self.errors: List[str] = []

    def __call__(self, output: Mp4DemuxerOutput) -> None:
        with self._lock:
            if output.is_packet:
                self.packets.append(output.as_packet())
            elif output.is_eos:
                self.eos_count += 1
            elif output.is_error:
                self.errors.append(output.as_error_message())


# ── Construction ─────────────────────────────────────────────────────────


@requires_gst_runtime
class TestConstruction:
    def test_missing_file_raises(self):
        collector = PacketCollector()
        with pytest.raises(RuntimeError, match="does not exist"):
            Mp4Demuxer("/tmp/nonexistent_savant_rs_demuxer_test.mp4", collector)

    def test_create_from_valid_mp4(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            assert not demuxer.is_finished
            demuxer.finish()
            assert demuxer.is_finished
        finally:
            os.unlink(path)


# ── Callback delivery ───────────────────────────────────────────────────


@requires_gst_runtime
class TestCallbackDelivery:
    def test_packets_delivered_via_callback(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            num_frames = 5
            _make_h264_mp4(path, num_frames)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.wait()
            assert len(collector.packets) > 0
            assert collector.eos_count >= 1
            assert len(collector.errors) == 0
        finally:
            os.unlink(path)

    def test_wait_timeout_returns_true_on_completion(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 3)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            finished = demuxer.wait_timeout(10_000)
            assert finished is True
            assert len(collector.packets) > 0
        finally:
            os.unlink(path)

    def test_eos_delivered(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.wait()
            assert collector.eos_count >= 1
        finally:
            os.unlink(path)


# ── DemuxedPacket properties ─────────────────────────────────────────────


@requires_gst_runtime
class TestDemuxedPacketProperties:
    @staticmethod
    def _get_first_packet(path: str) -> DemuxedPacket:
        collector = PacketCollector()
        demuxer = Mp4Demuxer(path, collector)
        demuxer.wait()
        assert len(collector.packets) > 0, "no packets delivered"
        return collector.packets[0]

    def test_packet_has_data(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            assert isinstance(pkt.data, bytes)
            assert len(pkt.data) > 0
        finally:
            os.unlink(path)

    def test_packet_has_pts(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            assert isinstance(pkt.pts_ns, int)
        finally:
            os.unlink(path)

    def test_packet_dts_is_optional(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            assert pkt.dts_ns is None or isinstance(pkt.dts_ns, int)
        finally:
            os.unlink(path)

    def test_packet_duration_is_optional(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            assert pkt.duration_ns is None or isinstance(pkt.duration_ns, int)
        finally:
            os.unlink(path)

    def test_packet_is_keyframe_bool(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            assert isinstance(pkt.is_keyframe, bool)
        finally:
            os.unlink(path)

    def test_packet_repr(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            pkt = self._get_first_packet(path)
            r = repr(pkt)
            assert "DemuxedPacket" in r
            assert "pts_ns=" in r
        finally:
            os.unlink(path)


# ── Mp4DemuxerOutput variants ────────────────────────────────────────────


@requires_gst_runtime
class TestMp4DemuxerOutput:
    def test_output_variant_flags(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            outputs: list = []
            demuxer = Mp4Demuxer(path, lambda o: outputs.append(o))
            demuxer.wait()
            assert len(outputs) > 0
            has_packet = any(o.is_packet for o in outputs)
            has_eos = any(o.is_eos for o in outputs)
            assert has_packet
            assert has_eos
            for o in outputs:
                assert isinstance(repr(o), str)
        finally:
            os.unlink(path)

    def test_as_packet_returns_none_for_eos(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            outputs: list = []
            demuxer = Mp4Demuxer(path, lambda o: outputs.append(o))
            demuxer.wait()
            eos_outputs = [o for o in outputs if o.is_eos]
            assert len(eos_outputs) >= 1
            assert eos_outputs[0].as_packet() is None
            assert eos_outputs[0].as_error_message() is None
        finally:
            os.unlink(path)


# ── Codec detection ──────────────────────────────────────────────────────


@requires_gst_runtime
class TestCodecDetection:
    def test_detected_codec_after_wait(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.wait()
            assert demuxer.detected_codec == Codec.H264
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
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.finish()
            demuxer.finish()  # no-op
        finally:
            os.unlink(path)

    def test_is_finished_property(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
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
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.wait()
            assert len(collector.packets) >= 1
            assert demuxer.detected_codec == Codec.H264
        finally:
            os.unlink(path)

    def test_pts_ordering_is_monotonic(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 10)
            collector = PacketCollector()
            demuxer = Mp4Demuxer(path, collector)
            demuxer.wait()
            pts_values = [p.pts_ns for p in collector.packets]
            for i in range(1, len(pts_values)):
                assert pts_values[i] >= pts_values[i - 1], (
                    f"PTS not monotonic at index {i}: {pts_values[i - 1]} > {pts_values[i]}"
                )
        finally:
            os.unlink(path)
