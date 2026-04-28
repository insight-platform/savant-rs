"""Tests for savant_rs.gstreamer.UriDemuxer (requires gst feature).

Parallel to test_mp4_demuxer.py. All tests build a local temp MP4 via
Mp4Muxer and consume it via ``file://`` URI, which is the lowest-friction
URI scheme that is guaranteed to work in any GStreamer build.

The last test class (``TestMp4DemuxerParity``) cross-checks the ground
truth: Mp4Demuxer and UriDemuxer consuming the same file must yield the
same VideoInfo, packet count, per-packet PTS / keyframe flag, and the
same normalized H.264 slice payload bytes.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import List

import pytest

try:
    from savant_rs.gstreamer import (
        Codec,
        DemuxedPacket,
        Mp4Demuxer,
        Mp4DemuxerOutput,
        Mp4Muxer,
        UriDemuxer,
        VideoInfo,
    )
except ImportError:
    Codec = None
    DemuxedPacket = None
    Mp4Demuxer = None
    Mp4DemuxerOutput = None
    Mp4Muxer = None
    UriDemuxer = None
    VideoInfo = None


def _gst_runtime_available() -> bool:
    if Mp4Muxer is None or UriDemuxer is None:
        return False
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        muxer = Mp4Muxer(Codec.H264, path)
        muxer.finish()
        return True
    except RuntimeError:
        return False
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


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
    muxer = Mp4Muxer(Codec.H264, path, fps_num=30, fps_den=1)
    dur = 33_333_333
    for i in range(num_frames):
        muxer.push(H264_SPS_PPS_IDR, pts_ns=i * dur, duration_ns=dur)
    muxer.finish()


def _file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


class PacketCollector:
    """Thread-safe collector for UriDemuxer callback outputs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.packets: List[DemuxedPacket] = []
        self.eos_count: int = 0
        self.errors: List[str] = []
        self.stream_infos: List[VideoInfo] = []

    def __call__(self, output: Mp4DemuxerOutput) -> None:
        with self._lock:
            if output.is_packet:
                self.packets.append(output.as_packet())
            elif output.is_eos:
                self.eos_count += 1
            elif output.is_error:
                self.errors.append(output.as_error_message())
            elif output.is_stream_info:
                self.stream_infos.append(output.as_stream_info())


# ── Construction ─────────────────────────────────────────────────────────


@requires_gst_runtime
class TestConstruction:
    def test_empty_uri_raises(self):
        with pytest.raises(RuntimeError):
            UriDemuxer("", lambda _o: None)

    def test_malformed_uri_raises(self):
        with pytest.raises(RuntimeError):
            UriDemuxer("not-a-uri", lambda _o: None)

    def test_create_from_file_uri(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path)
            demuxer = UriDemuxer(_file_uri(path), PacketCollector())
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
            _make_h264_mp4(path, 5)
            collector = PacketCollector()
            demuxer = UriDemuxer(_file_uri(path), collector)
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
            demuxer = UriDemuxer(_file_uri(path), collector)
            assert demuxer.wait_timeout(10_000) is True
            assert len(collector.packets) > 0
        finally:
            os.unlink(path)

    def test_stream_info_fires_once_before_first_packet(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 4)
            outputs: list = []
            demuxer = UriDemuxer(_file_uri(path), lambda o: outputs.append(o))
            demuxer.wait()

            stream_info_indices = [i for i, o in enumerate(outputs) if o.is_stream_info]
            packet_indices = [i for i, o in enumerate(outputs) if o.is_packet]
            meaningful = [
                i for i, o in enumerate(outputs) if o.is_stream_info or o.is_packet
            ]
            assert len(stream_info_indices) == 1
            assert len(packet_indices) > 0
            assert stream_info_indices[0] == min(meaningful)
        finally:
            os.unlink(path)


# ── Properties: bin_properties ──────────────────────────────────────────


@requires_gst_runtime
class TestBinProperties:
    def test_known_bin_property_accepted(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 3)
            collector = PacketCollector()
            # urisourcebin has a "buffer-size" property (guint).
            demuxer = UriDemuxer(
                _file_uri(path),
                collector,
                bin_properties={"buffer-size": 8_388_608},
            )
            assert demuxer.wait_timeout(10_000) is True
            assert len(collector.packets) > 0
            assert len(collector.errors) == 0
        finally:
            os.unlink(path)

    def test_unknown_bin_property_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            with pytest.raises(RuntimeError):
                UriDemuxer(
                    _file_uri(path),
                    lambda _o: None,
                    bin_properties={"no-such-property-xyz": 1},
                )
        finally:
            os.unlink(path)

    def test_invalid_property_value_type_raises_type_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            with pytest.raises(TypeError):
                UriDemuxer(
                    _file_uri(path),
                    lambda _o: None,
                    bin_properties={"buffer-size": [1, 2, 3]},  # list is unsupported
                )
        finally:
            os.unlink(path)


# ── Properties: source_properties (via source-setup signal) ────────────


@requires_gst_runtime
class TestSourceProperties:
    def test_unknown_source_property_surfaces_error_to_callback(self):
        """An unknown source-element property must NOT abort construction;
        it surfaces as an error output to the callback (so pipeline-wide
        side-effects stay contained)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            collector = PacketCollector()
            demuxer = UriDemuxer(
                _file_uri(path),
                collector,
                source_properties={"definitely-not-a-property-foo": 42},
            )
            demuxer.wait_timeout(5_000)
            assert any(
                "definitely-not-a-property-foo" in msg for msg in collector.errors
            )
        finally:
            os.unlink(path)


# ── VideoInfo + codec detection ────────────────────────────────────────


@requires_gst_runtime
class TestVideoInfo:
    def test_wait_for_video_info_returns_info(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        demuxer = None
        try:
            _make_h264_mp4(path, 3)
            demuxer = UriDemuxer(_file_uri(path), lambda _o: None)
            info = demuxer.wait_for_video_info(5_000)
            assert info is not None
            assert isinstance(info, VideoInfo)
            assert info.codec == Codec.H264
            assert info.width > 0
            assert info.height > 0
            assert demuxer.video_info is not None
            assert demuxer.video_info.codec == Codec.H264
        finally:
            if demuxer is not None:
                demuxer.finish()
            os.unlink(path)

    def test_detected_codec_is_h264(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 2)
            collector = PacketCollector()
            demuxer = UriDemuxer(_file_uri(path), collector)
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
            demuxer = UriDemuxer(_file_uri(path), PacketCollector())
            demuxer.finish()
            demuxer.finish()
        finally:
            os.unlink(path)

    def test_is_finished_property(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 1)
            demuxer = UriDemuxer(_file_uri(path), PacketCollector())
            assert not demuxer.is_finished
            demuxer.finish()
            assert demuxer.is_finished
        finally:
            os.unlink(path)


# ── Round-trip ─────────────────────────────────────────────────────────


@requires_gst_runtime
class TestRoundTrip:
    def test_pts_monotonic(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 8)
            collector = PacketCollector()
            demuxer = UriDemuxer(_file_uri(path), collector)
            demuxer.wait()
            pts = [p.pts_ns for p in collector.packets]
            assert len(pts) >= 1
            for i in range(1, len(pts)):
                assert pts[i] >= pts[i - 1]
        finally:
            os.unlink(path)


# ── Parity with Mp4Demuxer (ground truth) ───────────────────────────────


def _iter_annex_b_nals(buf: bytes) -> List[bytes]:
    """Split an Annex-B byte stream into raw NAL units (no start codes).

    Supports both 3-byte (``00 00 01``) and 4-byte (``00 00 00 01``) start
    codes. Mirrors the logic in savant_gstreamer's Rust parity test.
    """
    out: List[bytes] = []
    n = len(buf)

    def find_start(k: int):
        while k + 3 <= n:
            if buf[k] == 0 and buf[k + 1] == 0 and buf[k + 2] == 1:
                return (k, 3)
            if (
                k + 4 <= n
                and buf[k] == 0
                and buf[k + 1] == 0
                and buf[k + 2] == 0
                and buf[k + 3] == 1
            ):
                return (k, 4)
            k += 1
        return None

    first = find_start(0)
    if first is None:
        return out
    start, prefix = first
    i = start + prefix
    while i < n:
        nxt = find_start(i)
        if nxt is None:
            out.append(bytes(buf[i:]))
            break
        next_start, next_prefix = nxt
        out.append(bytes(buf[i:next_start]))
        i = next_start + next_prefix
    return out


def _h264_slice_nals(au: bytes) -> List[bytes]:
    """Strip AUD / SPS / PPS / SEI / filler NALs from an H.264 access unit.

    The two demuxers insert configuration NAL preambles slightly
    differently (parsebin vs qtdemux + h264parse), but the actual slice
    NALs must be byte-identical. See savant_gstreamer/tests/demuxer_parity.rs.
    """
    out: List[bytes] = []
    for nal in _iter_annex_b_nals(au):
        if not nal:
            continue
        # H.264 NAL header: forbidden_zero_bit (1), nal_ref_idc (2), nal_unit_type (5).
        nal_type = nal[0] & 0x1F
        # 9 = AUD, 7 = SPS, 8 = PPS, 6 = SEI, 12 = FillerData,
        # 10 = SeqEnd, 11 = StreamEnd, 13 = SpsExt, 15 = SubsetSps.
        if nal_type in (6, 7, 8, 9, 10, 11, 12, 13, 15):
            continue
        out.append(nal)
    return out


def _collect_via_mp4(path: str):
    collector = PacketCollector()
    demuxer = Mp4Demuxer(path, collector, parsed=True)
    demuxer.wait()
    return collector.packets, demuxer.video_info, demuxer.detected_codec


def _collect_via_uri(path: str):
    collector = PacketCollector()
    demuxer = UriDemuxer(_file_uri(path), collector, parsed=True)
    demuxer.wait()
    return collector.packets, demuxer.video_info, demuxer.detected_codec


@requires_gst_runtime
class TestMp4DemuxerParity:
    """Cross-check UriDemuxer against Mp4Demuxer on the same file."""

    def _assert_video_info_equal(self, a: VideoInfo, b: VideoInfo) -> None:
        assert a.codec == b.codec
        assert a.width == b.width
        assert a.height == b.height
        assert a.framerate_num == b.framerate_num
        assert a.framerate_den == b.framerate_den

    def test_h264_parity(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            _make_h264_mp4(path, 10)

            mp4_pkts, mp4_info, mp4_codec = _collect_via_mp4(path)
            uri_pkts, uri_info, uri_codec = _collect_via_uri(path)

            assert mp4_codec == uri_codec == Codec.H264
            assert mp4_info is not None and uri_info is not None
            self._assert_video_info_equal(mp4_info, uri_info)

            assert len(mp4_pkts) == len(uri_pkts), (
                f"packet count mismatch: mp4={len(mp4_pkts)} uri={len(uri_pkts)}"
            )
            assert len(mp4_pkts) >= 1

            for idx, (r, u) in enumerate(zip(mp4_pkts, uri_pkts)):
                assert r.pts_ns == u.pts_ns, f"packet[{idx}] pts_ns differ"
                assert r.is_keyframe == u.is_keyframe, (
                    f"packet[{idx}] is_keyframe differ"
                )
                if r.dts_ns is not None and u.dts_ns is not None:
                    assert r.dts_ns == u.dts_ns, f"packet[{idx}] dts_ns differ"
                if r.duration_ns is not None and u.duration_ns is not None:
                    assert r.duration_ns == u.duration_ns, (
                        f"packet[{idx}] duration_ns differ"
                    )

                r_slices = _h264_slice_nals(r.data)
                u_slices = _h264_slice_nals(u.data)
                assert r_slices == u_slices, (
                    f"packet[{idx}] slice NALs differ"
                )
                assert r_slices, f"packet[{idx}] has no slice NALs"
        finally:
            os.unlink(path)
