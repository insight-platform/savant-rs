"""Tests for savant_rs.nvtracker (DeepStream feature)."""

from __future__ import annotations

import os
import time
from typing import Dict, List

import pytest

try:
    from savant_rs.deepstream import (
        BufferGenerator,
        MemType,
        SavantIdMetaKind,
        VideoFormat,
        init_cuda,
    )
    from savant_rs.nvinfer import Roi
    from savant_rs.nvtracker import (
        NvTracker,
        NvTrackerConfig,
        TrackedFrame,
        TrackingIdResetMode,
        TrackState,
        TrackerOutput,
    )
    from savant_rs.primitives.geometry import RBBox

    HAS_NVTRACKER = True
except ImportError:
    HAS_NVTRACKER = False

pytestmark = pytest.mark.skipif(
    not HAS_NVTRACKER, reason="savant_rs built without deepstream / nvtracker"
)

NVTRACKER_ASSETS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "savant_deepstream",
    "nvtracker",
    "assets",
)
DEFAULT_LL = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
IOU_YML = os.path.join(NVTRACKER_ASSETS, "config_tracker_IOU.yml")

_GPU_READY = False


def _ensure_gpu() -> None:
    global _GPU_READY
    if not _GPU_READY:
        init_cuda(0)
        _GPU_READY = True


def _require_tracker_assets() -> None:
    if not os.path.isfile(DEFAULT_LL) or not os.path.isfile(IOU_YML):
        pytest.skip("DeepStream tracker library or IOU yaml not present")


def _make_frame(
    source: str,
    rois: Dict[int, List[Roi]],
    w: int = 320,
    h: int = 240,
    pts_ns: int | None = None,
) -> TrackedFrame:
    """Create a TrackedFrame with a single-surface buffer."""
    _ensure_gpu()
    gen = BufferGenerator(
        VideoFormat.RGBA, w, h, gpu_id=0, mem_type=MemType.DEFAULT
    )
    buf = gen.acquire(None)
    if pts_ns is not None:
        buf.pts_ns = pts_ns
    return TrackedFrame(source, buf, rois)


def _recv_tracking(tracker: NvTracker) -> TrackerOutput:
    """Drain non-tracking outputs until a ``TrackerOutput`` is available."""
    while True:
        out = tracker.recv()
        if out.is_tracking:
            t = out.as_tracking()
            assert t is not None
            return t
        if out.is_eos:
            raise AssertionError(
                f"unexpected EOS: {out.eos_source_id!r}"
            )
        if out.is_error:
            raise RuntimeError(out.error_message or "NvTracker error")


def test_enums_and_track_state() -> None:
    assert int(TrackingIdResetMode.NONE) == 0
    assert int(TrackState.ACTIVE) == 1


def test_nv_tracker_config_paths() -> None:
    _require_tracker_assets()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
        tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    )
    assert os.path.basename(cfg.ll_lib_file) == "libnvds_nvmultiobjecttracker.so"
    assert cfg.ll_config_file.endswith("config_tracker_IOU.yml")


def test_config_framework_fields_py() -> None:
    _require_tracker_assets()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        input_channel_capacity=8,
        output_channel_capacity=12,
        drain_poll_interval_ms=50,
    )
    assert cfg.input_channel_capacity == 8
    assert cfg.output_channel_capacity == 12
    assert cfg.drain_poll_interval_ms == 50


def test_single_source_tracking_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
    )
    tracker = NvTracker(cfg)

    bbox_a = RBBox.ltwh(40.0, 40.0, 80.0, 60.0)
    bbox_b = RBBox.ltwh(180.0, 100.0, 70.0, 70.0)

    ids = [(SavantIdMetaKind.FRAME, 1)]
    frame0 = _make_frame("cam-1", {0: [Roi(1, bbox_a), Roi(2, bbox_b)]})
    tracker.submit([frame0], ids)
    o0 = _recv_tracking(tracker)
    assert len(o0.current_tracks) == 2
    id_a = o0.current_tracks[0].object_id
    id_b = o0.current_tracks[1].object_id

    bbox_a2 = RBBox.ltwh(42.0, 40.0, 80.0, 60.0)
    bbox_b2 = RBBox.ltwh(182.0, 100.0, 70.0, 70.0)
    frame1 = _make_frame("cam-1", {0: [Roi(1, bbox_a2), Roi(2, bbox_b2)]})
    tracker.submit([frame1], [(SavantIdMetaKind.FRAME, 2)])
    o1 = _recv_tracking(tracker)
    ids_out = {t.object_id for t in o1.current_tracks}
    assert id_a in ids_out and id_b in ids_out

    tracker.shutdown()


def test_multi_source_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
    )
    tracker = NvTracker(cfg)

    roi = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
    frame_a = _make_frame("cam-a", {0: [roi]})
    frame_b = _make_frame("cam-b", {0: [roi]})
    tracker.submit(
        [frame_a, frame_b],
        [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
    )
    out = _recv_tracking(tracker)
    assert len(out.current_tracks) == 2
    by = {t.source_id: t.object_id for t in out.current_tracks}
    assert by["cam-a"] != by["cam-b"]

    tracker.shutdown()


def test_same_source_multi_frame_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
    )
    tracker = NvTracker(cfg)

    r0 = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
    r1 = Roi(1, RBBox.ltwh(55.0, 52.0, 60.0, 60.0))
    f0 = _make_frame("cam-a", {0: [r0]})
    f1 = _make_frame("cam-a", {0: [r1]})
    tracker.submit(
        [f0, f1],
        [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
    )
    out = _recv_tracking(tracker)
    assert len(out.current_tracks) >= 1

    tracker.shutdown()


def test_reset_stream_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
        tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    )
    tracker = NvTracker(cfg)

    roi = Roi(1, RBBox.ltwh(60.0, 60.0, 90.0, 90.0))
    ids = [(SavantIdMetaKind.FRAME, 1)]

    tracker.submit([_make_frame("cam-1", {0: [roi]})], ids)
    o0 = _recv_tracking(tracker)
    id0 = o0.current_tracks[0].object_id
    tracker.submit([_make_frame("cam-1", {0: [roi]})], ids)
    o1 = _recv_tracking(tracker)
    assert o1.current_tracks[0].object_id == id0

    tracker.reset_stream("cam-1")
    tracker.submit([_make_frame("cam-1", {0: [roi]})], ids)
    o2 = _recv_tracking(tracker)
    assert o2.current_tracks[0].object_id != id0

    tracker.shutdown()


def test_class_id_tracking_py() -> None:
    """Verify that class_id propagates through the tracker."""
    _require_tracker_assets()
    _ensure_gpu()

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
    )
    tracker = NvTracker(cfg)

    roi_a = Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))
    roi_b = Roi(2, RBBox.ltwh(180.0, 100.0, 70.0, 70.0))
    frame = _make_frame("cam-1", {0: [roi_a], 1: [roi_b]})

    tracker.submit([frame], [(SavantIdMetaKind.FRAME, 1)])
    out = _recv_tracking(tracker)
    assert len(out.current_tracks) == 2

    class_ids = {t.class_id for t in out.current_tracks}
    assert class_ids == {0, 1}, f"expected {{0, 1}}, got {class_ids}"

    tracker.shutdown()


def test_send_eos_round_trip_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
    )
    t = NvTracker(cfg)
    roi = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
    t.submit([_make_frame("eos-src", {0: [roi]})], [(SavantIdMetaKind.FRAME, 1)])
    _ = _recv_tracking(t)
    t.send_eos("eos-src")
    while True:
        out = t.recv()
        if out.is_eos:
            assert out.eos_source_id == "eos-src"
            break
        if out.is_tracking:
            continue
        if out.is_event:
            continue
        if out.is_error:
            pytest.fail(out.error_message)
    t.shutdown()


def test_is_failed_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
    )
    t = NvTracker(cfg)
    assert t.is_failed() is False
    t.shutdown()


def test_resolution_change_auto_reset_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
        tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    )
    t = NvTracker(cfg)
    roi = Roi(1, RBBox.ltwh(40.0, 40.0, 50.0, 50.0))
    t.submit([_make_frame("cam-r", {0: [roi]}, 320, 240)], [(SavantIdMetaKind.FRAME, 1)])
    o0 = _recv_tracking(t)
    id0 = o0.current_tracks[0].object_id
    t.submit([_make_frame("cam-r", {0: [roi]}, 640, 480)], [(SavantIdMetaKind.FRAME, 2)])
    o1 = _recv_tracking(t)
    id1 = o1.current_tracks[0].object_id
    assert id1 != id0
    t.shutdown()


def test_pts_regression_auto_reset_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
        tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    )
    t = NvTracker(cfg)
    roi = Roi(1, RBBox.ltwh(40.0, 40.0, 50.0, 50.0))
    t.submit(
        [_make_frame("cam-p", {0: [roi]}, pts_ns=1_000_000)],
        [(SavantIdMetaKind.FRAME, 1)],
    )
    o0 = _recv_tracking(t)
    id0 = o0.current_tracks[0].object_id
    t.submit(
        [_make_frame("cam-p", {0: [roi]}, pts_ns=500_000)],
        [(SavantIdMetaKind.FRAME, 2)],
    )
    o1 = _recv_tracking(t)
    id1 = o1.current_tracks[0].object_id
    assert id1 != id0
    t.shutdown()


def test_recv_timeout_none_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
    )
    t = NvTracker(cfg)
    # Drain GStreamer startup events (stream-start, caps, segment).
    time.sleep(0.1)
    while t.try_recv() is not None:
        pass
    assert t.recv_timeout(10) is None
    t.shutdown()


def test_try_recv_none_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
    )
    t = NvTracker(cfg)
    # Drain GStreamer startup events (stream-start, caps, segment).
    time.sleep(0.1)
    while t.try_recv() is not None:
        pass
    assert t.try_recv() is None
    t.shutdown()


def test_graceful_shutdown_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()
    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=2,
    )
    t = NvTracker(cfg)
    rois = {
        0: [
            Roi(
                1,
                RBBox.ltwh(40.0, 40.0, 80.0, 60.0),
            )
        ]
    }
    drained = t.graceful_shutdown(2_000)
    assert isinstance(drained, list)
    assert len(drained) >= 0
    with pytest.raises(RuntimeError, match="shut down|ShuttingDown"):
        t.submit(
            [_make_frame("cam-gs2", rois)],
            [(SavantIdMetaKind.FRAME, 2)],
        )
