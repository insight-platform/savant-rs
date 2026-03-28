"""Tests for savant_rs.nvtracker (DeepStream feature)."""

from __future__ import annotations

import os
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
) -> TrackedFrame:
    """Create a TrackedFrame with a single-surface buffer."""
    _ensure_gpu()
    gen = BufferGenerator(
        VideoFormat.RGBA, w, h, gpu_id=0, mem_type=MemType.DEFAULT
    )
    buf = gen.acquire(None)
    return TrackedFrame(source, buf, rois)


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


def test_single_source_tracking_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    out_holder: list[TrackerOutput] = []

    def _cb(out: TrackerOutput) -> None:
        out_holder.append(out)

    cfg = NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=4,
    )
    tracker = NvTracker(cfg, _cb)

    bbox_a = RBBox.ltwh(40.0, 40.0, 80.0, 60.0)
    bbox_b = RBBox.ltwh(180.0, 100.0, 70.0, 70.0)

    ids = [(SavantIdMetaKind.FRAME, 1)]
    frame0 = _make_frame("cam-1", {0: [Roi(1, bbox_a), Roi(2, bbox_b)]})
    o0 = tracker.track_sync([frame0], ids)
    assert len(o0.current_tracks) == 2
    id_a = o0.current_tracks[0].object_id
    id_b = o0.current_tracks[1].object_id

    bbox_a2 = RBBox.ltwh(42.0, 40.0, 80.0, 60.0)
    bbox_b2 = RBBox.ltwh(182.0, 100.0, 70.0, 70.0)
    frame1 = _make_frame("cam-1", {0: [Roi(1, bbox_a2), Roi(2, bbox_b2)]})
    o1 = tracker.track_sync([frame1], [(SavantIdMetaKind.FRAME, 2)])
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
    tracker = NvTracker(cfg, lambda _o: None)

    roi = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
    frame_a = _make_frame("cam-a", {0: [roi]})
    frame_b = _make_frame("cam-b", {0: [roi]})
    out = tracker.track_sync(
        [frame_a, frame_b],
        [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
    )
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
    tracker = NvTracker(cfg, lambda _o: None)

    r0 = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
    r1 = Roi(1, RBBox.ltwh(55.0, 52.0, 60.0, 60.0))
    f0 = _make_frame("cam-a", {0: [r0]})
    f1 = _make_frame("cam-a", {0: [r1]})
    out = tracker.track_sync(
        [f0, f1],
        [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
    )
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
    tracker = NvTracker(cfg, lambda _o: None)

    roi = Roi(1, RBBox.ltwh(60.0, 60.0, 90.0, 90.0))
    ids = [(SavantIdMetaKind.FRAME, 1)]

    o0 = tracker.track_sync([_make_frame("cam-1", {0: [roi]})], ids)
    id0 = o0.current_tracks[0].object_id
    o1 = tracker.track_sync([_make_frame("cam-1", {0: [roi]})], ids)
    assert o1.current_tracks[0].object_id == id0

    tracker.reset_stream("cam-1")
    o2 = tracker.track_sync([_make_frame("cam-1", {0: [roi]})], ids)
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
    tracker = NvTracker(cfg, lambda _o: None)

    roi_a = Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))
    roi_b = Roi(2, RBBox.ltwh(180.0, 100.0, 70.0, 70.0))
    frame = _make_frame("cam-1", {0: [roi_a], 1: [roi_b]})

    out = tracker.track_sync([frame], [(SavantIdMetaKind.FRAME, 1)])
    assert len(out.current_tracks) == 2

    class_ids = {t.class_id for t in out.current_tracks}
    assert class_ids == {0, 1}, f"expected {{0, 1}}, got {class_ids}"

    tracker.shutdown()
