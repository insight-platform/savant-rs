"""Tests for savant_rs.nvtracker NvTrackerBatchingOperator (DeepStream feature)."""

from __future__ import annotations

import os
import threading
from typing import List

import pytest

try:
    from savant_rs.deepstream import (
        BufferGenerator,
        MemType,
        VideoFormat,
        init_cuda,
    )
    from savant_rs.nvinfer import Roi
    from savant_rs.nvtracker import (
        NvTracker,
        NvTrackerBatchingOperator,
        NvTrackerBatchingOperatorConfig,
        NvTrackerConfig,
        TrackerBatchFormationResult,
        TrackingIdResetMode,
    )
    from savant_rs.primitives import (
        VideoFrame,
        VideoFrameContent,
        VideoFrameTranscodingMethod,
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


def _make_frame(source: str, width: int = 320, height: int = 240) -> VideoFrame:
    return VideoFrame(
        source_id=source,
        fps=(30, 1),
        width=width,
        height=height,
        content=VideoFrameContent.none(),
        transcoding_method=VideoFrameTranscodingMethod.Copy,
    )


def _make_buffer(width: int = 320, height: int = 240):
    _ensure_gpu()
    gen = BufferGenerator(
        VideoFormat.RGBA, width, height, gpu_id=0, mem_type=MemType.DEFAULT
    )
    return gen.acquire(None)


def _make_tracker_config(max_batch_size: int = 4) -> NvTrackerConfig:
    return NvTrackerConfig(
        DEFAULT_LL,
        IOU_YML,
        VideoFormat.RGBA,
        tracker_width=320,
        tracker_height=240,
        max_batch_size=max_batch_size,
        tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    )


def test_batching_operator_config_fields() -> None:
    _require_tracker_assets()
    cfg = NvTrackerBatchingOperatorConfig(
        max_batch_size=3,
        max_batch_wait_ms=150,
        nvtracker_config=_make_tracker_config(max_batch_size=8),
    )
    assert cfg.max_batch_size == 3
    assert cfg.max_batch_wait_ms == 150
    assert cfg.nvtracker_config.ll_config_file.endswith("config_tracker_IOU.yml")


def test_batch_formation_result_construction() -> None:
    roi = Roi(7, RBBox.ltwh(30.0, 20.0, 40.0, 50.0))
    result = TrackerBatchFormationResult(
        ids=[],
        rois=[{0: [roi]}],
    )
    assert isinstance(result.ids, list)
    assert "rois_len=1" in repr(result)


def test_single_source_batching_and_sealed_deliveries() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    callback_done = threading.Event()
    sealed_holder = []
    frame_holder = []

    def batch_formation(frames):
        # IDs are optional for this operator API; Batch(batch_id) is injected internally.
        rois = [{0: [Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))]} for _ in frames]
        return TrackerBatchFormationResult(ids=[], rois=rois)

    def result_callback(output):
        assert output.is_tracking
        frames = output.frames
        assert len(frames) == 1
        fo = frames[0]
        assert fo.frame.source_id == "cam-1"
        assert len(fo.tracked_objects) == 1

        sealed = output.take_deliveries()
        assert sealed is not None
        assert sealed.try_unseal() is None
        sealed_holder.append(sealed)
        frame_holder.append(fo.frame)
        callback_done.set()

    op = NvTrackerBatchingOperator(
        config=NvTrackerBatchingOperatorConfig(
            max_batch_size=1,
            max_batch_wait_ms=5000,
            nvtracker_config=_make_tracker_config(max_batch_size=4),
        ),
        batch_formation_callback=batch_formation,
        result_callback=result_callback,
    )
    try:
        op.add_frame(_make_frame("cam-1"), _make_buffer())
        assert callback_done.wait(timeout=30), "callback not invoked within 30 s"
        sealed = sealed_holder[0]
        assert sealed.is_released()
        pairs = sealed.unseal()
        assert len(pairs) == 1
        frame_out, _buffer_out = pairs[0]
        assert frame_out.source_id == frame_holder[0].source_id == "cam-1"
    finally:
        op.shutdown()


def test_multi_source_batching() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    callback_done = threading.Event()
    sources_holder: List[List[str]] = []

    def batch_formation(frames):
        rois = []
        for _ in frames:
            rois.append({0: [Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))]})
        return TrackerBatchFormationResult(ids=[], rois=rois)

    def result_callback(output):
        assert output.is_tracking
        sources_holder.append([fo.frame.source_id for fo in output.frames])
        callback_done.set()

    op = NvTrackerBatchingOperator(
        config=NvTrackerBatchingOperatorConfig(
            max_batch_size=2,
            max_batch_wait_ms=5000,
            nvtracker_config=_make_tracker_config(max_batch_size=4),
        ),
        batch_formation_callback=batch_formation,
        result_callback=result_callback,
    )
    try:
        op.add_frame(_make_frame("cam-a"), _make_buffer())
        op.add_frame(_make_frame("cam-b"), _make_buffer())
        assert callback_done.wait(timeout=30), "callback not invoked within 30 s"
        got = set(sources_holder[0])
        assert got == {"cam-a", "cam-b"}
    finally:
        op.shutdown()


def test_flush_partial_batch() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    callback_done = threading.Event()

    def batch_formation(frames):
        rois = [{0: [Roi(1, RBBox.ltwh(60.0, 60.0, 90.0, 90.0))]} for _ in frames]
        return TrackerBatchFormationResult(ids=[], rois=rois)

    def result_callback(output):
        assert output.is_tracking
        callback_done.set()

    op = NvTrackerBatchingOperator(
        config=NvTrackerBatchingOperatorConfig(
            max_batch_size=8,
            max_batch_wait_ms=60_000,
            nvtracker_config=_make_tracker_config(max_batch_size=8),
        ),
        batch_formation_callback=batch_formation,
        result_callback=result_callback,
    )
    try:
        op.add_frame(_make_frame("cam-flush"), _make_buffer())
        op.flush()
        assert callback_done.wait(timeout=30), "flush did not submit partial batch"
    finally:
        op.shutdown()


def test_existing_nvtracker_still_importable() -> None:
    # Smoke check to ensure adding batching API does not break existing class exposure.
    _ = NvTracker


def test_batching_send_eos_py() -> None:
    _require_tracker_assets()
    _ensure_gpu()

    tracking_done = threading.Event()
    eos_done = threading.Event()

    def batch_formation(frames):
        rois = [{0: [Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))]} for _ in frames]
        return TrackerBatchFormationResult(ids=[], rois=rois)

    def result_callback(output):
        if output.is_eos:
            assert output.eos_source_id == "eos-cam"
            eos_done.set()
            return
        if output.is_tracking:
            tracking_done.set()
            return
        if output.is_error:
            pytest.fail(output.error_message)

    op = NvTrackerBatchingOperator(
        config=NvTrackerBatchingOperatorConfig(
            max_batch_size=1,
            max_batch_wait_ms=5000,
            nvtracker_config=_make_tracker_config(max_batch_size=4),
        ),
        batch_formation_callback=batch_formation,
        result_callback=result_callback,
    )
    try:
        op.add_frame(_make_frame("eos-cam"), _make_buffer())
        assert tracking_done.wait(timeout=30)
        op.send_eos("eos-cam")
        assert eos_done.wait(timeout=30)
    finally:
        op.shutdown()
