"""Tests for FlexibleDecoder Python bindings (savant_rs.deepstream)."""

from __future__ import annotations

import io
import queue
import threading
import time
from typing import List

import pytest

from conftest import skip_no_ds_runtime

try:
    from savant_rs.deepstream import (
        FlexibleDecoder,
        FlexibleDecoderConfig,
        FlexibleDecoderOutput,
        FrameOutput,
        SkippedOutput,
        SealedDelivery,
        init_cuda,
    )
    from savant_rs.primitives import (
        VideoFrame,
        VideoFrameContent,
        VideoFrameTranscodingMethod,
    )

    HAS_DS = True
except ImportError:
    HAS_DS = False


pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream runtime not available")


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_frame(
    source_id: str,
    width: int,
    height: int,
    codec: str = "jpeg",
    pts: int = 0,
    keyframe: bool = True,
) -> VideoFrame:
    from savant_rs.primitives import Codec

    codec_enum = Codec.from_name(codec)
    return VideoFrame(
        source_id=source_id,
        fps=(30, 1),
        width=width,
        height=height,
        content=VideoFrameContent.none(),
        transcoding_method=VideoFrameTranscodingMethod.Copy,
        codec=codec_enum,
        keyframe=keyframe,
        time_base=(1, 1_000_000_000),
        pts=pts,
    )


# ── Unit / Config tests (no GPU) ─────────────────────────────────────────


class TestConfig:
    def test_construction_defaults(self):
        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=4)
        assert cfg.source_id == "cam-1"
        assert cfg.gpu_id == 0
        assert cfg.pool_size == 4
        assert cfg.detect_buffer_limit == 30

    def test_builder_chaining(self):
        cfg = (
            FlexibleDecoderConfig("s1", gpu_id=1, pool_size=8)
            .with_idle_timeout_ms(5000)
            .with_detect_buffer_limit(50)
        )
        assert cfg.source_id == "s1"
        assert cfg.gpu_id == 1
        assert cfg.pool_size == 8
        assert cfg.detect_buffer_limit == 50

    def test_repr(self):
        cfg = FlexibleDecoderConfig("x", gpu_id=0, pool_size=2)
        r = repr(cfg)
        assert "FlexibleDecoderConfig" in r
        assert "x" in r


# ── Integration tests (GPU + CUDA) ───────────────────────────────────────


@skip_no_ds_runtime
class TestSkipReasons:
    """Submit frames that should be rejected and verify SkipReason variants."""

    def setup_method(self):
        init_cuda(0)

    def test_no_payload_skip(self):
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, on_output)

        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame)
        time.sleep(0.5)

        dec.graceful_shutdown()
        skipped = [r for r in results if r.is_skipped]
        assert len(skipped) >= 1
        s = skipped[0].as_skipped()
        assert s is not None
        assert isinstance(s, SkippedOutput)
        assert s.reason.is_no_payload

    def test_source_id_mismatch_skip(self):
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, on_output)

        frame = _make_frame("cam-WRONG", 320, 240, codec="jpeg", pts=0)
        jpeg_data = _make_jpeg(320, 240)
        dec.submit(frame, jpeg_data)
        time.sleep(0.5)

        dec.graceful_shutdown()
        skipped = [r for r in results if r.is_skipped]
        assert len(skipped) >= 1
        s = skipped[0].as_skipped()
        assert s is not None
        assert s.reason.is_source_id_mismatch
        assert "cam-WRONG" in s.reason.detail

    def test_garbage_jpeg_skip(self):
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, on_output)

        garbage = bytes([0xDE, 0xAD] * 100)
        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, garbage)
        time.sleep(0.5)

        dec.graceful_shutdown()
        skipped = [r for r in results if r.is_skipped]
        assert len(skipped) >= 1
        s = skipped[0].as_skipped()
        assert s is not None
        assert s.reason.is_invalid_payload

    def test_parameter_change_during_detection_skip(self):
        """Codec change while pre-RAP H.264 packets are still buffered must
        surface as ``ParameterChangeDuringDetection`` (not
        ``WaitingForKeyframe``) so consumers can distinguish keyframe-pending
        stalls from input-parameter renegotiations.
        """
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, on_output)

        dummy_h264 = bytes([0] * 64)
        for i in range(3):
            f = _make_frame("cam-1", 320, 240, codec="h264", pts=i * 33_333_333)
            dec.submit(f, dummy_h264)

        jpeg_data = _make_jpeg(320, 240)
        jpeg_frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=3 * 33_333_333)
        dec.submit(jpeg_frame, jpeg_data)
        time.sleep(0.5)

        dec.graceful_shutdown()
        skipped = [r.as_skipped() for r in results if r.is_skipped]
        skipped = [s for s in skipped if s is not None]
        param_changes = [
            s for s in skipped if s.reason.is_parameter_change_during_detection
        ]
        assert len(param_changes) == 3, (
            f"expected 3 ParameterChangeDuringDetection skips, "
            f"got {len(param_changes)}; all skips: "
            f"{[(s.reason.__repr__()) for s in skipped]}"
        )
        for s in param_changes:
            assert s.reason.parameter_change_codec_changed is True
            assert s.reason.parameter_change_dims_changed is False
            assert not s.reason.is_waiting_for_keyframe
            detail = s.reason.detail
            assert detail is not None
            assert "codec_changed=true" in detail
            assert "dims_changed=false" in detail


@skip_no_ds_runtime
class TestJpegDecode:
    """Submit a real JPEG and verify decoded Frame output."""

    def setup_method(self):
        init_cuda(0)

    def test_valid_jpeg_produces_frame(self):
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=4)
        dec = FlexibleDecoder(cfg, on_output)

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=42_000_000)
        dec.submit(frame, jpeg_data)
        time.sleep(1.0)

        dec.graceful_shutdown()
        frame_outputs = [r for r in results if r.is_frame]
        assert len(frame_outputs) >= 1, f"Expected Frame outputs, got: {results}"

        fo = frame_outputs[0].as_frame()
        assert fo is not None
        assert isinstance(fo, FrameOutput)
        assert fo.frame.source_id == "cam-1"
        df = fo.decoded_frame
        assert df is not None
        assert df.pts_ns == 42_000_000

    def test_take_delivery_returns_sealed(self):
        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=4)
        dec = FlexibleDecoder(cfg, on_output)

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)
        time.sleep(1.0)

        dec.graceful_shutdown()
        frame_outputs = [r for r in results if r.is_frame]
        assert len(frame_outputs) >= 1

        fo = frame_outputs[0].as_frame()
        assert fo is not None
        sealed = fo.take_delivery()
        assert sealed is not None
        assert isinstance(sealed, SealedDelivery)

        second = fo.take_delivery()
        assert second is None, "take_delivery must return None on second call"


@skip_no_ds_runtime
class TestSealedDeliveryCrossThread:
    """Full SealedDelivery lifecycle across Python threads."""

    def setup_method(self):
        init_cuda(0)

    def test_unseal_blocks_until_output_dropped(self):
        delivery_queue: queue.Queue = queue.Queue()
        frame_count = 0

        def on_output(out: FlexibleDecoderOutput):
            nonlocal frame_count
            if out.is_frame:
                fo = out.as_frame()
                sealed = fo.take_delivery()
                assert sealed is not None
                assert not sealed.is_released()
                delivery_queue.put(sealed)
                frame_count += 1

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=4)
        dec = FlexibleDecoder(cfg, on_output)

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)

        unseal_results = []
        errors = []

        def consumer():
            try:
                sealed = delivery_queue.get(timeout=10)
                vf, buf = sealed.unseal(timeout_ms=15_000)
                assert vf.source_id == "cam-1"
                unseal_results.append((vf, buf))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=consumer)
        t.start()
        t.join(timeout=20)
        assert not t.is_alive(), "consumer thread hung"

        assert not errors, f"Consumer thread raised: {errors}"
        assert len(unseal_results) == 1
        assert frame_count >= 1
        dec.graceful_shutdown()

    def test_try_unseal_before_release(self):
        delivery_queue: queue.Queue = queue.Queue()
        hold_frame_output: list = []

        def on_output(out: FlexibleDecoderOutput):
            if out.is_frame:
                fo = out.as_frame()
                sealed = fo.take_delivery()
                assert sealed is not None
                delivery_queue.put(sealed)
                hold_frame_output.append(fo)

        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=4)
        dec = FlexibleDecoder(cfg, on_output)

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-1", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)
        time.sleep(1.0)

        sealed = delivery_queue.get(timeout=5)

        result = sealed.try_unseal()
        assert result is None, (
            "try_unseal should return None while FrameOutput is still held"
        )

        hold_frame_output.clear()
        time.sleep(0.2)

        result = sealed.try_unseal()

        dec.graceful_shutdown()


@skip_no_ds_runtime
class TestLifecycle:
    """Decoder lifecycle: shutdown and double-shutdown."""

    def setup_method(self):
        init_cuda(0)

    def test_shutdown_no_submit(self):
        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, lambda out: None)
        dec.shutdown()
        assert repr(dec) == "FlexibleDecoder(shut_down)"

    def test_submit_after_shutdown_raises(self):
        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, lambda out: None)
        dec.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            dec.submit(_make_frame("cam-1", 320, 240))

    def test_graceful_shutdown_after_shutdown_raises(self):
        cfg = FlexibleDecoderConfig("cam-1", gpu_id=0, pool_size=2)
        dec = FlexibleDecoder(cfg, lambda out: None)
        dec.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            dec.graceful_shutdown()
