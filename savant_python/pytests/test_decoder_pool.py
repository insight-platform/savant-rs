"""Tests for FlexibleDecoderPool Python bindings (savant_rs.deepstream)."""

from __future__ import annotations

import io
import threading
import time
from typing import List

import pytest

from conftest import skip_no_ds_runtime

try:
    from savant_rs.deepstream import (
        EvictionDecision,
        FlexibleDecoderOutput,
        FlexibleDecoderPool,
        FlexibleDecoderPoolConfig,
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
    pts: int = 0,
) -> VideoFrame:
    from savant_rs.primitives import Codec

    codec_enum = Codec.from_name("jpeg")
    return VideoFrame(
        source_id=source_id,
        fps=(30, 1),
        width=width,
        height=height,
        content=VideoFrameContent.none(),
        transcoding_method=VideoFrameTranscodingMethod.Copy,
        codec=codec_enum,
        keyframe=True,
        time_base=(1, 1_000_000_000),
        pts=pts,
    )


class OutputCollector:
    """Thread-safe output collector for decoder callbacks."""

    def __init__(self) -> None:
        self._outputs: List[FlexibleDecoderOutput] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

    def __call__(self, out: FlexibleDecoderOutput) -> None:
        with self._lock:
            self._outputs.append(out)
            self._event.set()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return sum(1 for o in self._outputs if o.is_frame)

    @property
    def outputs(self) -> List[FlexibleDecoderOutput]:
        with self._lock:
            return list(self._outputs)

    def wait_for_frames(self, count: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.frame_count < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {count} frames (got {self.frame_count})"
                )
            self._event.wait(timeout=min(remaining, 0.1))
            self._event.clear()


# ── Config tests (no GPU) ────────────────────────────────────────────────


class TestConfig:
    def test_construction(self):
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=30000)
        assert cfg.gpu_id == 0
        assert cfg.pool_size == 4
        assert cfg.eviction_ttl_ms == 30000
        assert cfg.detect_buffer_limit == 30

    def test_builder_chaining(self):
        cfg = (
            FlexibleDecoderPoolConfig(gpu_id=1, pool_size=8, eviction_ttl_ms=60000)
            .with_idle_timeout_ms(500)
            .with_detect_buffer_limit(50)
        )
        assert cfg.gpu_id == 1
        assert cfg.pool_size == 8
        assert cfg.idle_timeout_ms == 500
        assert cfg.detect_buffer_limit == 50
        assert cfg.eviction_ttl_ms == 60000

    def test_repr(self):
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=5000)
        r = repr(cfg)
        assert "FlexibleDecoderPoolConfig" in r
        assert "5000" in r


class TestEvictionDecision:
    def test_class_constants(self):
        assert EvictionDecision.EVICT.is_evict
        assert not EvictionDecision.EVICT.is_keep
        assert EvictionDecision.KEEP.is_keep
        assert not EvictionDecision.KEEP.is_evict

    def test_repr(self):
        assert "EVICT" in repr(EvictionDecision.EVICT)
        assert "KEEP" in repr(EvictionDecision.KEEP)

    def test_equality(self):
        assert EvictionDecision.EVICT == EvictionDecision.EVICT
        assert EvictionDecision.KEEP == EvictionDecision.KEEP


# ── GPU tests ─────────────────────────────────────────────────────────────


@skip_no_ds_runtime
class TestPoolDecode:
    @classmethod
    def setup_class(cls):
        init_cuda(0)

    def test_submit_two_streams(self):
        collector = OutputCollector()
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, collector)

        jpeg = _make_jpeg()
        f1 = _make_frame("cam-1", 320, 240, pts=0)
        f2 = _make_frame("cam-2", 320, 240, pts=0)
        pool.submit(f1, jpeg)
        pool.submit(f2, jpeg)

        collector.wait_for_frames(2)
        assert collector.frame_count >= 2
        pool.shutdown()

    def test_submit_same_stream_twice(self):
        collector = OutputCollector()
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, collector)

        jpeg = _make_jpeg()
        f1 = _make_frame("cam-1", 320, 240, pts=0)
        f2 = _make_frame("cam-1", 320, 240, pts=1_000_000)
        pool.submit(f1, jpeg)
        pool.submit(f2, jpeg)

        collector.wait_for_frames(2)
        assert collector.frame_count == 2
        pool.shutdown()


@skip_no_ds_runtime
class TestEvictionDefault:
    @classmethod
    def setup_class(cls):
        init_cuda(0)

    def test_auto_evict_after_ttl(self):
        collector = OutputCollector()
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=200)
        pool = FlexibleDecoderPool(cfg, collector)

        jpeg = _make_jpeg()
        f = _make_frame("cam-1", 320, 240)
        pool.submit(f, jpeg)
        collector.wait_for_frames(1)

        time.sleep(1.5)

        f2 = _make_frame("cam-1", 320, 240, pts=1_000_000)
        pool.submit(f2, jpeg)
        collector.wait_for_frames(2)
        pool.shutdown()


@skip_no_ds_runtime
class TestEvictionCallback:
    @classmethod
    def setup_class(cls):
        init_cuda(0)

    def test_keep_decision(self):
        collector = OutputCollector()
        eviction_calls: List[str] = []
        eviction_lock = threading.Lock()

        def on_eviction(source_id: str) -> EvictionDecision:
            with eviction_lock:
                eviction_calls.append(source_id)
            return EvictionDecision.KEEP

        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=200)
        pool = FlexibleDecoderPool(cfg, collector, eviction_callback=on_eviction)

        jpeg = _make_jpeg()
        f = _make_frame("cam-1", 320, 240)
        pool.submit(f, jpeg)
        collector.wait_for_frames(1)

        time.sleep(1.5)
        with eviction_lock:
            assert len(eviction_calls) >= 1

        f2 = _make_frame("cam-1", 320, 240, pts=1_000_000)
        pool.submit(f2, jpeg)
        collector.wait_for_frames(2)
        pool.shutdown()

    def test_evict_decision(self):
        collector = OutputCollector()
        eviction_calls: List[str] = []
        eviction_lock = threading.Lock()

        def on_eviction(source_id: str) -> EvictionDecision:
            with eviction_lock:
                eviction_calls.append(source_id)
            return EvictionDecision.EVICT

        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=200)
        pool = FlexibleDecoderPool(cfg, collector, eviction_callback=on_eviction)

        jpeg = _make_jpeg()
        f = _make_frame("cam-1", 320, 240)
        pool.submit(f, jpeg)
        collector.wait_for_frames(1)

        time.sleep(1.5)
        with eviction_lock:
            assert len(eviction_calls) >= 1

        f2 = _make_frame("cam-1", 320, 240)
        pool.submit(f2, jpeg)
        collector.wait_for_frames(2)
        pool.shutdown()


@skip_no_ds_runtime
class TestLifecycle:
    @classmethod
    def setup_class(cls):
        init_cuda(0)

    def test_graceful_shutdown(self):
        collector = OutputCollector()
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, collector)

        jpeg = _make_jpeg()
        f = _make_frame("cam-1", 320, 240)
        pool.submit(f, jpeg)
        collector.wait_for_frames(1)

        pool.graceful_shutdown()

    def test_shutdown(self):
        collector = OutputCollector()
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, collector)

        jpeg = _make_jpeg()
        f = _make_frame("cam-1", 320, 240)
        pool.submit(f, jpeg)
        collector.wait_for_frames(1)

        pool.shutdown()

    def test_submit_after_shutdown(self):
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, lambda _: None)
        pool.shutdown()

        f = _make_frame("cam-1", 320, 240)
        with pytest.raises(RuntimeError):
            pool.submit(f, None)

    def test_double_shutdown(self):
        cfg = FlexibleDecoderPoolConfig(gpu_id=0, pool_size=4, eviction_ttl_ms=60000)
        pool = FlexibleDecoderPool(cfg, lambda _: None)
        pool.shutdown()
        with pytest.raises(RuntimeError):
            pool.shutdown()
