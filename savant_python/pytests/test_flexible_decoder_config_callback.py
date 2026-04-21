"""Tests for FlexibleDecoder decoder_config callback + DecoderConfig bindings."""

from __future__ import annotations

import io
import platform
import threading
import time
from typing import List

import pytest

from conftest import skip_no_ds_runtime

IS_JETSON = platform.machine() == "aarch64"

try:
    from savant_rs.deepstream import (
        Av1DecoderConfig,
        DecoderConfig,
        FlexibleDecoder,
        FlexibleDecoderConfig,
        FlexibleDecoderOutput,
        FlexibleDecoderPoolConfig,
        H264DecoderConfig,
        H264StreamFormat,
        HevcDecoderConfig,
        HevcStreamFormat,
        JpegBackend,
        JpegDecoderConfig,
        PngDecoderConfig,
        RawRgbaDecoderConfig,
        RawRgbDecoderConfig,
        Vp8DecoderConfig,
        Vp9DecoderConfig,
        init_cuda,
    )
    from savant_rs.gstreamer import Codec
    from savant_rs.primitives import (
        VideoFrame,
        VideoFrameContent,
        VideoFrameTranscodingMethod,
    )

    # CudadecMemtype is only exposed on dGPU; on Jetson the ImportError is
    # the expected, documented behaviour.
    if not IS_JETSON:
        from savant_rs.deepstream import CudadecMemtype
    else:
        CudadecMemtype = None  # type: ignore[assignment]

    HAS_DS = True
except ImportError:
    HAS_DS = False
    # Placeholders so module-level ``@pytest.mark.parametrize`` decorators can
    # still be evaluated at collection time on builds without the deepstream
    # feature. The actual tests are skipped via ``pytestmark`` below.
    Vp8DecoderConfig = None  # type: ignore[assignment]
    Vp9DecoderConfig = None  # type: ignore[assignment]
    Av1DecoderConfig = None  # type: ignore[assignment]
    CudadecMemtype = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream runtime not available")


# ── Helpers ───────────────────────────────────────────────────────────────


def _poll_until(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.02,
) -> bool:
    """Poll ``predicate`` until truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval)


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


# ── Typed variants: builders and getters ─────────────────────────────────


class TestTypedVariants:
    def test_h264_builder_shared(self):
        """Platform-independent H264 tunables must round-trip on both arches."""
        cfg = (
            H264DecoderConfig(H264StreamFormat.BYTE_STREAM)
            .with_num_extra_surfaces(4)
            .with_drop_frame_interval(2)
        )
        assert cfg.stream_format == H264StreamFormat.BYTE_STREAM
        assert cfg.num_extra_surfaces == 4
        assert cfg.drop_frame_interval == 2
        assert "H264DecoderConfig" in repr(cfg)

    @pytest.mark.skipif(IS_JETSON, reason="dGPU-only nvv4l2 tunables")
    def test_h264_builder_dgpu_tunables(self):
        cfg = (
            H264DecoderConfig(H264StreamFormat.BYTE_STREAM)
            .with_cudadec_memtype(CudadecMemtype.PINNED)
            .with_low_latency_mode(True)
        )
        assert cfg.cudadec_memtype == CudadecMemtype.PINNED
        assert cfg.low_latency_mode is True
        assert not hasattr(cfg, "enable_max_performance")
        assert not hasattr(cfg, "low_latency")

    @pytest.mark.skipif(not IS_JETSON, reason="Jetson-only nvv4l2 tunables")
    def test_h264_builder_jetson_tunables(self):
        cfg = (
            H264DecoderConfig(H264StreamFormat.BYTE_STREAM)
            .with_enable_max_performance(True)
            .with_low_latency(True)
        )
        assert cfg.enable_max_performance is True
        assert cfg.low_latency is True
        assert not hasattr(cfg, "cudadec_memtype")
        assert not hasattr(cfg, "low_latency_mode")

    def test_hevc_builder_shared(self):
        cfg = (
            HevcDecoderConfig(HevcStreamFormat.HVC1)
            .with_num_extra_surfaces(2)
            .with_drop_frame_interval(3)
        )
        assert cfg.stream_format == HevcStreamFormat.HVC1
        assert cfg.num_extra_surfaces == 2
        assert cfg.drop_frame_interval == 3
        assert "HevcDecoderConfig" in repr(cfg)

    @pytest.mark.skipif(IS_JETSON, reason="dGPU-only nvv4l2 tunables")
    def test_hevc_builder_dgpu_tunables(self):
        cfg = HevcDecoderConfig(HevcStreamFormat.HVC1).with_low_latency_mode(False)
        assert cfg.low_latency_mode is False
        assert cfg.cudadec_memtype is None

    @pytest.mark.skipif(not IS_JETSON, reason="Jetson-only nvv4l2 tunables")
    def test_hevc_builder_jetson_tunables(self):
        cfg = HevcDecoderConfig(HevcStreamFormat.HVC1).with_low_latency(False)
        assert cfg.low_latency is False
        assert cfg.enable_max_performance is None

    @pytest.mark.parametrize(
        "ctor,name",
        [
            (Vp8DecoderConfig, "Vp8DecoderConfig"),
            (Vp9DecoderConfig, "Vp9DecoderConfig"),
            (Av1DecoderConfig, "Av1DecoderConfig"),
        ],
    )
    def test_nvv4l2_variants_builder_shared(self, ctor, name):
        cfg = ctor().with_num_extra_surfaces(3).with_drop_frame_interval(1)
        assert cfg.num_extra_surfaces == 3
        assert cfg.drop_frame_interval == 1
        assert name in repr(cfg)

    @pytest.mark.skipif(IS_JETSON, reason="dGPU-only nvv4l2 tunables")
    @pytest.mark.parametrize(
        "ctor", [Vp8DecoderConfig, Vp9DecoderConfig, Av1DecoderConfig]
    )
    def test_nvv4l2_variants_builder_dgpu(self, ctor):
        cfg = (
            ctor()
            .with_cudadec_memtype(CudadecMemtype.UNIFIED)
            .with_low_latency_mode(True)
        )
        assert cfg.cudadec_memtype == CudadecMemtype.UNIFIED
        assert cfg.low_latency_mode is True

    @pytest.mark.skipif(not IS_JETSON, reason="Jetson-only nvv4l2 tunables")
    @pytest.mark.parametrize(
        "ctor", [Vp8DecoderConfig, Vp9DecoderConfig, Av1DecoderConfig]
    )
    def test_nvv4l2_variants_builder_jetson(self, ctor):
        cfg = ctor().with_enable_max_performance(True).with_low_latency(True)
        assert cfg.enable_max_performance is True
        assert cfg.low_latency is True

    def test_jpeg_factories(self):
        gpu = JpegDecoderConfig.gpu()
        cpu = JpegDecoderConfig.cpu()
        assert gpu.backend == JpegBackend.GPU
        assert cpu.backend == JpegBackend.CPU
        assert "JpegDecoderConfig" in repr(gpu)

    def test_png_default(self):
        cfg = PngDecoderConfig()
        assert "PngDecoderConfig" in repr(cfg)

    def test_raw_rgba_rgb(self):
        rgba = RawRgbaDecoderConfig(640, 480)
        assert rgba.width == 640 and rgba.height == 480
        rgb = RawRgbDecoderConfig(320, 240)
        assert rgb.width == 320 and rgb.height == 240

    @pytest.mark.skipif(IS_JETSON, reason="CudadecMemtype is dGPU-only")
    def test_cudadec_memtype_enum(self):
        assert int(CudadecMemtype.DEVICE) == 0
        assert int(CudadecMemtype.PINNED) == 1
        assert int(CudadecMemtype.UNIFIED) == 2
        assert CudadecMemtype.DEVICE != CudadecMemtype.PINNED
        assert "DEVICE" in repr(CudadecMemtype.DEVICE)

    @pytest.mark.skipif(
        not IS_JETSON, reason="Jetson omits CudadecMemtype from the module"
    )
    def test_cudadec_memtype_absent_on_jetson(self):
        import savant_rs.deepstream as ds

        assert not hasattr(ds, "CudadecMemtype")


# ── Umbrella DecoderConfig round-tripping ────────────────────────────────


class TestDecoderConfigRoundTrip:
    def test_codec_query(self):
        dc = DecoderConfig.from_jpeg(JpegDecoderConfig.gpu())
        assert dc.codec() == Codec.Jpeg
        dc = DecoderConfig.from_h264(H264DecoderConfig(H264StreamFormat.AVC))
        assert dc.codec() == Codec.H264

    def test_as_correct_variant_returns_config(self):
        inner = Vp8DecoderConfig().with_num_extra_surfaces(7)
        if IS_JETSON:
            inner = inner.with_low_latency(True)
        else:
            inner = inner.with_low_latency_mode(True)
        dc = DecoderConfig.from_vp8(inner)
        rt = dc.as_vp8()
        assert rt is not None
        assert rt.num_extra_surfaces == 7
        if IS_JETSON:
            assert rt.low_latency is True
        else:
            assert rt.low_latency_mode is True

    def test_as_wrong_variant_returns_none(self):
        dc = DecoderConfig.from_png(PngDecoderConfig())
        assert dc.as_vp8() is None
        assert dc.as_h264() is None
        assert dc.as_png() is not None

    def test_with_variant_replaces_inner(self):
        dc = DecoderConfig.from_jpeg(JpegDecoderConfig.gpu())
        new_inner = Vp9DecoderConfig()
        if IS_JETSON:
            new_inner = new_inner.with_low_latency(True)
        else:
            new_inner = new_inner.with_low_latency_mode(True)
        dc2 = dc.with_vp9(new_inner)
        assert dc2.codec() == Codec.Vp9
        rt = dc2.as_vp9()
        assert rt is not None
        if IS_JETSON:
            assert rt.low_latency is True
        else:
            assert rt.low_latency_mode is True

    def test_repr_contains_variant(self):
        dc = DecoderConfig.from_hevc(HevcDecoderConfig(HevcStreamFormat.HEV1))
        assert "DecoderConfig" in repr(dc)


# ── Config-level callback install (no GPU) ───────────────────────────────


class TestCallbackInstall:
    def test_install_on_decoder_config(self):
        cfg = FlexibleDecoderConfig(
            "cam-1", gpu_id=0, pool_size=2
        ).with_decoder_config_callback(lambda dc, frame: dc)
        assert "cam-1" in repr(cfg)

    def test_install_on_pool_config(self):
        cfg = FlexibleDecoderPoolConfig(
            gpu_id=0, pool_size=2, eviction_ttl_ms=1000
        ).with_decoder_config_callback(lambda dc, frame: dc)
        assert repr(cfg)


# ── Callback invocation (GPU-only) ───────────────────────────────────────


@skip_no_ds_runtime
class TestCallbackInvocation:
    """Drive the JPEG codepath and verify the callback is invoked."""

    def setup_method(self):
        init_cuda(0)

    def test_callback_receives_config_and_frame(self):
        invocations: List[tuple] = []
        lock = threading.Lock()

        def on_cfg(dc: DecoderConfig, frame: VideoFrame) -> DecoderConfig:
            with lock:
                invocations.append((dc.codec(), frame.source_id))
            return dc

        cfg = FlexibleDecoderConfig(
            "cam-cb", gpu_id=0, pool_size=2
        ).with_decoder_config_callback(on_cfg)

        results: List[FlexibleDecoderOutput] = []

        def on_output(out: FlexibleDecoderOutput):
            results.append(out)

        dec = FlexibleDecoder(cfg, on_output)
        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-cb", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)
        _poll_until(lambda: len(invocations) >= 1, timeout=5.0)
        dec.graceful_shutdown()

        assert len(invocations) >= 1, f"callback never invoked; results: {results}"
        codec, src = invocations[0]
        assert codec == Codec.Jpeg
        assert src == "cam-cb"

    def test_callback_customizes_jpeg_backend(self):
        seen_backends: List[JpegBackend] = []

        def on_cfg(dc: DecoderConfig, frame: VideoFrame) -> DecoderConfig:
            jpeg = dc.as_jpeg()
            if jpeg is not None:
                seen_backends.append(jpeg.backend)
                return dc.with_jpeg(JpegDecoderConfig.gpu())
            return dc

        cfg = FlexibleDecoderConfig(
            "cam-jpeg", gpu_id=0, pool_size=2
        ).with_decoder_config_callback(on_cfg)
        dec = FlexibleDecoder(cfg, lambda _o: None)

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-jpeg", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)
        _poll_until(lambda: len(seen_backends) >= 1, timeout=5.0)
        dec.graceful_shutdown()

        assert len(seen_backends) >= 1

    def test_callback_exception_falls_back(self):
        calls: List[int] = []

        def on_cfg(dc: DecoderConfig, frame: VideoFrame) -> DecoderConfig:
            calls.append(1)
            raise RuntimeError("boom")

        cfg = FlexibleDecoderConfig(
            "cam-err", gpu_id=0, pool_size=2
        ).with_decoder_config_callback(on_cfg)
        results: List[FlexibleDecoderOutput] = []
        dec = FlexibleDecoder(cfg, lambda o: results.append(o))

        jpeg_data = _make_jpeg(320, 240)
        frame = _make_frame("cam-err", 320, 240, codec="jpeg", pts=0)
        dec.submit(frame, jpeg_data)
        _poll_until(lambda: len(calls) >= 1, timeout=5.0)
        dec.graceful_shutdown()

        assert len(calls) >= 1
        frame_outputs = [r for r in results if r.is_frame]
        assert len(frame_outputs) >= 1, "decode must still succeed on fallback"
