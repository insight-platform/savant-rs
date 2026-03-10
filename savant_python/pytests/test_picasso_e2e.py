"""E2E tests for Picasso GPU video pipeline.

Ports missing e2e scenarios from savant_deepstream/picasso/tests/ to Python.
Requires the deepstream feature and a working DeepStream/CUDA runtime.

Reference: savant_python/aux/picasso/kb/ (api.md, deps.md, patterns.md).
"""

from __future__ import annotations

import math
import struct
import threading
import time

import pytest

_mod = pytest.importorskip("savant_rs.picasso")
if not hasattr(_mod, "PicassoEngine"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)

_ds = pytest.importorskip("savant_rs.deepstream")
if not hasattr(_ds, "DsNvSurfaceBufferGenerator"):
    pytest.skip("deepstream feature disabled", allow_module_level=True)


def _ds_runtime_available() -> bool:
    try:
        from savant_rs.deepstream import init_cuda

        init_cuda(0)
        return True
    except Exception:
        return False


if not _ds_runtime_available():
    pytest.skip("DeepStream/CUDA runtime not available", allow_module_level=True)


from savant_rs.deepstream import (
    DsNvSurfaceBufferGenerator,
    TransformConfig,
    VideoFormat,
    init_cuda,
)
from savant_rs.draw_spec import (
    BoundingBoxDraw,
    ColorDraw,
    DotDraw,
    LabelDraw,
    LabelPosition,
    ObjectDraw,
    PaddingDraw,
)
from savant_rs.gstreamer import Codec
from savant_rs.picasso import (
    Callbacks,
    CodecSpec,
    ConditionalSpec,
    DgpuPreset,
    EncoderConfig,
    EncoderProperties,
    EvictionDecision,
    GeneralSpec,
    H264DgpuProps,
    JpegProps,
    ObjectDrawSpec,
    PicassoEngine,
    PngProps,
    SourceSpec,
    TuningPreset,
)
from savant_rs.primitives import (
    IdCollisionResolutionPolicy,
    VideoFrame,
    VideoFrameContent,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox

# ─── Constants ─────────────────────────────────────────────────────────────

WIDTH = 1280
HEIGHT = 720
FPS = 30
FRAME_DURATION_NS = 1_000_000_000 // FPS
PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

CLASSES = [
    ("person", (255, 80, 80)),
    ("car", (80, 200, 255)),
    ("truck", (255, 180, 40)),
    ("bicycle", (80, 255, 120)),
    ("dog", (220, 100, 255)),
    ("bus", (255, 255, 80)),
    ("bike", (80, 255, 255)),
    ("sign", (255, 140, 140)),
]


# ─── Helpers ───────────────────────────────────────────────────────────────


def pseudo_rand(seed1: int, seed2: int) -> float:
    """Deterministic pseudo-random in [0, 1)."""
    mask64 = 0xFFFF_FFFF_FFFF_FFFF
    h = ((seed1 * 6_364_136_223_846_793_005) & mask64 + seed2) & mask64
    h ^= h >> 33
    h = (h * 0xFF51_AFD7_ED55_8CCD) & mask64
    h ^= h >> 33
    return (h & 0x00FF_FFFF) / 0x0100_0000


def make_frame(
    source_id: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> VideoFrame:
    """Create a VideoFrame with default params."""
    return VideoFrame(
        source_id=source_id,
        framerate="30/1",
        width=width,
        height=height,
        content=VideoFrameContent.none(),
        time_base=(1, 1_000_000_000),
        pts=0,
    )


def make_nvmm_buffer(gen: DsNvSurfaceBufferGenerator, frame_id: int) -> int:
    """Acquire a GPU buffer."""
    return gen.acquire_surface(id=frame_id)


def add_objects_to_frame(
    frame: VideoFrame,
    frame_idx: int,
    num_boxes: int = 20,
) -> None:
    """Add detection objects with deterministic positions."""
    w, h = frame.width, frame.height
    scene_w = w - min(w * 0.22, 340.0)
    t = frame_idx / 60.0

    for i in range(num_boxes):
        seed = i
        cx_base = pseudo_rand(seed, 100) * scene_w * 0.7 + scene_w * 0.15
        cy_base = pseudo_rand(seed, 200) * h * 0.7 + h * 0.15
        orbit_rx = pseudo_rand(seed, 300) * scene_w * 0.12 + 20.0
        orbit_ry = pseudo_rand(seed, 400) * h * 0.10 + 15.0
        speed = 0.3 + pseudo_rand(seed, 500) * 0.7
        phase = pseudo_rand(seed, 600) * math.tau

        cx = cx_base + math.cos(t * speed + phase) * orbit_rx
        cy = cy_base + math.sin(t * speed * 0.8 + phase) * orbit_ry
        bw = 50.0 + pseudo_rand(seed, 700) * 140.0
        bh = 40.0 + pseudo_rand(seed, 800) * 160.0
        class_idx = int(pseudo_rand(seed, 900) * len(CLASSES)) % len(CLASSES)
        cls_name = CLASSES[class_idx][0]

        obj = VideoObject(
            id=0,
            namespace="detector",
            label=cls_name,
            detection_box=RBBox(cx, cy, bw, bh),
            attributes=[],
            confidence=None,
            track_id=None,
            track_box=None,
        )
        frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)


def build_draw_spec() -> ObjectDrawSpec:
    """Build ObjectDrawSpec with bbox + label + dot per class."""
    spec = ObjectDrawSpec()
    for cls_name, (r, g, b) in CLASSES:
        border = ColorDraw(r, g, b, 255)
        bg = ColorDraw(r, g, b, 50)
        bb = BoundingBoxDraw(border, bg, 2, PaddingDraw.default_padding())
        dot = DotDraw(ColorDraw(r, g, b, 255), 4)
        label = LabelDraw(
            font_color=ColorDraw(0, 0, 0, 255),
            background_color=ColorDraw(r, g, b, 200),
            border_color=ColorDraw(0, 0, 0, 0),
            font_scale=1.4,
            thickness=1,
            position=LabelPosition.default_position(),
            padding=PaddingDraw(4, 2, 4, 2),
            format=["{label} #{id}", "{confidence}"],
        )
        od = ObjectDraw(bounding_box=bb, central_dot=dot, label=label)
        spec.insert("detector", cls_name, od)
    return spec


def build_h264_encoder_config(
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> EncoderConfig:
    """Build H.264 dGPU encoder config."""
    props = EncoderProperties.h264_dgpu(
        H264DgpuProps(
            bitrate=4_000_000,
            preset=DgpuPreset.P1,
            tuning_info=TuningPreset.LOW_LATENCY,
            iframeinterval=30,
        )
    )
    cfg = EncoderConfig(Codec.H264, width, height)
    cfg.format(VideoFormat.RGBA)
    cfg.fps(fps, 1)
    cfg.properties(props)
    return cfg


def build_jpeg_encoder_config(
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> EncoderConfig:
    """Build JPEG encoder config."""
    props = EncoderProperties.jpeg(JpegProps(quality=90))
    cfg = EncoderConfig(Codec.JPEG, width, height)
    cfg.format(VideoFormat.RGBA)
    cfg.fps(fps, 1)
    cfg.properties(props)
    return cfg


def build_png_encoder_config(
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> EncoderConfig:
    """Build PNG encoder config (CPU-based, lossless)."""
    props = EncoderProperties.png(PngProps(compression_level=6))
    cfg = EncoderConfig(Codec.PNG, width, height)
    cfg.format(VideoFormat.RGBA)
    cfg.fps(fps, 1)
    cfg.properties(props)
    return cfg


def build_source_spec(
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    use_render: bool = False,
    use_gpumat: bool = False,
    encoder_config: EncoderConfig | None = None,
    draw: ObjectDrawSpec | None = None,
    font_family: str = "monospace",
    conditional: ConditionalSpec | None = None,
    idle_timeout_secs: int | None = None,
) -> SourceSpec:
    """Build encode SourceSpec."""
    enc = encoder_config or build_h264_encoder_config(width, height)
    return SourceSpec(
        codec=CodecSpec.encode(TransformConfig(), enc),
        draw=draw or build_draw_spec(),
        font_family=font_family,
        use_on_render=use_render,
        use_on_gpumat=use_gpumat,
        conditional=conditional,
        idle_timeout_secs=idle_timeout_secs,
    )


# ─── TestPngEncode ────────────────────────────────────────────────────────


class TestPngEncode:
    """PNG encoding with validation (port of test_e2e_png_encode.rs)."""

    def test_png_encode_produces_valid_image(self) -> None:
        """Encode frames to PNG and validate signature + dimensions."""
        init_cuda(0)

        png_bytes: list[bytes] = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            if output.is_video_frame:
                vf = output.as_video_frame()
                if vf.content.is_internal():
                    with lock:
                        png_bytes.append(vf.content.get_data())

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        dst_w, dst_h = 640, 480
        enc = build_png_encoder_config(dst_w, dst_h)
        spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=ObjectDrawSpec(),
            font_family="sans-serif",
        )
        engine.set_source_spec("png", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, 320, 240, FPS, 1, 0)

        frame = make_frame("png", 320, 240)
        frame.pts = 0
        frame.duration = FRAME_DURATION_NS
        obj = VideoObject(
            id=0,
            namespace="det",
            label="person",
            detection_box=RBBox(160.0, 120.0, 80.0, 120.0),
            attributes=[],
            confidence=0.95,
            track_id=None,
            track_box=None,
        )
        frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)
        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("png", frame, buf_ptr)
        engine.send_eos("png")

        time.sleep(3)
        engine.shutdown()

        assert len(png_bytes) >= 1, "expected at least one PNG frame"
        data = png_bytes[0]

        assert len(data) >= len(PNG_SIGNATURE), "PNG output too short"
        assert data[: len(PNG_SIGNATURE)] == PNG_SIGNATURE, "Invalid PNG signature"

        # Read dimensions from IHDR chunk (bytes 16-24: width, height big-endian)
        assert len(data) >= 24, "PNG too short for IHDR"
        width, height = struct.unpack(">II", data[16:24])
        assert width == dst_w, f"PNG width mismatch: expected {dst_w}, got {width}"
        assert height == dst_h, f"PNG height mismatch: expected {dst_h}, got {height}"


# ─── TestAsyncDrain ───────────────────────────────────────────────────────


class TestAsyncDrain:
    """Async drain behavior (port of test_e2e_async_drain.rs)."""

    def test_async_drain_delivers_independently(self) -> None:
        """Drain delivers encoded frames without additional send_frame calls."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        enc = build_jpeg_encoder_config()
        spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
        num_frames = 10

        for i in range(num_frames):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) == num_frames, (
            f"expected {num_frames} frames, got {len(video_frames)}"
        )

    def test_draw_spec_hot_swap_preserves_drain(self) -> None:
        """Draw spec hot-swap (same codec) preserves encoder and drain."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        enc = build_jpeg_encoder_config()
        spec1 = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec1)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(4):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        # Hot-swap draw spec only (same codec)
        spec2 = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=ObjectDrawSpec(),
        )
        engine.set_source_spec("src-0", spec2)

        for i in range(4, 8):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        eos_count = sum(1 for o in results if o.is_eos)
        assert len(video_frames) == 8, f"expected 8 frames, got {len(video_frames)}"
        assert eos_count == 1, f"expected 1 EOS, got {eos_count}"

    def test_sustained_throughput_no_frame_loss(self) -> None:
        """Rapid submission (100 frames) without frame loss."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        enc = build_jpeg_encoder_config()
        spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
        num_frames = 100

        for i in range(num_frames):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(15)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) == num_frames, (
            f"expected {num_frames} frames, got {len(video_frames)}"
        )

    def test_eos_flushes_all_in_flight(self) -> None:
        """EOS flushes all in-flight frames."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        enc = build_jpeg_encoder_config()
        spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
        num_frames = 20

        for i in range(num_frames):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        eos_list = [o for o in results if o.is_eos]
        assert len(video_frames) == num_frames
        assert len(eos_list) >= 1
        # EOS should come after all video frames
        last_vf_idx = max(i for i, o in enumerate(results) if o.is_video_frame)
        first_eos_idx = min(i for i, o in enumerate(results) if o.is_eos)
        assert first_eos_idx > last_vf_idx, "EOS should follow all video frames"


# ─── TestHotSwapEncodeParams ──────────────────────────────────────────────


class TestHotSwapEncodeParams:
    """Hot-swap encode params (port of test_e2e_hot_swap_encode_params.rs)."""

    def test_hot_swap_resolution(self) -> None:
        """Resolution change mid-stream: 1280x720 -> 640x480."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        enc1 = build_jpeg_encoder_config(1280, 720)
        spec1 = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc1),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec1)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, 1280, 720, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0", 1280, 720)
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        enc2 = build_jpeg_encoder_config(640, 480)
        spec2 = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc2),
            draw=build_draw_spec(),
        )
        engine.set_source_spec("src-0", spec2)

        gen2 = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, 640, 480, FPS, 1, 0)
        for i in range(5, 10):
            frame = make_frame("src-0", 640, 480)
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen2, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        resolutions = {
            (vf.width, vf.height) for vf in (o.as_video_frame() for o in video_frames)
        }
        assert (1280, 720) in resolutions
        assert (640, 480) in resolutions
        eos_count = sum(1 for o in results if o.is_eos)
        assert eos_count == 1


# ─── TestConditionalSelectiveRecording ────────────────────────────────────


class TestConditionalSelectiveRecording:
    """Conditional encode/render based on frame attributes."""

    def test_conditional_encode_and_render(self) -> None:
        """Only frames with required attributes are encoded/rendered."""
        init_cuda(0)

        from savant_rs.primitives import AttributeValue

        encoded_frames: list = []
        render_calls: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            if output.is_video_frame:
                with lock:
                    encoded_frames.append(output.as_video_frame())

        def on_render(source_id: str, fbo_id: int, w: int, h: int, frame) -> None:
            with lock:
                render_calls.append((source_id, fbo_id))

        callbacks = Callbacks(
            on_encoded_frame=on_encoded,
            on_render=on_render,
        )
        conditional = ConditionalSpec(
            encode_attribute=("recording", "active"),
            render_attribute=("scene", "has_objects"),
        )
        enc = build_h264_encoder_config()
        spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc),
            draw=build_draw_spec(),
            conditional=conditional,
            use_on_render=True,
        )
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        # Frame 0: no attributes -> dropped
        frame0 = make_frame("src-0")
        frame0.pts = 0
        frame0.duration = FRAME_DURATION_NS
        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("src-0", frame0, buf_ptr)

        # Frame 1: recording.active -> encoded, no objects -> no render
        frame1 = make_frame("src-0")
        frame1.pts = FRAME_DURATION_NS
        frame1.duration = FRAME_DURATION_NS
        frame1.set_persistent_attribute(
            "recording", "active", False, "hint", [AttributeValue.boolean(True)]
        )
        buf_ptr = make_nvmm_buffer(gen, 1)
        engine.send_frame("src-0", frame1, buf_ptr)

        # Frame 2: recording + scene.has_objects -> encoded and rendered
        frame2 = make_frame("src-0")
        frame2.pts = 2 * FRAME_DURATION_NS
        frame2.duration = FRAME_DURATION_NS
        frame2.set_persistent_attribute(
            "recording", "active", False, "hint", [AttributeValue.boolean(True)]
        )
        frame2.set_persistent_attribute(
            "scene", "has_objects", False, "hint", [AttributeValue.boolean(True)]
        )
        add_objects_to_frame(frame2, 2)
        buf_ptr = make_nvmm_buffer(gen, 2)
        engine.send_frame("src-0", frame2, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(4)
        engine.shutdown()

        assert len(encoded_frames) >= 2, "expected at least 2 encoded frames"
        assert len(render_calls) >= 1, "expected at least 1 render call"


# ─── TestMixedCodecs ──────────────────────────────────────────────────────


class TestMixedCodecs:
    """Mixed Drop/Bypass/Encode sources in one engine."""

    def test_mixed_drop_bypass_encode(self) -> None:
        """Three sources with different codec specs."""
        init_cuda(0)

        bypass_results: list = []
        encoded_results: list = []
        lock = threading.Lock()

        def on_bypass(output) -> None:
            with lock:
                bypass_results.append(output)

        def on_encoded(output) -> None:
            with lock:
                encoded_results.append(output)

        callbacks = Callbacks(
            on_bypass_frame=on_bypass,
            on_encoded_frame=on_encoded,
        )
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        engine.set_source_spec("drop-src", SourceSpec(codec=CodecSpec.drop_frames()))
        engine.set_source_spec("bypass-src", SourceSpec(codec=CodecSpec.bypass()))
        engine.set_source_spec(
            "encode-src",
            build_source_spec(encoder_config=build_h264_encoder_config()),
        )

        gen_bypass = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
        gen_encode = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame_b = make_frame("bypass-src")
            frame_b.pts = i * FRAME_DURATION_NS
            frame_b.duration = FRAME_DURATION_NS
            buf_ptr = make_nvmm_buffer(gen_bypass, i)
            engine.send_frame("bypass-src", frame_b, buf_ptr)

        for i in range(5):
            frame_d = make_frame("drop-src")
            frame_d.pts = i * FRAME_DURATION_NS
            frame_d.duration = FRAME_DURATION_NS
            buf_ptr = make_nvmm_buffer(gen_bypass, i + 10)
            engine.send_frame("drop-src", frame_d, buf_ptr)

        for i in range(5):
            frame_e = make_frame("encode-src")
            frame_e.pts = i * FRAME_DURATION_NS
            frame_e.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame_e, i)
            buf_ptr = make_nvmm_buffer(gen_encode, i)
            engine.send_frame("encode-src", frame_e, buf_ptr)

        engine.send_eos("bypass-src")
        engine.send_eos("drop-src")
        engine.send_eos("encode-src")

        time.sleep(4)
        engine.shutdown()

        assert len(bypass_results) >= 5
        assert len([o for o in encoded_results if o.is_video_frame]) >= 3
        eos_count = sum(1 for o in encoded_results if o.is_eos)
        assert eos_count >= 1


# ─── TestEncodeEosReencode ────────────────────────────────────────────────


class TestEncodeEosReencode:
    """Encoder re-creation after EOS (camera reconnect)."""

    def test_reencode_after_eos(self) -> None:
        """Frames -> EOS -> new frames; encoder is re-created."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec())

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(2)

        for i in range(5, 10):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(4)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        eos_count = sum(1 for o in results if o.is_eos)
        assert len(video_frames) >= 10
        assert eos_count == 2


# ─── TestOnGpuMat ────────────────────────────────────────────────────────


class TestOnGpuMat:
    """OnGpuMat callback (port of test_e2e_on_gpumat.rs)."""

    def test_on_gpumat_fires_when_enabled(self) -> None:
        """OnGpuMat fires with non-zero data_ptr when use_on_gpumat=True."""
        init_cuda(0)

        gpumat_calls: list = []
        lock = threading.Lock()

        def on_gpumat(
            source_id: str,
            frame,
            data_ptr: int,
            pitch: int,
            width: int,
            height: int,
            cuda_stream: int,
        ) -> None:
            with lock:
                gpumat_calls.append((source_id, data_ptr, pitch, width, height, cuda_stream))

        callbacks = Callbacks(on_gpumat=on_gpumat)
        spec = build_source_spec(use_gpumat=True)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(4)
        engine.shutdown()

        assert len(gpumat_calls) >= 3
        for source_id, data_ptr, pitch, w, h, cuda_stream in gpumat_calls:
            assert source_id == "src-0"
            assert data_ptr != 0
            assert w == WIDTH
            assert h == HEIGHT
            assert cuda_stream != 0, "cuda_stream should be non-zero"

    def test_on_gpumat_does_not_fire_when_disabled(self) -> None:
        """OnGpuMat does not fire when use_on_gpumat=False."""
        init_cuda(0)

        gpumat_calls: list = []
        lock = threading.Lock()

        def on_gpumat(
            source_id: str,
            frame,
            data_ptr: int,
            pitch: int,
            width: int,
            height: int,
            cuda_stream: int,
        ) -> None:
            with lock:
                gpumat_calls.append((source_id, data_ptr))

        callbacks = Callbacks(on_gpumat=on_gpumat)
        spec = build_source_spec(use_gpumat=False)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(3):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(3)
        engine.shutdown()

        assert len(gpumat_calls) == 0


# ─── TestOnObjectDrawSpec ─────────────────────────────────────────────────


class TestOnObjectDrawSpec:
    """Dynamic per-object draw override (port of test_e2e_on_object_draw_spec.rs)."""

    def test_dynamic_per_object_draw_override(self) -> None:
        """on_object_draw_spec callback fires per object."""
        init_cuda(0)

        draw_spec_calls: list = []
        encoded_results: list = []
        lock = threading.Lock()

        def on_object_draw_spec(source_id: str, obj, current_spec) -> None:
            with lock:
                draw_spec_calls.append((source_id, obj.label))
            return current_spec

        def on_encoded(output) -> None:
            if output.is_video_frame:
                with lock:
                    encoded_results.append(output)

        callbacks = Callbacks(
            on_object_draw_spec=on_object_draw_spec,
            on_encoded_frame=on_encoded,
        )
        spec = build_source_spec(draw=build_draw_spec())
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", spec)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        frame = make_frame("src-0")
        frame.pts = 0
        frame.duration = FRAME_DURATION_NS
        for label in ["person", "car", "truck"]:
            obj = VideoObject(
                id=0,
                namespace="detector",
                label=label,
                detection_box=RBBox(100.0, 100.0, 50.0, 50.0),
                attributes=[],
                confidence=None,
                track_id=None,
                track_box=None,
            )
            frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)

        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("src-0", frame, buf_ptr)
        engine.send_eos("src-0")

        time.sleep(3)
        engine.shutdown()

        assert len(draw_spec_calls) >= 3
        assert len(encoded_results) >= 1


# ─── TestEvictionKeepFor ──────────────────────────────────────────────────


class TestEvictionKeepFor:
    """Idle eviction with KeepFor then Terminate."""

    def test_eviction_keep_for_then_terminate(self) -> None:
        """Eviction callback returns KeepFor first, then Terminate."""
        init_cuda(0)

        eviction_calls: list = []
        eos_received: list = []
        lock = threading.Lock()

        def on_eviction(source_id: str):
            with lock:
                eviction_calls.append(source_id)
                if len(eviction_calls) == 1:
                    return EvictionDecision.keep_for(1)
                return EvictionDecision.terminate()

        def on_bypass(output) -> None:
            if output.is_eos:
                with lock:
                    eos_received.append(output.as_eos())

        callbacks = Callbacks(
            on_eviction=on_eviction,
            on_bypass_frame=on_bypass,
        )
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=1), callbacks)
        engine.set_source_spec("src-0", SourceSpec(codec=CodecSpec.bypass()))

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
        frame = make_frame("src-0")
        frame.pts = 0
        frame.duration = FRAME_DURATION_NS
        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("src-0", frame, buf_ptr)
        engine.send_eos("src-0")

        time.sleep(5)
        engine.shutdown()

        assert len(eviction_calls) == 2
        assert len(eos_received) >= 1


# ─── TestWatchdog ────────────────────────────────────────────────────────


class TestWatchdog:
    """Watchdog reaps idle worker; source can be re-created."""

    def test_watchdog_reaps_idle_worker(self) -> None:
        """After idle timeout, set_source_spec succeeds again."""
        init_cuda(0)

        bypass_results: list = []
        lock = threading.Lock()

        def on_bypass(output) -> None:
            with lock:
                bypass_results.append(output)

        callbacks = Callbacks(on_bypass_frame=on_bypass)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=1), callbacks)
        engine.set_source_spec("src-0", SourceSpec(codec=CodecSpec.bypass()))

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        frame1 = make_frame("src-0")
        frame1.pts = 0
        frame1.duration = FRAME_DURATION_NS
        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("src-0", frame1, buf_ptr)
        engine.send_eos("src-0")

        time.sleep(3)

        engine.set_source_spec("src-0", SourceSpec(codec=CodecSpec.bypass()))
        frame2 = make_frame("src-0")
        frame2.pts = 0
        frame2.duration = FRAME_DURATION_NS
        buf_ptr = make_nvmm_buffer(gen, 1)
        engine.send_frame("src-0", frame2, buf_ptr)
        engine.send_eos("src-0")

        time.sleep(3)
        engine.shutdown()

        assert len(bypass_results) >= 2


# ─── TestFontFamilyHotSwap ───────────────────────────────────────────────


class TestFontFamilyHotSwap:
    """Font family change triggers DrawContext rebuild."""

    def test_font_family_change(self) -> None:
        """Swap font_family mid-stream; all frames encoded."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        spec1 = build_source_spec(font_family="sans-serif")
        engine.set_source_spec("src-0", spec1)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(4):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        spec2 = build_source_spec(font_family="monospace")
        engine.set_source_spec("src-0", spec2)

        for i in range(4, 8):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(5)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) >= 8


# ─── TestSustainedMultiSourceEos ──────────────────────────────────────────


class TestSustainedMultiSourceEos:
    """Multi-source with staggered EOS."""

    def test_staggered_eos_multi_source(self) -> None:
        """4 sources with staggered EOS; correct frame counts."""
        init_cuda(0)

        source_results: dict = {}
        lock = threading.Lock()

        def on_bypass(output) -> None:
            if not output.is_video_frame:
                return
            frame = output.as_video_frame()
            with lock:
                sid = frame.source_id
                if sid not in source_results:
                    source_results[sid] = []
                source_results[sid].append(frame)

        callbacks = Callbacks(on_bypass_frame=on_bypass)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)

        sources = ["s1", "s2", "s3", "s4"]
        for sid in sources:
            engine.set_source_spec(sid, SourceSpec(codec=CodecSpec.bypass()))

        gens = {
            sid: DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)
            for sid in sources
        }

        for i in range(10):
            for sid in sources:
                frame = make_frame(sid)
                frame.pts = i * FRAME_DURATION_NS
                frame.duration = FRAME_DURATION_NS
                buf_ptr = make_nvmm_buffer(gens[sid], i)
                engine.send_frame(sid, frame, buf_ptr)

        engine.send_eos("s1")
        time.sleep(0.5)
        for i in range(10, 15):
            for sid in ["s2", "s3", "s4"]:
                frame = make_frame(sid)
                frame.pts = i * FRAME_DURATION_NS
                frame.duration = FRAME_DURATION_NS
                buf_ptr = make_nvmm_buffer(gens[sid], i)
                engine.send_frame(sid, frame, buf_ptr)

        engine.send_eos("s2")
        engine.send_eos("s3")
        time.sleep(0.5)
        for i in range(15, 20):
            for sid in ["s4"]:
                frame = make_frame(sid)
                frame.pts = i * FRAME_DURATION_NS
                frame.duration = FRAME_DURATION_NS
                buf_ptr = make_nvmm_buffer(gens[sid], i)
                engine.send_frame(sid, frame, buf_ptr)

        engine.send_eos("s4")
        time.sleep(5)
        engine.shutdown()

        assert len(source_results.get("s1", [])) >= 10
        assert len(source_results.get("s2", [])) >= 15
        assert len(source_results.get("s3", [])) >= 15
        assert len(source_results.get("s4", [])) >= 20


# ─── TestHighObjectCount ──────────────────────────────────────────────────


class TestHighObjectCount:
    """Stress test with many objects per frame."""

    def test_200_objects_per_frame(self) -> None:
        """200 objects per frame; pipeline completes."""
        init_cuda(0)

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            with lock:
                results.append(output)

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec())

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        for i in range(5):
            frame = make_frame("src-0")
            frame.pts = i * FRAME_DURATION_NS
            frame.duration = FRAME_DURATION_NS
            add_objects_to_frame(frame, i, num_boxes=200)
            buf_ptr = make_nvmm_buffer(gen, i)
            engine.send_frame("src-0", frame, buf_ptr)

        engine.send_eos("src-0")
        time.sleep(10)
        engine.shutdown()

        video_frames = [o for o in results if o.is_video_frame]
        assert len(video_frames) >= 1


# ─── TestPerSourceIdleTimeout ──────────────────────────────────────────────


class TestPerSourceIdleTimeout:
    """Per-source idle timeout override."""

    def test_per_source_idle_timeout(self) -> None:
        """Different idle_timeout_secs per source."""
        init_cuda(0)

        eviction_sources: list = []
        lock = threading.Lock()

        def on_eviction(source_id: str):
            with lock:
                eviction_sources.append(source_id)
            return EvictionDecision.terminate()

        callbacks = Callbacks(on_eviction=on_eviction)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=60), callbacks)

        spec_fast = SourceSpec(
            codec=CodecSpec.bypass(),
            idle_timeout_secs=1,
        )
        spec_slow = SourceSpec(
            codec=CodecSpec.bypass(),
            idle_timeout_secs=60,
        )
        engine.set_source_spec("fast", spec_fast)
        engine.set_source_spec("slow", spec_slow)

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        frame_fast = make_frame("fast")
        frame_fast.pts = 0
        frame_fast.duration = FRAME_DURATION_NS
        engine.send_frame("fast", frame_fast, make_nvmm_buffer(gen, 0))
        engine.send_eos("fast")

        frame_slow = make_frame("slow")
        frame_slow.pts = 0
        frame_slow.duration = FRAME_DURATION_NS
        engine.send_frame("slow", frame_slow, make_nvmm_buffer(gen, 1))
        engine.send_eos("slow")

        time.sleep(5)
        engine.shutdown()

        assert "fast" in eviction_sources


# ─── TestFrameMetadataPreservation ────────────────────────────────────────


class TestFrameMetadataPreservation:
    """Objects, bboxes, and attributes survive encode pipeline."""

    def test_metadata_bboxes_and_attributes_preserved(self) -> None:
        """Verify bboxes and attributes survive encode."""
        init_cuda(0)

        from savant_rs.primitives import AttributeValue

        results: list = []
        lock = threading.Lock()

        def on_encoded(output) -> None:
            if output.is_video_frame:
                with lock:
                    results.append(output.as_video_frame())

        callbacks = Callbacks(on_encoded_frame=on_encoded)
        engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        engine.set_source_spec("src-0", build_source_spec())

        gen = DsNvSurfaceBufferGenerator(VideoFormat.RGBA, WIDTH, HEIGHT, FPS, 1, 0)

        frame = make_frame("src-0")
        frame.pts = 0
        frame.duration = FRAME_DURATION_NS
        obj = VideoObject(
            id=0,
            namespace="detector",
            label="person",
            detection_box=RBBox(100.0, 200.0, 80.0, 120.0),
            attributes=[],
            confidence=0.95,
            track_id=None,
            track_box=None,
        )
        obj.set_persistent_attribute(
            "custom", "score", False, "hint", [AttributeValue.float(0.87)]
        )
        frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)

        buf_ptr = make_nvmm_buffer(gen, 0)
        engine.send_frame("src-0", frame, buf_ptr)
        engine.send_eos("src-0")

        time.sleep(3)
        engine.shutdown()

        assert len(results) >= 1
        vf = results[0]
        objects = vf.get_all_objects()
        assert len(objects) >= 1
        o = objects[0]
        assert o.detection_box is not None
        assert abs(o.detection_box.xc - 100.0) < 0.01
        assert abs(o.detection_box.yc - 200.0) < 0.01
        assert abs(o.detection_box.width - 80.0) < 0.01
        assert abs(o.detection_box.height - 120.0) < 0.01
