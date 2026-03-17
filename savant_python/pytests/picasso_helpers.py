"""Shared helpers for Picasso GPU pipeline tests.

Consolidates constants, frame/object builders, draw-spec construction,
encoder configuration, and source-spec helpers used by both
``test_picasso_pipeline.py`` and ``test_picasso_e2e.py``.

Matches the Rust benchmark (``savant_deepstream/picasso/benches/skia_pipeline.rs``)
and the Rust test helpers (``savant_deepstream/picasso/tests/common/mod.rs``).
"""

from __future__ import annotations

import math
from typing import Optional

from savant_rs.deepstream import (
    DsNvSurfaceBufferGenerator,
    SurfaceView,
    TransformConfig,
    VideoFormat,
    has_nvenc,
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
    CodecSpec,
    ConditionalSpec,
    DgpuPreset,
    EncoderConfig,
    EncoderProperties,
    H264DgpuProps,
    JpegProps,
    ObjectDrawSpec,
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


# ─── Constants matching the Rust benchmark ──────────────────────────────

WIDTH = 1280
HEIGHT = 720
FPS = 30
FRAME_DURATION_NS = 1_000_000_000 // FPS

CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("person", (255, 80, 80)),
    ("car", (80, 200, 255)),
    ("truck", (255, 180, 40)),
    ("bicycle", (80, 255, 120)),
    ("dog", (220, 100, 255)),
    ("bus", (255, 255, 80)),
    ("bike", (80, 255, 255)),
    ("sign", (255, 140, 140)),
]

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


# ─── Deterministic pseudo-random (matches Rust version) ────────────────


def pseudo_rand(seed1: int, seed2: int) -> float:
    """Deterministic pseudo-random in [0, 1), matching the Rust benchmark."""
    mask64 = 0xFFFF_FFFF_FFFF_FFFF
    h = ((seed1 * 6_364_136_223_846_793_005) & mask64 + seed2) & mask64
    h ^= h >> 33
    h = (h * 0xFF51_AFD7_ED55_8CCD) & mask64
    h ^= h >> 33
    return (h & 0x00FF_FFFF) / 0x0100_0000


# ─── Frame / object helpers ────────────────────────────────────────────


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


def make_nvmm_buffer(
    gen: DsNvSurfaceBufferGenerator,
    frame_id: int,
) -> SurfaceView:
    """Acquire a GPU buffer and wrap as SurfaceView (mirrors Rust test helper)."""
    buf = gen.acquire_surface(id=frame_id)
    return SurfaceView.from_buffer(buf)


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


# ─── Draw spec ─────────────────────────────────────────────────────────


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


# ─── Encoder configuration ─────────────────────────────────────────────


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


def build_default_encoder_config(
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> EncoderConfig:
    """Build encoder config: H.264 when NVENC is available, JPEG otherwise."""
    if has_nvenc(0):
        return build_h264_encoder_config(width, height, fps)
    return build_jpeg_encoder_config(width, height, fps)


# ─── Source spec ───────────────────────────────────────────────────────


def build_source_spec(
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    use_render: bool = False,
    use_gpumat: bool = False,
    encoder_config: Optional[EncoderConfig] = None,
    draw: Optional[ObjectDrawSpec] = None,
    font_family: str = "monospace",
    conditional: Optional[ConditionalSpec] = None,
    idle_timeout_secs: Optional[int] = None,
) -> SourceSpec:
    """Build an encode SourceSpec with sensible defaults."""
    enc = encoder_config or build_default_encoder_config(width, height)
    return SourceSpec(
        codec=CodecSpec.encode(TransformConfig(), enc),
        draw=draw or build_draw_spec(),
        font_family=font_family,
        use_on_render=use_render,
        use_on_gpumat=use_gpumat,
        conditional=conditional,
        idle_timeout_secs=idle_timeout_secs,
    )
