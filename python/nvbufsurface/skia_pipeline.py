#!/usr/bin/env python3
"""Skia GPU-rendered encoding pipeline using the Picasso engine with draw spec.

Demonstrates Picasso's ``ObjectDrawSpec`` for automatic bounding-box
rendering combined with a custom ``on_render`` callback for the legend.

Detection objects are attached as :class:`VideoObject` instances to each
:class:`VideoFrame`.  Picasso renders their bounding boxes via the draw
spec — no manual Skia bbox drawing needed.

The ``on_render`` callback draws the legend (detection list) on top of
the scene after Picasso has finished its own rendering.

The input NvBufSurface is pre-filled with a dark background via OpenCV
CUDA (``nvgstbuf_as_gpu_mat``); Picasso loads it into its Skia FBO, draws the
bboxes from the draw spec, then calls ``on_render`` for the legend.

Output pipeline (when ``--output`` is given)::

    appsrc (bitstream) -> h26Xparse -> qtmux -> filesink

Usage::

    # 300 frames to MP4
    python skia_pipeline.py --num-frames 300 --output /tmp/skia_demo.mp4

    # Infinite run (Ctrl-C to stop)
    python skia_pipeline.py

    # Custom resolution and codec with 8 Mbps bitrate
    python skia_pipeline.py --width 1280 --height 720 --codec h264 --bitrate 8000000
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import skia

from savant_rs.deepstream import SkiaCanvas, VideoFormat, nvgstbuf_as_gpu_mat  # noqa: E402
from savant_rs.draw_spec import (  # noqa: E402
    BoundingBoxDraw,
    ColorDraw,
    DotDraw,
    LabelDraw,
    LabelPosition,
    ObjectDraw,
    PaddingDraw,
)
from savant_rs.picasso import ObjectDrawSpec  # noqa: E402
from savant_rs.primitives import (  # noqa: E402
    IdCollisionResolutionPolicy,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox  # noqa: E402

from common import PicassoSession, add_common_args

NUM_BOXES = 20
NAMESPACE = "detector"


# ===========================================================================
# Detection classes — used for draw spec colours and object generation
# ===========================================================================


@dataclass
class DetectionClass:
    name: str
    r: int
    g: int
    b: int


CLASSES = [
    DetectionClass("person", 255, 80, 80),
    DetectionClass("car", 80, 200, 255),
    DetectionClass("truck", 255, 180, 40),
    DetectionClass("bicycle", 80, 255, 120),
    DetectionClass("dog", 220, 100, 255),
    DetectionClass("bus", 255, 255, 80),
    DetectionClass("bike", 80, 255, 255),
    DetectionClass("sign", 255, 140, 140),
]


# ===========================================================================
# Pseudo-random (deterministic animation)
# ===========================================================================


def pseudo_rand(seed1: int, seed2: int) -> float:
    MASK64 = 0xFFFF_FFFF_FFFF_FFFF
    h = ((seed1 * 6364136223846793005) + seed2) & MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    return (h & 0x00FF_FFFF) / 0x0100_0000


# ===========================================================================
# Build Picasso ObjectDrawSpec — one entry per detection class
# ===========================================================================


def build_draw_spec() -> ObjectDrawSpec:
    """Create an :class:`ObjectDrawSpec` with per-class bounding boxes only."""
    spec = ObjectDrawSpec()
    for cls in CLASSES:
        border = ColorDraw(cls.r, cls.g, cls.b, 255)
        bg = ColorDraw(cls.r, cls.g, cls.b, 50)
        bb = BoundingBoxDraw(border, bg, 2, PaddingDraw.default_padding())
        label = LabelDraw(
            font_color=ColorDraw(0, 0, 0, 255),
            background_color=ColorDraw(cls.r, cls.g, cls.b, 200),
            border_color=ColorDraw(0, 0, 0, 0),
            font_scale=1.4,
            thickness=1,
            position=LabelPosition.default_position(),
            padding=PaddingDraw(4, 2, 4, 2),
            format=["{label} #{id}", "{confidence}"],
        )
        dot = DotDraw(ColorDraw(cls.r, cls.g, cls.b, 255), 4)
        od = ObjectDraw(bounding_box=bb, label=label, central_dot=dot)
        spec.insert(NAMESPACE, cls.name, od)
    return spec


# ===========================================================================
# Add detection objects to a VideoFrame
# ===========================================================================


def add_objects(
    frame: object,
    scene_w: float,
    height: float,
    frame_idx: int,
) -> None:
    """Generate pseudo-random detections and attach them to *frame*."""
    t = frame_idx / 60.0
    for i in range(NUM_BOXES):
        seed = i
        cx_base = pseudo_rand(seed, 100) * scene_w * 0.7 + scene_w * 0.15
        cy_base = pseudo_rand(seed, 200) * height * 0.7 + height * 0.15
        orbit_rx = pseudo_rand(seed, 300) * scene_w * 0.12 + 20.0
        orbit_ry = pseudo_rand(seed, 400) * height * 0.10 + 15.0
        speed = 0.3 + pseudo_rand(seed, 500) * 0.7
        phase = pseudo_rand(seed, 600) * math.tau

        cx = cx_base + math.cos(t * speed + phase) * orbit_rx
        cy = cy_base + math.sin(t * speed * 0.8 + phase) * orbit_ry

        bw = 50.0 + pseudo_rand(seed, 700) * 140.0
        bh = 40.0 + pseudo_rand(seed, 800) * 160.0
        class_idx = int(pseudo_rand(seed, 900) * len(CLASSES)) % len(CLASSES)
        confidence = 0.55 + pseudo_rand(seed, 1000) * 0.44

        bx = max(0.0, min(cx - bw / 2.0, scene_w - bw))
        by = max(0.0, min(cy - bh / 2.0, height - bh))

        obj = VideoObject(
            id=0,
            namespace=NAMESPACE,
            label=CLASSES[class_idx].name,
            detection_box=RBBox(bx + bw / 2.0, by + bh / 2.0, bw, bh),
            attributes=[],
            confidence=confidence,
            track_id=None,
            track_box=None,
        )
        frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)


# ===========================================================================
# Legend drawing (on_render overlay)
# ===========================================================================


class LegendCtx:
    """Cached fonts and paints for the legend overlay."""

    def __init__(self):
        bold_tf = skia.Typeface("monospace", skia.FontStyle.Bold())
        normal_tf = skia.Typeface("monospace", skia.FontStyle.Normal())
        self.title_font = skia.Font(bold_tf, 18)
        self.legend_font = skia.Font(normal_tf, 13)

        self.white_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        self.sidebar_bg_paint = skia.Paint(Color=skia.Color(15, 18, 25, 210))
        self.separator_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 100),
            StrokeWidth=1.0,
            Style=skia.Paint.kStroke_Style,
        )
        self.divider_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 60),
            StrokeWidth=1.0,
            Style=skia.Paint.kStroke_Style,
        )
        self.dot_paint = skia.Paint(AntiAlias=True)
        self.legend_text_paint = skia.Paint(
            AntiAlias=True, Color=skia.Color(255, 255, 255, 220)
        )

    _cls_color_cache: dict[str, tuple[int, int, int]] = {}

    def class_color(self, label: str) -> tuple[int, int, int]:
        if label not in self._cls_color_cache:
            for cls in CLASSES:
                self._cls_color_cache[cls.name] = (cls.r, cls.g, cls.b)
        return self._cls_color_cache.get(label, (200, 200, 200))


def draw_legend(
    canvas: skia.Canvas,
    ctx: LegendCtx,
    frame: object,
    width: float,
    height: float,
) -> None:
    """Draw the legend (detection list) on *canvas*."""
    sidebar_w = min(340.0, width * 0.22)
    sx = width - sidebar_w

    canvas.drawRect(skia.Rect.MakeXYWH(sx, 0, sidebar_w, height), ctx.sidebar_bg_paint)
    canvas.drawLine(sx, 0, sx, height, ctx.separator_paint)
    canvas.drawString("DETECTIONS", sx + 12, 28, ctx.title_font, ctx.white_paint)
    canvas.drawLine(sx + 8, 36, sx + sidebar_w - 8, 36, ctx.divider_paint)

    y_off = 52.0
    row_h = 18.0
    objects = frame.get_all_objects()
    num_objects = len(objects)

    for i, obj in enumerate(objects):
        if y_off + row_h > height - 20.0:
            ctx.legend_text_paint.setColor(skia.Color(255, 255, 255, 180))
            canvas.drawString(
                f"... +{num_objects - i} more",
                sx + 12,
                y_off + 14,
                ctx.legend_font,
                ctx.legend_text_paint,
            )
            break

        r, g, b = ctx.class_color(obj.label)
        ctx.dot_paint.setColor(skia.Color(r, g, b, 255))
        canvas.drawCircle(sx + 16, y_off + 6, 4.0, ctx.dot_paint)

        box = obj.detection_box
        conf = obj.confidence if obj.confidence is not None else 0.0
        entry = (
            f"{obj.label:<8} #{obj.id:<2} "
            f"({int(box.xc):>4},{int(box.yc):>4}) {conf * 100:>3.0f}%"
        )
        ctx.legend_text_paint.setColor(skia.Color(255, 255, 255, 220))
        canvas.drawString(
            entry, sx + 26, y_off + 10, ctx.legend_font, ctx.legend_text_paint
        )
        y_off += row_h


# ===========================================================================
# on_render callback wrapper
# ===========================================================================


class SkiaRenderer:
    """Callable ``on_render`` callback that draws the legend overlay.

    Picasso draws the detection bounding boxes via the draw spec before
    this callback fires.  The callback adds the legend (detection list)
    on top of the rendered scene using :class:`SkiaCanvas`.
    """

    def __init__(self):
        self._canvas: SkiaCanvas | None = None
        self._legend_ctx: LegendCtx | None = None

    def __call__(
        self,
        source_id: str,
        fbo_id: int,
        width: int,
        height: int,
        frame: object,
    ) -> None:
        if self._canvas is None:
            self._canvas = SkiaCanvas.from_fbo(fbo_id, width, height)
            self._legend_ctx = LegendCtx()
            print("SkiaCanvas created on Picasso worker thread")

        # Picasso's draw-spec renderer uses its own GrDirectContext on the
        # same GL context.  Tell our context that external GL state changes
        # may have occurred so it re-queries everything.
        self._canvas.gr_context.resetContext()

        draw_legend(
            self._canvas.canvas(),
            self._legend_ctx,
            frame,
            float(width),
            float(height),
        )
        self._canvas.gr_context.flushAndSubmit()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skia + draw-spec encoding pipeline (Picasso engine)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    w, h = args.width, args.height
    sidebar_w = min(340.0, w * 0.22)
    scene_w = w - sidebar_w

    draw_spec = build_draw_spec()
    renderer = SkiaRenderer()
    session = PicassoSession(
        args,
        video_format=VideoFormat.RGBA,
        on_render=renderer,
        draw=draw_spec,
    )

    # -- Push loop ---------------------------------------------------------
    i = 0
    while i < session.limit and session.is_running:
        try:
            buf_ptr = session.acquire_surface(frame_id=i)
        except Exception as e:
            print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
            break

        with nvgstbuf_as_gpu_mat(buf_ptr) as (mat, stream):
            mat.setTo((18, 20, 28, 255), stream=stream)

        pts_ns = i * session.frame_duration_ns
        frame = session.make_frame(pts_ns=pts_ns, duration_ns=session.frame_duration_ns)
        add_objects(frame, scene_w, float(h), i)

        try:
            session.send_frame(frame, buf_ptr)
            i += 1
        except Exception as e:
            print(f"Submit failed at frame {i}: {e}", file=sys.stderr)
            break

    session.shutdown()


if __name__ == "__main__":
    main()
