#!/usr/bin/env python3
"""Skia GPU-rendered NVMM encoding pipeline using deepstream_encoders SDK.

Demonstrates GPU-accelerated Skia rendering + NvEncoder SDK:
each frame is drawn with skia-python on an OpenGL texture (GPU), copied
into the NvBufSurface via CUDA-GL interop (GPU-to-GPU, no CPU), and
encoded with NVENC through the deepstream_encoders SDK.

The entire pixel data path is GPU-side -- no CPU copies occur.

All EGL/GL/CUDA-GL interop boilerplate is handled by the SkiaContext
PyO3 class from ``deepstream_nvbufsurface``, exposed through the
``SkiaCanvas`` helper.

The SDK handles encoding, format conversion, B-frame prevention, and
PTS validation.  The sample only uses a trivial GStreamer pipeline for
MP4 muxing when ``--output`` is given.

Output pipeline (when ``--output`` is given)::

    appsrc (bitstream) -> h26Xparse -> qtmux -> filesink

Usage::

    # 300 frames to MP4
    python skia_pipeline.py --num-frames 300 --output /tmp/skia_demo.mp4

    # Infinite run (Ctrl-C to stop)
    python skia_pipeline.py

    # Custom resolution and codec
    python skia_pipeline.py --width 1280 --height 720 --codec h264

    # JPEG output (no MP4 container)
    python skia_pipeline.py --codec jpeg --num-frames 100
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass

import skia

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import SkiaCanvas, init_cuda
from deepstream_encoders import NvEncoder, EncoderConfig, Codec
from savant_gstreamer import Mp4Muxer


# ===========================================================================
# SVG asset — pre-rendered to a GPU texture for zero per-frame transfer
# ===========================================================================

TIGER_SVG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/d/d2/"
    "Ghostscript_tiger_%28original_background%29.svg"
)


def _load_svg_as_gpu_texture(
    source: str,
    gr_context: skia.GrDirectContext,
    *,
    render_width: int = 0,
    render_height: int = 0,
) -> skia.Image | None:
    """Download/read an SVG, render it to a raster, and upload to GPU texture.

    The SVG is rasterised once at full resolution (or *render_width* x
    *render_height* if given), then uploaded to a GPU texture via
    ``makeTextureImage``.  Subsequent ``drawImage`` calls on a GPU
    canvas are pure GPU work with zero CPU->GPU transfers.

    Returns:
        A GPU-backed ``skia.Image``, or ``None`` on failure.
    """
    # -- Fetch bytes -------------------------------------------------------
    try:
        if source.startswith(("http://", "https://")):
            print(f"Downloading SVG from {source} ...")
            req = urllib.request.Request(source, headers={"User-Agent": "skia_pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
        else:
            with open(source, "rb") as f:
                data = f.read()
        print(f"SVG fetched ({len(data)} bytes)")
    except Exception as exc:
        print(f"Warning: could not load SVG: {exc}", file=sys.stderr)
        return None

    # -- Parse SVG DOM -----------------------------------------------------
    stream = skia.MemoryStream.MakeDirect(data)
    dom = skia.SVGDOM.MakeFromStream(stream)
    if dom is None:
        print("Warning: skia.SVGDOM.MakeFromStream returned None", file=sys.stderr)
        return None

    svg_size = dom.containerSize()
    svg_w, svg_h = svg_size.width(), svg_size.height()
    if svg_w <= 0 or svg_h <= 0:
        print("Warning: SVG has zero dimensions", file=sys.stderr)
        return None

    # -- Determine render size ---------------------------------------------
    rw = render_width if render_width > 0 else int(svg_w)
    rh = render_height if render_height > 0 else int(svg_h)

    # -- Render SVG to a CPU raster (one-time) -----------------------------
    info = skia.ImageInfo.MakeN32Premul(rw, rh)
    surface = skia.Surface.MakeRaster(info)
    if surface is None:
        print("Warning: failed to create raster surface for SVG", file=sys.stderr)
        return None

    svg_canvas = surface.getCanvas()
    svg_canvas.clear(skia.ColorTRANSPARENT)
    # Scale to fill the render area
    svg_canvas.scale(rw / svg_w, rh / svg_h)
    dom.render(svg_canvas)
    raster_snapshot = surface.makeImageSnapshot()

    raw_mb = rw * rh * 4 / (1024 * 1024)
    print(f"SVG pre-rendered: {rw}x{rh} (~{raw_mb:.1f} MB raw RGBA)")

    # -- Upload to GPU texture (one-time) ----------------------------------
    gpu_image = raster_snapshot.makeTextureImage(gr_context)
    if gpu_image is not None:
        print("  -> uploaded to GPU texture (zero CPU transfer per frame)")
        return gpu_image

    print("Warning: makeTextureImage failed for SVG raster", file=sys.stderr)
    return None


# ===========================================================================
# Detection class definitions
# ===========================================================================

NUM_BOXES = 20


@dataclass
class DetectionClass:
    name: str
    color: int  # ARGB as a 32-bit int


CLASSES = [
    DetectionClass("person",  0xFFFF5050),
    DetectionClass("car",     0xFF50C8FF),
    DetectionClass("truck",   0xFFFFB428),
    DetectionClass("bicycle", 0xFF50FF78),
    DetectionClass("dog",     0xFFDC64FF),
    DetectionClass("bus",     0xFFFFFF50),
    DetectionClass("bike",    0xFF50FFFF),
    DetectionClass("sign",    0xFFFF8C8C),
]


def _with_alpha(c: int, a: int) -> int:
    return (a << 24) | (c & 0x00FFFFFF)


# ===========================================================================
# Pseudo-random
# ===========================================================================

def pseudo_rand(seed1: int, seed2: int) -> float:
    MASK64 = 0xFFFF_FFFF_FFFF_FFFF
    h = ((seed1 * 6364136223846793005) + seed2) & MASK64
    h ^= (h >> 33)
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= (h >> 33)
    return (h & 0x00FF_FFFF) / 0x0100_0000


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    class_idx: int
    confidence: float
    id: int


def hsv_to_color(h_deg: float, s: float, v: float) -> int:
    h = ((h_deg % 360.0) + 360.0) % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    sector = int(h / 60.0) % 6
    if sector == 0:   r, g, b = c, x, 0.0
    elif sector == 1: r, g, b = x, c, 0.0
    elif sector == 2: r, g, b = 0.0, c, x
    elif sector == 3: r, g, b = 0.0, x, c
    elif sector == 4: r, g, b = x, 0.0, c
    else:             r, g, b = c, 0.0, x
    return (0xFF << 24) | (int((r+m)*255) << 16) | (int((g+m)*255) << 8) | int((b+m)*255)


# ===========================================================================
# Drawing context -- caches fonts/paints across frames
# ===========================================================================

class DrawCtx:
    """Cached fonts and paints for efficient per-frame rendering."""

    def __init__(self):
        bold_tf = skia.Typeface("monospace", skia.FontStyle.Bold())
        normal_tf = skia.Typeface("monospace", skia.FontStyle.Normal())
        self.label_font = skia.Font(bold_tf, 16)
        self.title_font = skia.Font(bold_tf, 18)
        self.legend_font = skia.Font(normal_tf, 13)
        self.footer_font = skia.Font(bold_tf, 14)

        self.white_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        self.black_paint = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)
        self.sidebar_bg_paint = skia.Paint(Color=skia.Color(15, 18, 25, 210))
        self.separator_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 100),
            StrokeWidth=1.0, Style=skia.Paint.kStroke_Style,
        )
        self.divider_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 60),
            StrokeWidth=1.0, Style=skia.Paint.kStroke_Style,
        )
        self.footer_bg_paint = skia.Paint(Color=skia.Color(0, 0, 0, 180))
        self.footer_text_paint = skia.Paint(Color=skia.Color(200, 200, 200, 200), AntiAlias=True)
        self.fill_paint = skia.Paint(AntiAlias=True)
        self.stroke_paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.0)
        self.label_bg_paint = skia.Paint()
        self.dot_paint = skia.Paint(AntiAlias=True)
        self.legend_text_paint = skia.Paint(AntiAlias=True, Color=skia.Color(255, 255, 255, 220))
        self.bg_paint = skia.Paint()
        self.svg_paint = skia.Paint(AntiAlias=True)
        self.boxes: list[BBox] = []
        self.svg_gpu_image: skia.Image | None = None


# ===========================================================================
# draw_frame
# ===========================================================================

def draw_frame(skia_canvas: SkiaCanvas, ctx: DrawCtx, frame_idx: int,
               width: float, height: float) -> None:
    canvas = skia_canvas.canvas()

    # -- Background gradient -----------------------------------------------
    hue_shift = (frame_idx * 0.3) % 360.0
    bg1 = hsv_to_color(hue_shift, 0.15, 0.10)
    bg2 = hsv_to_color(hue_shift + 40.0, 0.20, 0.18)
    shader = skia.GradientShader.MakeLinear(
        points=[skia.Point(0, 0), skia.Point(width, height)],
        colors=[bg1, bg2],
    )
    if shader:
        bg_with_shader = skia.Paint(ctx.bg_paint)
        bg_with_shader.setShader(shader)
        canvas.drawRect(skia.Rect.MakeWH(width, height), bg_with_shader)
    else:
        canvas.clear(skia.Color(18, 20, 28))

    # -- Sidebar dimensions ------------------------------------------------
    sidebar_w = min(340.0, width * 0.22)
    scene_w = width - sidebar_w
    t = frame_idx / 60.0

    # -- SVG tiger (GPU texture, centered, gently floating) -----------------
    if ctx.svg_gpu_image is not None:
        iw = ctx.svg_gpu_image.width()
        ih = ctx.svg_gpu_image.height()
        if iw > 0 and ih > 0:
            # Scale to fit ~55% of the scene area, preserving aspect ratio
            target_w = scene_w * 0.55
            target_h = height * 0.55
            img_scale = min(target_w / iw, target_h / ih)
            scaled_w = iw * img_scale
            scaled_h = ih * img_scale

            # Center + gentle floating motion
            cx = (scene_w - scaled_w) / 2.0 + math.sin(t * 0.4) * 15.0
            cy = (height - scaled_h) / 2.0 + math.cos(t * 0.3) * 10.0

            canvas.save()
            canvas.translate(cx, cy)
            canvas.scale(img_scale, img_scale)

            # Render with slight transparency so detections are visible on top
            canvas.saveLayerAlpha(None, 160)
            canvas.drawImage(ctx.svg_gpu_image, 0, 0, paint=ctx.svg_paint)
            canvas.restore()  # layer

            canvas.restore()  # translate+scale

    # -- Generate bounding boxes -------------------------------------------
    ctx.boxes.clear()
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

        ctx.boxes.append(BBox(
            x=max(0.0, min(cx - bw / 2.0, scene_w - bw)),
            y=max(0.0, min(cy - bh / 2.0, height - bh)),
            w=bw, h=bh,
            class_idx=class_idx,
            confidence=confidence,
            id=i,
        ))

    # -- Draw bounding boxes with semitransparent circles --------------------
    for b in ctx.boxes:
        cls = CLASSES[b.class_idx]
        rect = skia.Rect.MakeXYWH(b.x, b.y, b.w, b.h)

        # Semitransparent circle centered on the bbox, pulsing radius
        cx = b.x + b.w / 2.0
        cy = b.y + b.h / 2.0
        base_r = max(b.w, b.h) * 0.4
        pulse = 0.85 + 0.15 * math.sin(t * 2.0 + b.id * 0.7)
        radius = base_r * pulse

        ctx.fill_paint.setColor(_with_alpha(cls.color, 35))
        canvas.drawCircle(cx, cy, radius, ctx.fill_paint)

        ctx.stroke_paint.setColor(_with_alpha(cls.color, 80))
        ctx.stroke_paint.setStrokeWidth(1.5)
        canvas.drawCircle(cx, cy, radius, ctx.stroke_paint)
        ctx.stroke_paint.setStrokeWidth(2.0)

        # Filled bbox
        ctx.fill_paint.setColor(_with_alpha(cls.color, 50))
        canvas.drawRect(rect, ctx.fill_paint)

        # Bbox outline
        ctx.stroke_paint.setColor(cls.color)
        canvas.drawRect(rect, ctx.stroke_paint)

        # Label
        label_text = f"{cls.name} #{b.id} {b.confidence * 100:.0f}%"
        tw = ctx.label_font.measureText(label_text)
        lh = 22.0
        lx = b.x
        ly = b.y - lh - 2.0 if b.y >= lh + 2.0 else b.y

        ctx.label_bg_paint.setColor(_with_alpha(cls.color, 200))
        canvas.drawRect(skia.Rect.MakeXYWH(lx, ly, tw + 10, lh), ctx.label_bg_paint)

        canvas.drawString(label_text, lx + 5, ly + lh - 5, ctx.label_font, ctx.black_paint)

    # -- Sidebar -----------------------------------------------------------
    sx = scene_w
    canvas.drawRect(skia.Rect.MakeXYWH(sx, 0, sidebar_w, height), ctx.sidebar_bg_paint)
    canvas.drawLine(sx, 0, sx, height, ctx.separator_paint)

    canvas.drawString("DETECTIONS", sx + 12, 28, ctx.title_font, ctx.white_paint)
    canvas.drawLine(sx + 8, 36, sx + sidebar_w - 8, 36, ctx.divider_paint)

    y_off = 52.0
    row_h = 18.0

    for i, b in enumerate(ctx.boxes):
        if y_off + row_h > height - 40.0:
            ctx.legend_text_paint.setColor(skia.Color(255, 255, 255, 180))
            canvas.drawString(
                f"... +{NUM_BOXES - i} more",
                sx + 12, y_off + 14, ctx.legend_font, ctx.legend_text_paint,
            )
            break

        cls = CLASSES[b.class_idx]
        ctx.dot_paint.setColor(cls.color)
        canvas.drawCircle(sx + 16, y_off + 6, 4.0, ctx.dot_paint)

        entry = f"{cls.name:<8} #{b.id:<2} ({int(b.x):>4},{int(b.y):>4}) {b.confidence * 100:>3.0f}%"
        ctx.legend_text_paint.setColor(skia.Color(255, 255, 255, 220))
        canvas.drawString(entry, sx + 26, y_off + 10, ctx.legend_font, ctx.legend_text_paint)

        y_off += row_h

    # -- Footer ------------------------------------------------------------
    canvas.drawRect(skia.Rect.MakeXYWH(sx, height - 32, sidebar_w, 32), ctx.footer_bg_paint)
    footer = f"F:{frame_idx:>6} {int(width)}x{int(height)} {NUM_BOXES}obj"
    canvas.drawString(footer, sx + 10, height - 11, ctx.footer_font, ctx.footer_text_paint)


# ===========================================================================
# Helpers
# ===========================================================================

def rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def resolve_codec(name: str) -> Codec:
    """Map CLI codec name to Codec enum."""
    return Codec.from_name("hevc" if name == "h265" else name)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Skia GPU-rendered encoding pipeline (deepstream_encoders SDK)")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--codec", type=str, default="h265",
                        choices=["h264", "h265", "hevc", "jpeg"],
                        help="Video codec")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output MP4 file path")
    parser.add_argument("--num-frames", "-n", type=int, default=None,
                        help="Number of frames (omit for infinite)")
    parser.add_argument("--svg", type=str, default=TIGER_SVG_URL,
                        help="URL or local path to SVG file (use 'none' to disable)")
    args = parser.parse_args()

    # -- Init --------------------------------------------------------------
    Gst.init(None)
    init_cuda(args.gpu_id)

    frame_duration_ns = 1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
    w, h = args.width, args.height
    codec = resolve_codec(args.codec)

    # -- GPU Skia canvas (all EGL/GL/CUDA managed by SkiaContext) ----------
    skia_canvas = SkiaCanvas.create(w, h, gpu_id=args.gpu_id)
    draw_ctx = DrawCtx()
    print(f"SkiaCanvas created: {w}x{h} (gpu {args.gpu_id})")

    # -- Load SVG as GPU texture -------------------------------------------
    if args.svg and args.svg.lower() != "none":
        draw_ctx.svg_gpu_image = _load_svg_as_gpu_texture(
            args.svg,
            skia_canvas.gr_context,
            render_width=w,
            render_height=h,
        )

    # -- Encoder (RGBA - Skia's native format) -----------------------------
    config = EncoderConfig(
        codec, w, h,
        format="RGBA",
        fps_num=args.fps,
        fps_den=1,
        gpu_id=args.gpu_id,
    )
    encoder = NvEncoder(config)
    print(
        f"Encoder created: {w}x{h} RGBA @ {args.fps} fps, "
        f"codec={codec.name()} (gpu {args.gpu_id})"
    )

    # -- Optional MP4 muxer ------------------------------------------------
    muxer: Mp4Muxer | None = None
    if args.output:
        muxer = Mp4Muxer(codec, args.output, fps_num=args.fps)
    else:
        print("No output file — encoded frames will be discarded (benchmark mode)")

    # -- Run ---------------------------------------------------------------
    limit = args.num_frames if args.num_frames is not None else sys.maxsize
    if args.num_frames is not None:
        print(f"Running ({args.num_frames} frames)...\n")
    else:
        print("Running (Ctrl-C to stop)...\n")

    running = True

    def _sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    # -- Stats reporter ----------------------------------------------------
    frame_count = 0
    encoded_count = 0
    encoded_bytes = 0
    count_lock = threading.Lock()

    def stats_reporter():
        nonlocal frame_count, encoded_count, encoded_bytes
        last_count = 0
        last_time = time.monotonic()
        while running:
            time.sleep(1.0)
            now = time.monotonic()
            with count_lock:
                count = frame_count
                enc = encoded_count
                ebytes = encoded_bytes
            elapsed = now - last_time
            delta = count - last_count
            fps = delta / elapsed if elapsed > 0 else 0.0
            rss = rss_kb()
            print(
                f"submitted: {count:>8}  |  encoded: {enc:>8}  |  "
                f"fps: {fps:>8.1f}  |  bitstream: {ebytes // 1024} KB  |  "
                f"RSS: {rss // 1024} MB"
            )
            last_count = count
            last_time = now

    stats_thread = threading.Thread(target=stats_reporter, daemon=True)
    stats_thread.start()

    # -- Push loop ---------------------------------------------------------
    i = 0

    while i < limit and running:
        # 1. Draw with Skia (GPU)
        draw_frame(skia_canvas, draw_ctx, i, float(w), float(h))

        # 2. Acquire NvBufSurface buffer and render Skia into it
        try:
            buf_ptr = encoder.acquire_surface(id=i)
        except Exception as e:
            print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
            break

        try:
            skia_canvas.render_to_nvbuf(buf_ptr)
        except Exception as e:
            print(f"render_to_nvbuf failed at frame {i}: {e}", file=sys.stderr)
            break

        # 3. Submit the rendered buffer to the encoder
        pts_ns = i * frame_duration_ns
        try:
            encoder.submit_frame(buf_ptr, frame_id=i, pts_ns=pts_ns, duration_ns=frame_duration_ns)
            with count_lock:
                frame_count += 1
            i += 1
        except Exception as e:
            print(f"Submit failed at frame {i}: {e}", file=sys.stderr)
            break

        # 4. Pull any ready encoded frames (non-blocking)
        while True:
            frame = encoder.pull_encoded()
            if frame is None:
                break
            with count_lock:
                encoded_count += 1
                encoded_bytes += frame.size
            if muxer is not None:
                muxer.push(frame.data, frame.pts_ns, frame.dts_ns, frame.duration_ns)

        # Check encoder for pipeline errors
        try:
            encoder.check_error()
        except Exception as e:
            print(f"Encoder error: {e}", file=sys.stderr)
            break

    # -- Shutdown ----------------------------------------------------------
    print("\nStopping...")
    running = False

    # Drain remaining encoded frames from the encoder
    remaining = encoder.finish()
    for frame in remaining:
        with count_lock:
            encoded_count += 1
            encoded_bytes += frame.size
        if muxer is not None:
            muxer.push(frame.data, frame.pts_ns, frame.dts_ns, frame.duration_ns)

    # Finalize muxer
    if muxer is not None:
        muxer.finish()
        print(f"Output written to: {args.output}")

    stats_thread.join(timeout=2)

    with count_lock:
        total_submitted = frame_count
        total_encoded = encoded_count
        total_bytes = encoded_bytes
    print(
        f"Total submitted: {total_submitted}  |  "
        f"Total encoded: {total_encoded}  |  "
        f"Bitstream: {total_bytes // 1024} KB"
    )


if __name__ == "__main__":
    main()
