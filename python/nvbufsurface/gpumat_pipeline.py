#!/usr/bin/env python3
"""OpenCV CUDA GpuMat NVMM encoding pipeline example.

Demonstrates the ``as_gpu_mat`` / ``from_gpumat`` API: each frame is drawn
with OpenCV CUDA operations on a ``GpuMat`` that wraps the NvBufSurface
CUDA memory in-place (zero-copy), then pushed through NVENC.

A single semi-transparent green rectangle bounces around the frame.

Pipeline::

    GpuMat draw -> NvBufSurface (RGBA NVMM)
        -> appsrc -> nvvideoconvert -> nvv4l2h26Xenc -> h26Xparse -> sink

Usage::

    # 300 frames to MP4
    python gpumat_pipeline.py --num-frames 300 --output /tmp/gpumat_demo.mp4

    # Infinite run (Ctrl-C to stop)
    python gpumat_pipeline.py

    # Custom resolution and codec
    python gpumat_pipeline.py --width 1280 --height 720 --codec h264
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import signal
import sys
import threading
import time

import cv2

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import (
    NvBufSurfaceGenerator,
    as_gpu_mat,
    init_cuda,
    bridge_savant_id_meta_py,
)


# ===========================================================================
# Drawing
# ===========================================================================

class RectState:
    """Bouncing rectangle state."""

    def __init__(self, width: int, height: int):
        self.scene_w = float(width)
        self.scene_h = float(height)
        # Rectangle size
        self.rw = max(80.0, width * 0.12)
        self.rh = max(60.0, height * 0.10)
        # Start near centre
        self.x = (width - self.rw) / 2.0
        self.y = (height - self.rh) / 2.0
        # Velocity (pixels per frame)
        self.vx = 4.0
        self.vy = 3.0

    def step(self) -> tuple[int, int, int, int]:
        """Advance one frame and return (x, y, w, h) as ints."""
        self.x += self.vx
        self.y += self.vy

        # Bounce off edges
        if self.x <= 0:
            self.x = 0
            self.vx = abs(self.vx)
        elif self.x + self.rw >= self.scene_w:
            self.x = self.scene_w - self.rw
            self.vx = -abs(self.vx)

        if self.y <= 0:
            self.y = 0
            self.vy = abs(self.vy)
        elif self.y + self.rh >= self.scene_h:
            self.y = self.scene_h - self.rh
            self.vy = -abs(self.vy)

        return int(self.x), int(self.y), int(self.rw), int(self.rh)


def draw_frame(
    mat: cv2.cuda.GpuMat,
    stream: cv2.cuda.Stream,
    rect: RectState,
    frame_idx: int,
) -> None:
    """Draw one frame: dark background + bouncing green rectangle."""
    # Clear to dark grey
    mat.setTo((20, 20, 28, 255), stream=stream)

    x, y, w, h = rect.step()

    # Pulsing alpha for the fill
    t = frame_idx / 60.0
    alpha = int(120 + 60 * math.sin(t * 2.5))

    # Draw a filled rectangle via a ROI.
    # Clamp to valid region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(mat.size()[0], x + w)  # cols
    y2 = min(mat.size()[1], y + h)  # rows
    if x2 > x1 and y2 > y1:
        roi = cv2.cuda.GpuMat(mat, (y1, y2), (x1, x2))
        roi.setTo((0, 255, 0, alpha), stream=stream)

    # Draw a 2-pixel border (slightly brighter)
    border = 2
    for bx, by, bw, bh in [
        (x1, y1, x2 - x1, border),              # top
        (x1, max(y1, y2 - border), x2 - x1, border),  # bottom
        (x1, y1, border, y2 - y1),               # left
        (max(x1, x2 - border), y1, border, y2 - y1),   # right
    ]:
        if bw > 0 and bh > 0:
            bx2 = min(mat.size()[0], bx + bw)
            by2 = min(mat.size()[1], by + bh)
            if bx2 > bx and by2 > by:
                edge = cv2.cuda.GpuMat(mat, (by, by2), (bx, bx2))
                edge.setTo((0, 255, 0, 255), stream=stream)


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


def container_mux_for_ext(ext: str) -> str | None:
    return {
        "mp4": "qtmux", "m4v": "qtmux",
        "mkv": "matroskamux",
        "ts": "mpegtsmux",
    }.get(ext)


# ===========================================================================
# Low-level GstBuffer push
# ===========================================================================

_libgstapp: ctypes.CDLL | None = None


def _push_buffer_with_ts(appsrc_ptr: int, buf_ptr: int, pts_ns: int, duration_ns: int):
    """Set PTS/duration on a raw GstBuffer and push to AppSrc."""
    global _libgstapp
    if _libgstapp is None:
        _libgstapp = ctypes.CDLL("libgstapp-1.0.so.0")
    # GstBuffer layout: PTS at offset 72, duration at offset 88
    ctypes.c_uint64.from_address(buf_ptr + 72).value = pts_ns
    ctypes.c_uint64.from_address(buf_ptr + 88).value = duration_ns
    _libgstapp.gst_app_src_push_buffer.restype = ctypes.c_int
    _libgstapp.gst_app_src_push_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ret = _libgstapp.gst_app_src_push_buffer(
        ctypes.c_void_p(appsrc_ptr), ctypes.c_void_p(buf_ptr))
    if ret != 0:
        raise RuntimeError(f"gst_app_src_push_buffer returned {ret}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV CUDA GpuMat NVMM encoding pipeline"
    )
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--pool-size", type=int, default=4, help="Buffer pool size")
    parser.add_argument(
        "--codec", type=str, default="h265",
        choices=["h264", "h265", "hevc", "jpeg"],
        help="Video codec",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (.mp4, .mkv, ...)",
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=None,
        help="Number of frames (omit for infinite)",
    )
    args = parser.parse_args()

    # -- Init --------------------------------------------------------------
    Gst.init(None)
    init_cuda(args.gpu_id)

    frame_duration_ns = 1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
    w, h = args.width, args.height

    # -- Generator (RGBA) --------------------------------------------------
    gen = NvBufSurfaceGenerator(
        "RGBA", w, h,
        fps_num=args.fps,
        fps_den=1,
        gpu_id=args.gpu_id,
        pool_size=args.pool_size,
    )

    # -- Pipeline ----------------------------------------------------------
    codec = "h265" if args.codec == "hevc" else args.codec
    if codec == "jpeg":
        enc_name = "nvjpegenc"
        parse_name = "jpegparse"
    else:
        enc_name = f"nvv4l2{codec}enc"
        parse_name = f"{codec}parse"

    pipeline = Gst.Pipeline.new("pipeline")
    appsrc = Gst.ElementFactory.make("appsrc", "src")
    convert = Gst.ElementFactory.make("nvvideoconvert", "convert")
    enc = Gst.ElementFactory.make(enc_name, "enc")
    parse = Gst.ElementFactory.make(parse_name, "parse")
    assert appsrc and convert and enc and parse, "Failed to create pipeline elements"

    # Bridge SavantIdMeta across the encoder
    bridge_savant_id_meta_py(hash(enc))

    container_mux = None
    if args.output:
        sink = Gst.ElementFactory.make("filesink", "sink")
        assert sink, "Failed to create filesink"
        sink.set_property("location", args.output)
        ext = os.path.splitext(args.output)[1].lstrip(".").lower()
        mux_factory = container_mux_for_ext(ext)
        if mux_factory:
            container_mux = Gst.ElementFactory.make(mux_factory, "cmux")
            assert container_mux, f"Failed to create {mux_factory}"
    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        assert sink, "Failed to create fakesink"
        sink.set_property("sync", False)

    caps = Gst.Caps.from_string(gen.nvmm_caps_str())
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("stream-type", 0)

    chain: list[Gst.Element] = [appsrc, convert, enc, parse]
    if container_mux:
        chain.append(container_mux)
    chain.append(sink)

    for elem in chain:
        pipeline.add(elem)
    for idx in range(len(chain) - 1):
        assert chain[idx].link(chain[idx + 1])

    if args.output:
        cmux_label = f"{container_mux.get_factory().get_name()} -> " if container_mux else ""
        sink_label = f"{cmux_label}filesink({args.output})"
    else:
        sink_label = "fakesink"
    print(f"Pipeline: GpuMat -> appsrc(RGBA) -> nvvideoconvert -> {enc_name} -> {parse_name} -> {sink_label}")
    print(f"Resolution: {w}x{h} @ {args.fps} fps, pool_size={args.pool_size}")

    ret = pipeline.set_state(Gst.State.PLAYING)
    assert ret != Gst.StateChangeReturn.FAILURE, "Failed to start pipeline"

    limit = args.num_frames if args.num_frames is not None else sys.maxsize
    if args.num_frames is not None:
        print(f"Pipeline running ({args.num_frames} frames)...\n")
    else:
        print("Pipeline running (Ctrl-C to stop)...\n")

    # -- Ctrl-C ------------------------------------------------------------
    running = True

    def _sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    # -- Stats reporter ----------------------------------------------------
    frame_count = 0
    count_lock = threading.Lock()

    def stats_reporter():
        nonlocal frame_count
        last_count = 0
        last_time = time.monotonic()
        while running:
            time.sleep(1.0)
            now = time.monotonic()
            with count_lock:
                count = frame_count
            elapsed = now - last_time
            delta = count - last_count
            fps = delta / elapsed if elapsed > 0 else 0.0
            rss = rss_kb()
            print(f"frames: {count:>8}  |  fps: {fps:>8.1f}  |  RSS: {rss // 1024} MB")
            last_count = count
            last_time = now

    stats_thread = threading.Thread(target=stats_reporter, daemon=True)
    stats_thread.start()

    # -- Push loop ---------------------------------------------------------
    appsrc_ptr = hash(appsrc)
    bus = pipeline.get_bus()
    rect = RectState(w, h)
    i = 0

    while i < limit and running:
        # 1. Acquire NvBufSurface buffer
        try:
            buf_ptr = gen.acquire_surface(id=i)
        except Exception as e:
            print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
            break

        # 2. Draw with OpenCV CUDA via as_gpu_mat
        with as_gpu_mat(buf_ptr) as (mat, stream):
            draw_frame(mat, stream, rect, i)

        # 3. Set timestamps and push to appsrc
        pts_ns = i * frame_duration_ns
        try:
            _push_buffer_with_ts(appsrc_ptr, buf_ptr, pts_ns, frame_duration_ns)
            with count_lock:
                frame_count += 1
            i += 1
        except Exception as e:
            print(f"Push failed at frame {i}: {e}", file=sys.stderr)
            break

        msg = bus.pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"Pipeline error: {err}", file=sys.stderr)
            break

    # -- Shutdown ----------------------------------------------------------
    print("\nStopping...")
    running = False

    NvBufSurfaceGenerator.send_eos(appsrc_ptr)
    bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    stats_thread.join(timeout=2)

    with count_lock:
        total = frame_count
    print(f"Total frames pushed: {total}")


if __name__ == "__main__":
    main()
