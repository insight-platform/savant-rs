#!/usr/bin/env python3
"""OpenCV CUDA GpuMat encoding pipeline using deepstream_encoders SDK.

Demonstrates the ``as_gpu_mat`` API: each frame is drawn with OpenCV CUDA
operations on a ``GpuMat`` that wraps the NvBufSurface CUDA memory
in-place (zero-copy), then encoded via the NvEncoder SDK.

A single semi-transparent green rectangle bounces around the frame.

The SDK handles encoding, format conversion, B-frame prevention, and PTS
validation.  The sample only uses a trivial GStreamer pipeline for MP4
muxing when ``--output`` is given.

Output pipeline (when ``--output`` is given)::

    appsrc (bitstream) -> h26Xparse -> qtmux -> filesink

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
import math
import signal
import sys
import threading
import time

import cv2

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import as_gpu_mat, init_cuda
from deepstream_encoders import NvEncoder, EncoderConfig, Codec
from savant_gstreamer import Mp4Muxer


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


def resolve_codec(name: str) -> Codec:
    """Map CLI codec name to Codec enum."""
    return Codec.from_name("hevc" if name == "h265" else name)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV CUDA GpuMat encoding pipeline (deepstream_encoders SDK)"
    )
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--codec", type=str, default="h265",
        choices=["h264", "h265", "hevc", "jpeg"],
        help="Video codec",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output MP4 file path",
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
    codec = resolve_codec(args.codec)

    # -- Encoder (RGBA - GpuMat draws in RGBA) -----------------------------
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
    rect = RectState(w, h)
    i = 0

    while i < limit and running:
        # 1. Acquire NvBufSurface buffer from the encoder's internal pool
        try:
            buf_ptr = encoder.acquire_surface(id=i)
        except Exception as e:
            print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
            break

        # 2. Draw with OpenCV CUDA via as_gpu_mat (zero-copy)
        with as_gpu_mat(buf_ptr) as (mat, stream):
            draw_frame(mat, stream, rect, i)

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
