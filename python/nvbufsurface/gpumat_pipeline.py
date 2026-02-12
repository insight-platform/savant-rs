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

    # Custom resolution and codec with 8 Mbps bitrate
    python gpumat_pipeline.py --width 1280 --height 720 --codec h264 --bitrate 8000000

    # AV1 encoding
    python gpumat_pipeline.py --codec av1 --num-frames 300 --output /tmp/av1_demo.mp4

    # JPEG with quality 95
    python gpumat_pipeline.py --codec jpeg --quality 95 --num-frames 100
"""

from __future__ import annotations

import argparse
import math
import sys

import cv2

from deepstream_nvbufsurface import as_gpu_mat  # noqa: E402
from deepstream_nvbufsurface import VideoFormat  # noqa: E402

from common import EncodingSession, add_common_args


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
        (x1, y1, x2 - x1, border),  # top
        (x1, max(y1, y2 - border), x2 - x1, border),  # bottom
        (x1, y1, border, y2 - y1),  # left
        (max(x1, x2 - border), y1, border, y2 - y1),  # right
    ]:
        if bw > 0 and bh > 0:
            bx2 = min(mat.size()[0], bx + bw)
            by2 = min(mat.size()[1], by + bh)
            if bx2 > bx and by2 > by:
                edge = cv2.cuda.GpuMat(mat, (by, by2), (bx, bx2))
                edge.setTo((0, 255, 0, 255), stream=stream)


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV CUDA GpuMat encoding pipeline (deepstream_encoders SDK)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    w, h = args.width, args.height
    session = EncodingSession(args, video_format=VideoFormat.RGBA)

    # -- Push loop ---------------------------------------------------------
    rect = RectState(w, h)
    i = 0

    while i < session.limit and session.is_running:
        # 1. Acquire NvBufSurface buffer from the encoder's internal pool
        try:
            buf_ptr = session.encoder.acquire_surface(id=i)
        except Exception as e:
            print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
            break

        # 2. Draw with OpenCV CUDA via as_gpu_mat (zero-copy)
        with as_gpu_mat(buf_ptr) as (mat, stream):
            draw_frame(mat, stream, rect, i)

        # 3. Submit the rendered buffer to the encoder
        pts_ns = i * session.frame_duration_ns
        try:
            session.submit(
                buf_ptr,
                frame_id=i,
                pts_ns=pts_ns,
                duration_ns=session.frame_duration_ns,
            )
            i += 1
        except Exception as e:
            print(f"Submit failed at frame {i}: {e}", file=sys.stderr)
            break

        # 4. Pull any ready encoded frames (non-blocking)
        session.pull_encoded()
        session.check_error()
        if not session.is_running:
            break

    session.shutdown()


if __name__ == "__main__":
    main()
