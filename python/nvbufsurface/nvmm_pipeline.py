#!/usr/bin/env python3
"""NVMM encoding pipeline example using the Picasso engine.

Acquires NvBufSurface-backed GPU buffers from the generator pool,
fills each frame with a cycling solid colour via ``nvgstbuf_as_gpu_mat``
(zero-copy GpuMat view), and submits them to the Picasso encoder.

This is the simplest pipeline: no bounding-box drawing, no Skia overlay,
no ``on_gpumat`` / ``on_render`` callbacks — just colour fill + encode.

On Jetson the NvBufSurface pool allocates NVMM surface-array memory
(required by VIC-backed NvBufSurfTransform); using raw CUDA device
memory (``GpuMatCudaArray``) would fail because VIC cannot read from
``NVBUF_MEM_CUDA_DEVICE``.

Output pipeline (when ``--output`` is given)::

    appsrc (bitstream) -> h26Xparse -> qtmux -> filesink

Usage::

    # Infinite run, discard output (benchmark mode)
    python nvmm_pipeline.py --width 1920 --height 1080

    # 300 frames of RGBA -> H.264 at 8 Mbps to an MP4 file
    python nvmm_pipeline.py --codec h264 --bitrate 8000000 --num-frames 300 --output /tmp/test.mp4

    # 600 frames of RGBA -> H.265, no container
    python nvmm_pipeline.py --num-frames 600

    # 100 frames of RGBA -> JPEG at quality 95, discarded
    python nvmm_pipeline.py --codec jpeg --quality 95 --num-frames 100

    # 300 frames of AV1 to an MP4 file
    python nvmm_pipeline.py --codec av1 --num-frames 300 --output /tmp/av1_test.mp4
"""

from __future__ import annotations

import argparse
import sys

from savant_rs.deepstream import (
    SurfaceView,
    VideoFormat,
    nvbuf_as_gpu_mat,
)

from common import PicassoSession, add_common_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVMM encoding pipeline (Picasso engine, NvBufSurface source)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    session = PicassoSession(args, video_format=VideoFormat.RGBA)

    # -- Push loop ---------------------------------------------------------
    i = 0
    while i < session.limit and session.is_running:
        pts_ns = i * session.frame_duration_ns
        for s in range(session.jobs):
            try:
                buf = session.acquire_surface(source_idx=s, frame_id=i)
            except Exception as e:
                print(f"acquire_surface failed at frame {i}: {e}", file=sys.stderr)
                break

            view = SurfaceView.from_buffer(buf, 0)
            with nvbuf_as_gpu_mat(
                view.data_ptr, view.pitch, view.width, view.height
            ) as (mat, stream):
                grey = i % 255
                mat.setTo((grey, grey, grey, 255), stream=stream)

            try:
                session.submit(
                    view,
                    source_idx=s,
                    pts_ns=pts_ns,
                    duration_ns=session.frame_duration_ns,
                )
            except Exception as e:
                print(f"Submit failed at frame {i} src {s}: {e}", file=sys.stderr)
                break
        else:
            i += 1
            continue
        break

    session.shutdown()


if __name__ == "__main__":
    main()
