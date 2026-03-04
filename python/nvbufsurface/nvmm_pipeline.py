#!/usr/bin/env python3
"""NVMM encoding pipeline example using the Picasso engine.

Allocates ``cv2.cuda.GpuMat`` frames, fills them with a solid colour,
and submits them to Picasso via ``__cuda_array_interface__``.  This
demonstrates the zero-copy path where plain CUDA memory (no
NvBufSurface) is handed directly to the encoder.

The Picasso engine handles all encoding internals (encoder creation,
format conversion, PTS validation).  The sample only needs to:

1. Configure and create a :class:`PicassoSession`.
2. Allocate a ``GpuMat``, draw into it, wrap with
   :class:`GpuMatCudaArray`, and submit.
3. (Encoded frames are delivered asynchronously via callback and
   optionally pushed into the muxer.)

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

from savant_rs.deepstream import VideoFormat  # noqa: E402

from common import GpuMatCudaArray, PicassoSession, add_common_args, make_gpu_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVMM encoding pipeline (Picasso engine, GpuMat source)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    session = PicassoSession(args, video_format=VideoFormat.RGBA, use_generator=False)

    mat = make_gpu_mat(args.width, args.height)

    # -- Push loop ---------------------------------------------------------
    i = 0
    while i < session.limit and session.is_running:
        mat.setTo((20, 20, 28, 255))
        cuda_frame = GpuMatCudaArray(mat)

        pts_ns = i * session.frame_duration_ns
        try:
            session.submit(
                cuda_frame,
                pts_ns=pts_ns,
                duration_ns=session.frame_duration_ns,
            )
            i += 1
        except Exception as e:
            print(f"Submit failed at frame {i}: {e}", file=sys.stderr)
            break

    session.shutdown()


if __name__ == "__main__":
    main()
