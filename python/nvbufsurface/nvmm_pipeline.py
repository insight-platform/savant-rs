#!/usr/bin/env python3
"""NVMM encoding pipeline example using deepstream_encoders SDK.

Pushes NVMM frames through the NvEncoder SDK as fast as possible, printing
throughput statistics every second.  Optionally muxes the encoded bitstream
into an MP4 container via a minimal GStreamer pipeline.

The SDK handles all encoding internals (encoder creation, format conversion,
B-frame prevention, PTS validation).  The sample only needs to:

1. Configure and create an ``NvEncoder``.
2. Acquire GPU buffers, submit them to the encoder.
3. Pull encoded frames and (optionally) push them into a muxer.

Output pipeline (when ``--output`` is given)::

    appsrc (bitstream) -> h26Xparse -> qtmux -> filesink

Usage::

    # Infinite run, discard output (benchmark mode)
    python nvmm_pipeline.py --width 1920 --height 1080

    # 300 frames of RGBA -> H.264 at 8 Mbps to an MP4 file
    python nvmm_pipeline.py --format RGBA --codec h264 --bitrate 8000000 --num-frames 300 --output /tmp/test.mp4

    # 600 frames of NV12 -> H.265, no container
    python nvmm_pipeline.py --num-frames 600

    # 100 frames of JPEG at quality 95, discarded
    python nvmm_pipeline.py --format I420 --codec jpeg --quality 95 --num-frames 100

    # 300 frames of AV1 to an MP4 file
    python nvmm_pipeline.py --codec av1 --num-frames 300 --output /tmp/av1_test.mp4
"""

from __future__ import annotations

import argparse
import sys

from deepstream_nvbufsurface import VideoFormat  # noqa: E402

from common import EncodingSession, add_common_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVMM encoding pipeline (deepstream_encoders SDK)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--format", type=str, default="NV12", help="Video format (NV12, RGBA, ...)"
    )
    parser.add_argument(
        "--mem-type", type=int, default=0, help="NvBufSurface memory type (0=Default)"
    )
    args = parser.parse_args()

    video_format = VideoFormat.from_name(args.format)
    session = EncodingSession(args, video_format=video_format)

    # -- Push loop ---------------------------------------------------------
    i = 0
    while i < session.limit and session.is_running:
        pts_ns = i * session.frame_duration_ns
        try:
            buf_ptr = session.encoder.acquire_surface(id=i)
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

        session.pull_encoded()
        session.check_error()
        if not session.is_running:
            break

    session.shutdown()


if __name__ == "__main__":
    main()
