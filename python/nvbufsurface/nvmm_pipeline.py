#!/usr/bin/env python3
"""NVMM encoding pipeline example using the Picasso engine.

Pushes NVMM frames through the Picasso rendering/encoding pipeline as
fast as possible, printing throughput statistics every second.  Optionally
muxes the encoded bitstream into an MP4 container via a minimal GStreamer
pipeline.

The Picasso engine handles all encoding internals (encoder creation,
format conversion, PTS validation).  The sample only needs to:

1. Configure and create a :class:`PicassoSession`.
2. Acquire GPU buffers, submit them to the engine.
3. (Encoded frames are delivered asynchronously via callback and
   optionally pushed into the muxer.)

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

from savant_rs.deepstream import VideoFormat  # noqa: E402

from common import PicassoSession, add_common_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVMM encoding pipeline (Picasso engine)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--format", type=str, default="NV12", help="Video format (NV12, RGBA, ...)"
    )
    args = parser.parse_args()

    video_format = VideoFormat.from_name(args.format)
    session = PicassoSession(args, video_format=video_format)

    # -- Push loop ---------------------------------------------------------
    i = 0
    while i < session.limit and session.is_running:
        pts_ns = i * session.frame_duration_ns
        try:
            buf = session.acquire_surface(frame_id=i)
            session.submit(
                buf,
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
