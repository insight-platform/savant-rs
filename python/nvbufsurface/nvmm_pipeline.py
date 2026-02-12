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

    # 300 frames of RGBA -> H.264 to an MP4 file
    python nvmm_pipeline.py --format RGBA --codec h264 --num-frames 300 --output /tmp/test.mp4

    # 600 frames of NV12 -> H.265, no container
    python nvmm_pipeline.py --num-frames 600

    # 100 frames of JPEG, discarded
    python nvmm_pipeline.py --format I420 --codec jpeg --num-frames 100
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import init_cuda
from deepstream_encoders import NvEncoder, EncoderConfig, Codec
from savant_gstreamer import Mp4Muxer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_kb() -> int:
    """Read VmRSS from /proc/self/status (Linux only)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NVMM encoding pipeline (deepstream_encoders SDK)")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--format", type=str, default="NV12", help="Video format (NV12, RGBA, ...)")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--mem-type", type=int, default=0, help="NvBufSurface memory type (0=Default)")
    parser.add_argument("--codec", type=str, default="h265",
                        choices=["h264", "h265", "hevc", "jpeg"],
                        help="Video codec")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output MP4 file path")
    parser.add_argument("--num-frames", "-n", type=int, default=None,
                        help="Number of frames (omit for infinite)")
    args = parser.parse_args()

    # -- Init --------------------------------------------------------------
    Gst.init(None)
    init_cuda(args.gpu_id)

    frame_duration_ns = 1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
    codec = resolve_codec(args.codec)

    # -- Encoder -----------------------------------------------------------
    config = EncoderConfig(
        codec,
        args.width,
        args.height,
        format=args.format,
        fps_num=args.fps,
        fps_den=1,
        gpu_id=args.gpu_id,
        mem_type=args.mem_type,
    )
    encoder = NvEncoder(config)
    print(
        f"Encoder created: {args.width}x{args.height} {args.format} "
        f"@ {args.fps} fps, codec={codec.name()} (gpu {args.gpu_id})"
    )

    # -- Optional MP4 muxer ------------------------------------------------
    muxer: Mp4Muxer | None = None
    if args.output:
        muxer = Mp4Muxer(codec, args.output, fps_num=args.fps)
    else:
        print("No output file — encoded frames will be discarded (benchmark mode)")

    # -- Ctrl-C handling ---------------------------------------------------
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

    # -- Stats reporter thread ---------------------------------------------
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
        pts_ns = i * frame_duration_ns
        try:
            # 1. Acquire NVMM buffer from the encoder's internal pool
            buf_ptr = encoder.acquire_surface(id=i)

            # 2. Submit to encoder
            encoder.submit_frame(buf_ptr, frame_id=i, pts_ns=pts_ns, duration_ns=frame_duration_ns)

            with count_lock:
                frame_count += 1
            i += 1
        except Exception as e:
            print(f"Submit failed at frame {i}: {e}", file=sys.stderr)
            break

        # 3. Pull any ready encoded frames (non-blocking)
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
