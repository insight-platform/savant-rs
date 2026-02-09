#!/usr/bin/env python3
"""NVMM encoding pipeline example (Python port).

Pushes NVMM frames via NvBufSurfaceGenerator through a DeepStream
encoding pipeline as fast as possible, printing throughput statistics
every second.

The generator allocates buffers in the requested format (RGBA, NV12, etc.)
directly in GPU memory.  When the format is not encoder-native (i.e. not
NV12 or I420), ``nvvideoconvert`` is inserted to do GPU-to-GPU conversion
before encoding -- no CPU involvement at any stage.

Pipeline (NV12/I420, no output file)::

    appsrc (memory:NVMM) -> nvv4l2h26Xenc -> h26Xparse -> fakesink

Pipeline (RGBA, with .mp4 output)::

    appsrc (memory:NVMM) -> nvvideoconvert -> nvv4l2h26Xenc -> h26Xparse -> qtmux -> filesink

Pipeline (JPEG)::

    appsrc (memory:NVMM) -> [nvvideoconvert ->] nvjpegenc -> jpegparse -> sink

Usage::

    # Infinite run, discard output (benchmark mode)
    python nvmm_pipeline.py --width 1920 --height 1080

    # 300 frames of RGBA -> H.264 to an MP4 file
    python nvmm_pipeline.py --format RGBA --codec h264 --num-frames 300 --output /tmp/test.mp4

    # 600 frames of NV12 -> H.265 raw elementary stream
    python nvmm_pipeline.py --num-frames 600 --output /tmp/test.h265

    # Infinite run to a Matroska container (Ctrl-C to stop)
    python nvmm_pipeline.py --codec h265 --output /tmp/test.mkv

    # 100 frames of JPEG (I420 native, no conversion)
    python nvmm_pipeline.py --format I420 --codec jpeg --num-frames 100
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from deepstream_nvbufsurface import NvBufSurfaceGenerator, init_cuda, bridge_savant_id_meta_py


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


def container_mux_for_ext(ext: str) -> str | None:
    """Return a GStreamer muxer element name for a container extension, or None."""
    return {
        "mp4": "qtmux",
        "m4v": "qtmux",
        "mkv": "matroskamux",
        "webm": "matroskamux",
        "ts": "mpegtsmux",
    }.get(ext)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NVMM encoding pipeline")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--format", type=str, default="NV12", help="Video format (NV12, RGBA, ...)")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--mem-type", type=int, default=0, help="NvBufSurface memory type (0=Default)")
    parser.add_argument("--pool-size", type=int, default=4, help="Buffer pool size")
    parser.add_argument("--codec", type=str, default="h265",
                        choices=["h264", "h265", "hevc", "jpeg"],
                        help="Video codec")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (.mp4, .mkv, .h264, .h265, ...)")
    parser.add_argument("--num-frames", "-n", type=int, default=None,
                        help="Number of frames (omit for infinite)")
    args = parser.parse_args()

    # -- Init --------------------------------------------------------------
    Gst.init(None)
    init_cuda(args.gpu_id)

    frame_duration_ns = 1_000_000_000 // args.fps if args.fps > 0 else 33_333_333

    # -- Generator ---------------------------------------------------------
    gen = NvBufSurfaceGenerator(
        args.format,
        args.width,
        args.height,
        fps_num=args.fps,
        fps_den=1,
        gpu_id=args.gpu_id,
        mem_type=args.mem_type,
        pool_size=args.pool_size,
    )
    print(
        f"Generator created: {args.width}x{args.height} {args.format} "
        f"@ {args.fps} fps (gpu {args.gpu_id}, pool {args.pool_size})"
    )

    # -- Pipeline elements -------------------------------------------------
    codec = "h265" if args.codec == "hevc" else args.codec

    if codec == "jpeg":
        enc_name = "nvjpegenc"
        parse_name = "jpegparse"
        # nvjpegenc requires I420; insert nvvideoconvert when needed
        needs_convert = args.format != "I420"
    else:
        enc_name = f"nvv4l2{codec}enc"
        parse_name = f"{codec}parse"
        needs_convert = args.format not in ("NV12", "I420")

    pipeline = Gst.Pipeline.new("pipeline")

    appsrc = Gst.ElementFactory.make("appsrc", "src")
    enc = Gst.ElementFactory.make(enc_name, "enc")
    parse = Gst.ElementFactory.make(parse_name, "parse")

    assert appsrc and enc and parse, f"Failed to create pipeline elements ({enc_name})"

    # Bridge SavantIdMeta across the encoder (PTS-keyed pad probes)
    bridge_savant_id_meta_py(hash(enc))

    # -- Sink: fakesink or filesink (with optional container muxer) --------
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

    # -- Configure appsrc --------------------------------------------------
    caps = Gst.Caps.from_string(gen.nvmm_caps_str())
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("stream-type", 0)  # GST_APP_STREAM_TYPE_STREAM

    # -- Assemble pipeline -------------------------------------------------
    chain: list[Gst.Element] = [appsrc]
    if needs_convert:
        convert = Gst.ElementFactory.make("nvvideoconvert", "convert")
        assert convert, "Failed to create nvvideoconvert"
        chain.append(convert)
    chain.extend([enc, parse])
    if container_mux:
        chain.append(container_mux)
    chain.append(sink)

    for elem in chain:
        pipeline.add(elem)
    for i in range(len(chain) - 1):
        assert chain[i].link(chain[i + 1]), f"Failed to link {chain[i].get_name()} -> {chain[i+1].get_name()}"

    # -- Print pipeline description ----------------------------------------
    sink_label: str
    if args.output:
        cmux_label = f"{container_mux.get_factory().get_name()} -> " if container_mux else ""
        sink_label = f"{cmux_label}filesink({args.output})"
    else:
        sink_label = "fakesink"

    if needs_convert:
        print(f"Pipeline: appsrc({args.format}) -> nvvideoconvert -> {enc_name} -> {parse_name} -> {sink_label}")
    else:
        print(f"Pipeline: appsrc({args.format}) -> {enc_name} -> {parse_name} -> {sink_label}")

    # -- Start -------------------------------------------------------------
    ret = pipeline.set_state(Gst.State.PLAYING)
    assert ret != Gst.StateChangeReturn.FAILURE, "Failed to start pipeline"

    limit = args.num_frames if args.num_frames is not None else sys.maxsize
    if args.num_frames is not None:
        print(f"Pipeline running ({args.num_frames} frames)...\n")
    else:
        print("Pipeline running (Ctrl-C to stop)...\n")

    # -- Ctrl-C handling ---------------------------------------------------
    running = True

    def _sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    # -- Stats reporter thread ---------------------------------------------
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
    i = 0
    bus = pipeline.get_bus()

    while i < limit and running:
        pts_ns = i * frame_duration_ns
        try:
            gen.push_to_appsrc(appsrc_ptr, pts_ns=pts_ns, duration_ns=frame_duration_ns, id=i)
            with count_lock:
                frame_count += 1
            i += 1
        except Exception as e:
            print(f"Push failed at frame {i}: {e}", file=sys.stderr)
            break

        # Check pipeline bus for errors (non-blocking)
        msg = bus.pop_filtered(Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"Pipeline error: {err}", file=sys.stderr)
            break

    # -- Shutdown ----------------------------------------------------------
    print("\nStopping...")
    running = False

    NvBufSurfaceGenerator.send_eos(appsrc_ptr)

    # Wait for EOS to propagate
    bus.timed_pop_filtered(
        5 * Gst.SECOND,
        Gst.MessageType.EOS | Gst.MessageType.ERROR,
    )

    pipeline.set_state(Gst.State.NULL)
    stats_thread.join(timeout=2)

    with count_lock:
        total = frame_count
    print(f"Total frames pushed: {total}")


if __name__ == "__main__":
    main()
