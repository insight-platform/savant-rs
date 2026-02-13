"""Shared helpers for the nvbufsurface encoding pipeline examples.

Provides reusable CLI argument definitions, encoder property builders,
statistics tracking, and the encode-pull-mux-drain lifecycle via
:class:`EncodingSession`.
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
from gi.repository import Gst  # noqa: E402

from deepstream_nvbufsurface import init_cuda  # noqa: E402
from deepstream_nvbufsurface import VideoFormat  # noqa: E402
from savant_gstreamer import Codec  # noqa: E402
from deepstream_encoders import (  # noqa: E402
    NvEncoder,
    EncoderConfig,
    H264DgpuProps,
    HevcDgpuProps,
    Av1DgpuProps,
    JpegProps,
    RateControl,
    H264Profile,
    HevcProfile,
    DgpuPreset,
    TuningPreset,
)
from savant_gstreamer import Mp4Muxer  # noqa: E402


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def init_gst_and_cuda(gpu_id: int = 0) -> None:
    """Idempotent GStreamer + CUDA initialisation."""
    Gst.init(None)
    init_cuda(gpu_id)


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
    """Map CLI codec name to :class:`Codec` enum."""
    return Codec.from_name("hevc" if name == "h265" else name)


def build_encoder_properties(
    codec: Codec,
    *,
    bitrate: int | None = None,
    quality: int | None = None,
):
    """Build sensible predefined encoder properties for a given codec.

    - H264/HEVC/AV1: Main/High profile, VBR, P4 preset, low-latency tuning.
      ``bitrate`` overrides the default (4 Mbps).
    - JPEG: ``quality`` overrides the default (85).
    """
    default_bitrate = bitrate or 4_000_000  # 4 Mbps

    if codec == Codec.H264:
        return H264DgpuProps(
            profile=H264Profile.MAIN,
            control_rate=RateControl.VBR,
            bitrate=default_bitrate,
            preset=DgpuPreset.P1,
            tuning_info=TuningPreset.LOW_LATENCY,
            iframeinterval=30,
        )
    elif codec == Codec.HEVC:
        return HevcDgpuProps(
            profile=HevcProfile.MAIN,
            control_rate=RateControl.VBR,
            bitrate=default_bitrate,
            preset=DgpuPreset.P1,
            tuning_info=TuningPreset.LOW_LATENCY,
            iframeinterval=30,
        )
    elif codec == Codec.AV1:
        return Av1DgpuProps(
            control_rate=RateControl.VBR,
            bitrate=default_bitrate,
            preset=DgpuPreset.P1,
            tuning_info=TuningPreset.LOW_LATENCY,
            iframeinterval=30,
        )
    elif codec == Codec.JPEG:
        return JpegProps(quality=quality or 85)
    else:
        return None


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the CLI arguments shared by all pipeline examples."""
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--codec",
        type=str,
        default="h265",
        choices=["h264", "h265", "hevc", "jpeg", "av1"],
        help="Video codec",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=None,
        help="Bitrate in bps for H264/HEVC/AV1 (default: 4000000)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="JPEG quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output MP4 file path",
    )
    parser.add_argument(
        "--num-frames",
        "-n",
        type=int,
        default=None,
        help="Number of frames (omit for infinite)",
    )


# ---------------------------------------------------------------------------
# Encoding session — wraps encoder + muxer + stats + signal handling
# ---------------------------------------------------------------------------


class EncodingSession:
    """Manages encoder lifecycle, optional MP4 muxer, live stats, and
    graceful Ctrl-C shutdown.

    Usage::

        session = EncodingSession(args, video_format=VideoFormat.RGBA)

        while session.is_running and i < session.limit:
            buf = session.encoder.acquire_surface(id=i)
            # ... render into buf ...
            session.submit(buf, frame_id=i, pts_ns=pts_ns, duration_ns=dur)
            session.pull_encoded()
            session.check_error()
            i += 1

        session.shutdown()
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        video_format: VideoFormat,
    ) -> None:
        # -- GStreamer + CUDA init (idempotent) --------------------------------
        init_gst_and_cuda(args.gpu_id)

        self.codec = resolve_codec(args.codec)
        self.frame_duration_ns = (
            1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
        )
        self.limit = args.num_frames if args.num_frames is not None else sys.maxsize
        self._output_path = args.output

        # -- Encoder -----------------------------------------------------------
        enc_props = build_encoder_properties(
            self.codec, bitrate=args.bitrate, quality=args.quality
        )
        config = EncoderConfig(
            self.codec,
            args.width,
            args.height,
            format=video_format,
            fps_num=args.fps,
            fps_den=1,
            gpu_id=args.gpu_id,
            mem_type=getattr(args, "mem_type", 0),
            properties=enc_props,
        )
        self.encoder = NvEncoder(config)
        print(
            f"Encoder created: {args.width}x{args.height} {video_format.name()} "
            f"@ {args.fps} fps, codec={self.codec.name()} (gpu {args.gpu_id})"
        )
        print(f"Encoder properties: {enc_props}")

        # -- Optional MP4 muxer -----------------------------------------------
        self.muxer: Mp4Muxer | None = None
        if args.output:
            self.muxer = Mp4Muxer(self.codec, args.output, fps_num=args.fps)
        else:
            print("No output file — encoded frames will be discarded (benchmark mode)")

        # -- Run banner --------------------------------------------------------
        if args.num_frames is not None:
            print(f"Running ({args.num_frames} frames)...\n")
        else:
            print("Running (Ctrl-C to stop)...\n")

        # -- Ctrl-C handler ----------------------------------------------------
        self._running = True

        def _sigint(_signum, _frame):
            self._running = False

        signal.signal(signal.SIGINT, _sigint)

        # -- Stats counters ----------------------------------------------------
        self._frame_count = 0
        self._encoded_count = 0
        self._encoded_bytes = 0
        self._lock = threading.Lock()

        self._stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        self._stats_thread.start()

    # -- Public API -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``False`` after Ctrl-C."""
        return self._running

    def submit(
        self,
        buffer_ptr: int,
        *,
        frame_id: int,
        pts_ns: int,
        duration_ns: int | None = None,
    ) -> None:
        """Submit a filled buffer to the encoder and update the submit counter."""
        self.encoder.submit_frame(
            buffer_ptr,
            frame_id=frame_id,
            pts_ns=pts_ns,
            duration_ns=duration_ns,
        )
        with self._lock:
            self._frame_count += 1

    def pull_encoded(self) -> None:
        """Non-blocking drain of all ready encoded frames into the muxer."""
        while True:
            frame = self.encoder.pull_encoded()
            if frame is None:
                break
            with self._lock:
                self._encoded_count += 1
                self._encoded_bytes += frame.size
            if self.muxer is not None:
                muxer = self.muxer
                muxer.push(frame.data, frame.pts_ns, frame.dts_ns, frame.duration_ns)

    def check_error(self) -> None:
        """Relay :meth:`NvEncoder.check_error`, printing to stderr on failure."""
        try:
            self.encoder.check_error()
        except Exception as e:
            print(f"Encoder error: {e}", file=sys.stderr)
            self._running = False

    def shutdown(self) -> None:
        """Send EOS, drain remaining frames, finalise muxer, print totals."""
        print("\nStopping...")
        self._running = False

        remaining = self.encoder.finish()
        for frame in remaining:
            with self._lock:
                self._encoded_count += 1
                self._encoded_bytes += frame.size
            if self.muxer is not None:
                self.muxer.push(
                    frame.data, frame.pts_ns, frame.dts_ns, frame.duration_ns
                )

        if self.muxer is not None:
            self.muxer.finish()
            print(f"Output written to: {self._output_path}")

        self._stats_thread.join(timeout=2)

        with self._lock:
            total_sub = self._frame_count
            total_enc = self._encoded_count
            total_bytes = self._encoded_bytes
        print(
            f"Total submitted: {total_sub}  |  "
            f"Total encoded: {total_enc}  |  "
            f"Bitstream: {total_bytes // 1024} KB"
        )

    # -- Internal -------------------------------------------------------------

    def _stats_reporter(self) -> None:
        last_count = 0
        last_time = time.monotonic()
        while self._running:
            time.sleep(1.0)
            now = time.monotonic()
            with self._lock:
                count = self._frame_count
                enc = self._encoded_count
                ebytes = self._encoded_bytes
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
