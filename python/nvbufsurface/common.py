"""Shared helpers for the nvbufsurface encoding pipeline examples.

Provides reusable CLI argument definitions, encoder property builders,
statistics tracking, and the Picasso-based encode-mux-drain lifecycle via
:class:`PicassoSession`.

All pipeline scripts refuse to run against a **debug** build of savant_rs by
default, because benchmark results would be meaningless.  Pass
``--allow-debug-build`` to override this check when only functional
correctness (not throughput) matters.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Any, Callable

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst  # noqa: E402

import savant_rs  # noqa: E402
from savant_rs.deepstream import (  # noqa: E402
    BufferGenerator,
    SharedBuffer,
    SurfaceView,
    TransformConfig,
    VideoFormat,
    gpu_mem_used_mib,
    init_cuda,
)
from savant_rs.gstreamer import Codec, Mp4Muxer  # noqa: E402
from savant_rs.picasso import (  # noqa: E402
    Callbacks,
    CodecSpec,
    EncoderConfig,
    EncoderProperties,
    GeneralSpec,
    PicassoEngine,
    SourceSpec,
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
from savant_rs.primitives import VideoFrame, VideoFrameContent  # noqa: E402


def _source_id(idx: int) -> str:
    return f"src-{idx}"


# ---------------------------------------------------------------------------
# Release-build guard
# ---------------------------------------------------------------------------


def check_release_build(args: argparse.Namespace) -> None:
    """Exit with an error when running against a debug build of savant_rs.

    These scripts measure real-world encoding throughput; a debug build of
    the Rust library is orders of magnitude slower than a release build, so
    benchmark numbers would be meaningless.

    Pass ``--allow-debug-build`` to override this check (e.g. for functional
    smoke-tests where absolute performance does not matter).
    """
    if not savant_rs.is_release_build() and not getattr(args, "allow_debug_build", False):
        print(
            "ERROR: savant_rs was compiled in DEBUG mode.\n"
            "  Performance benchmarks against a debug build are meaningless —\n"
            "  the Rust library can be 10–100x slower than a release build.\n"
            "\n"
            "  Re-build with  `cargo build --release`  (or install a release wheel),\n"
            "  or pass  --allow-debug-build  if you only need a functional smoke-test.",
            file=sys.stderr,
        )
        sys.exit(1)


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


def gpu_mem_mib(gpu_id: int = 0) -> int | None:
    """Return GPU memory used in MiB, or None if unavailable."""
    try:
        return gpu_mem_used_mib(gpu_id)
    except Exception:
        return None


def resolve_codec(name: str) -> Codec:
    """Map CLI codec name to :class:`Codec` enum."""
    return Codec.from_name("hevc" if name == "h265" else name)


def build_encoder_properties(
    codec: Codec,
    *,
    bitrate: int | None = None,
    quality: int | None = None,
) -> EncoderProperties | None:
    """Build encoder properties wrapped in :class:`EncoderProperties`.

    - H264/HEVC/AV1: Main/High profile, VBR, P1 preset, low-latency tuning.
      ``bitrate`` overrides the default (4 Mbps).
    - JPEG: ``quality`` overrides the default (85).
    """
    default_bitrate = bitrate or 4_000_000  # 4 Mbps

    if codec == Codec.H264:
        return EncoderProperties.h264_dgpu(
            H264DgpuProps(
                profile=H264Profile.MAIN,
                control_rate=RateControl.VARIABLE_BITRATE,
                bitrate=default_bitrate,
                preset=DgpuPreset.P1,
                tuning_info=TuningPreset.LOW_LATENCY,
                iframeinterval=30,
            )
        )
    elif codec == Codec.HEVC:
        return EncoderProperties.hevc_dgpu(
            HevcDgpuProps(
                profile=HevcProfile.MAIN,
                control_rate=RateControl.VARIABLE_BITRATE,
                bitrate=default_bitrate,
                preset=DgpuPreset.P1,
                tuning_info=TuningPreset.LOW_LATENCY,
                iframeinterval=30,
            )
        )
    elif codec == Codec.AV1:
        return EncoderProperties.av1_dgpu(
            Av1DgpuProps(
                control_rate=RateControl.VARIABLE_BITRATE,
                bitrate=default_bitrate,
                preset=DgpuPreset.P1,
                tuning_info=TuningPreset.LOW_LATENCY,
                iframeinterval=30,
            )
        )
    elif codec == Codec.JPEG:
        return EncoderProperties.jpeg(JpegProps(quality=quality or 85))
    else:
        return None


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the CLI arguments shared by all pipeline examples."""
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Framerate numerator")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--codec",
        type=str,
        default="jpeg",
        choices=["h264", "h265", "hevc", "jpeg", "av1"],
        help="Video codec (default: jpeg; h264/h265/av1 require NVENC)",
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
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel sources (incompatible with --output)",
    )
    parser.add_argument(
        "--allow-debug-build",
        action="store_true",
        default=False,
        help=(
            "Allow running against a debug build of savant_rs. "
            "Use only for functional smoke-tests — benchmark numbers will be unreliable."
        ),
    )


# ---------------------------------------------------------------------------
# Picasso session — wraps engine + generator + muxer + stats + signal handling
# ---------------------------------------------------------------------------


class PicassoSession:
    """Manages Picasso engine lifecycle, NvBufSurface buffer generation,
    optional MP4 muxer, live stats, and graceful Ctrl-C shutdown.

    Supports ``--jobs=N`` parallel sources: each source has its own
    NvBufSurface generator and Picasso source spec, feeding frames
    concurrently to a single shared :class:`PicassoEngine`.

    Frames are submitted via :meth:`submit`; the engine handles transform,
    rendering, and encoding asynchronously.  Encoded output is pushed to
    an optional MP4 muxer via the ``on_encoded_frame`` callback.

    Optional *on_gpumat* / *on_render* callbacks enable custom GPU drawing
    inside the Picasso worker thread:

    - ``on_gpumat(source_id, frame, data_ptr, pitch, width, height, cuda_stream)`` —
      raw CUDA pointer for OpenCV CUDA drawing.
    - ``on_render(source_id, fbo_id, width, height, frame)`` —
      OpenGL FBO for Skia / GL drawing.

    The corresponding ``use_on_gpumat`` / ``use_on_render`` flags on the
    ``SourceSpec`` are set automatically when a callback is provided.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        video_format: VideoFormat,
        on_gpumat: Callable[..., Any] | None = None,
        on_render: Callable[..., Any] | None = None,
        draw: object | None = None,
        use_generator: bool = True,
        name: str = "picasso",
    ) -> None:
        check_release_build(args)

        self.jobs: int = getattr(args, "jobs", 1)
        if args.output and self.jobs > 1:
            raise SystemExit("--output is incompatible with --jobs > 1")

        # -- GStreamer + CUDA init (idempotent) --------------------------------
        init_gst_and_cuda(args.gpu_id)

        self._codec = resolve_codec(args.codec)
        self._fps = args.fps
        self._width = args.width
        self._height = args.height
        self.frame_duration_ns = (
            1_000_000_000 // args.fps if args.fps > 0 else 33_333_333
        )
        self.limit = args.num_frames if args.num_frames is not None else sys.maxsize
        self._output_path = args.output

        self.source_ids: list[str] = [_source_id(i) for i in range(self.jobs)]

        # -- Encoder configuration (Picasso) -----------------------------------
        enc_props = build_encoder_properties(
            self._codec, bitrate=args.bitrate, quality=args.quality
        )
        enc_cfg = EncoderConfig(self._codec, self._width, self._height)
        enc_cfg.format(video_format)
        enc_cfg.fps(self._fps, 1)
        enc_cfg.gpu_id(args.gpu_id)
        if enc_props is not None:
            enc_cfg.properties(enc_props)

        print(
            f"Encoder config: {self._width}x{self._height} {video_format!r} "
            f"@ {self._fps} fps, codec={self._codec!r} (gpu {args.gpu_id}), "
            f"sources={self.jobs}"
        )
        print(f"Encoder properties: {enc_props}")

        # -- NvBufSurface generators (one per source) --------------------------
        self._generators: list[BufferGenerator] = []
        if use_generator:
            for _ in range(self.jobs):
                self._generators.append(
                    BufferGenerator(
                        video_format,
                        self._width,
                        self._height,
                        self._fps,
                        1,
                        args.gpu_id,
                    )
                )

        # -- Ctrl-C handler ----------------------------------------------------
        self._running = True

        def _sigint(_signum, _frame):
            self._running = False

        signal.signal(signal.SIGINT, _sigint)

        # -- Stats counters ----------------------------------------------------
        self._lock = threading.Lock()
        self._frame_count = 0
        self._encoded_count = 0
        self._encoded_bytes = 0
        self._eos_remaining = self.jobs
        self._eos_event = threading.Event()
        self._gpu_id = args.gpu_id

        # -- Optional MP4 muxer (single-source only) ---------------------------
        self._muxer: Mp4Muxer | None = None
        if args.output:
            self._muxer = Mp4Muxer(self._codec, args.output, fps_num=self._fps)
        else:
            print("No output file — encoded frames will be discarded (benchmark mode)")

        # -- Picasso callbacks -------------------------------------------------
        def _on_encoded(output) -> None:
            try:
                if output.is_eos:
                    with self._lock:
                        self._eos_remaining -= 1
                        if self._eos_remaining <= 0:
                            self._eos_event.set()
                    return
                if output.is_video_frame:
                    vf = output.as_video_frame()
                    if vf.content.is_internal():
                        data = vf.content.get_data()
                        with self._lock:
                            self._encoded_count += 1
                            self._encoded_bytes += len(data)
                        if self._muxer is not None:
                            self._muxer.push(
                                data,
                                vf.pts,
                                vf.dts if vf.dts is not None else vf.pts,
                                vf.duration
                                if vf.duration is not None
                                else self.frame_duration_ns,
                            )
            except Exception as e:
                print(f"Encoder callback error: {e}", file=sys.stderr)
                self._running = False

        callbacks = Callbacks(
            on_encoded_frame=_on_encoded,
            on_gpumat=on_gpumat,
            on_render=on_render,
        )

        # -- Picasso engine ----------------------------------------------------
        self._engine = PicassoEngine(
            GeneralSpec(name=name, idle_timeout_secs=300), callbacks
        )
        source_spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc_cfg),
            draw=draw,
            font_family="sans-serif",
            use_on_gpumat=on_gpumat is not None,
            use_on_render=on_render is not None,
        )
        for sid in self.source_ids:
            self._engine.set_source_spec(sid, source_spec)

        # -- Run banner --------------------------------------------------------
        jobs_str = f", {self.jobs} sources" if self.jobs > 1 else ""
        if args.num_frames is not None:
            print(f"Running ({args.num_frames} frames{jobs_str})...\n")
        else:
            print(f"Running (Ctrl-C to stop{jobs_str})...\n")

        # -- Stats reporter thread ---------------------------------------------
        self._stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        self._stats_thread.start()

    # -- Public API -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``False`` after Ctrl-C or an encoder callback error."""
        return self._running

    def acquire_surface(
        self, *, source_idx: int = 0, frame_id: int
    ) -> SharedBuffer:
        """Acquire an NvBufSurface GPU buffer from the pool of *source_idx*.

        Requires ``use_generator=True`` (the default).
        """
        if not self._generators:
            raise RuntimeError(
                "BufferGenerator was not created (use_generator=False)"
            )
        return self._generators[source_idx].acquire(id=frame_id)

    def acquire_surface_view(
        self, *, source_idx: int = 0, frame_id: int
    ) -> SurfaceView:
        """Acquire an NvBufSurface GPU buffer wrapped in a ``SurfaceView``."""
        buf = self.acquire_surface(source_idx=source_idx, frame_id=frame_id)
        return SurfaceView.from_buffer(buf, 0)

    def make_frame(
        self,
        *,
        source_idx: int = 0,
        pts_ns: int,
        duration_ns: int | None = None,
    ) -> VideoFrame:
        """Create a :class:`VideoFrame` for source *source_idx*."""
        frame = VideoFrame(
            source_id=self.source_ids[source_idx],
            framerate=f"{self._fps}/1",
            width=self._width,
            height=self._height,
            content=VideoFrameContent.none(),
            time_base=(1, 1_000_000_000),
            pts=pts_ns,
        )
        if duration_ns is not None:
            frame.duration = duration_ns
        return frame

    def send_frame(
        self,
        frame: VideoFrame,
        buf: SurfaceView | SharedBuffer | int | Any,
        *,
        source_idx: int = 0,
    ) -> None:
        """Submit a pre-built :class:`VideoFrame` to source *source_idx*."""
        self._engine.send_frame(self.source_ids[source_idx], frame, buf)
        with self._lock:
            self._frame_count += 1

    def submit(
        self,
        buf: SurfaceView | SharedBuffer | int | Any,
        *,
        source_idx: int = 0,
        pts_ns: int,
        duration_ns: int | None = None,
    ) -> None:
        """Shorthand: create a :class:`VideoFrame` and submit it."""
        frame = self.make_frame(
            source_idx=source_idx, pts_ns=pts_ns, duration_ns=duration_ns
        )
        self.send_frame(frame, buf, source_idx=source_idx)

    def shutdown(self) -> None:
        """Send EOS to all sources, wait for drain, finalise muxer."""
        print("\nStopping...")
        self._running = False

        for sid in self.source_ids:
            self._engine.send_eos(sid)
        self._eos_event.wait(timeout=10 * self.jobs)
        self._engine.shutdown()

        if self._muxer is not None:
            self._muxer.finish()
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
        multi = self.jobs > 1
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
            gpu_mib = gpu_mem_mib(self._gpu_id)
            gpu_str = f"{gpu_mib} MiB" if gpu_mib is not None else "N/A"
            if multi:
                frame_idx = count // self.jobs if self.jobs else count
                print(
                    f"frame {frame_idx:>8} | src {self.jobs} | "
                    f"enc {enc:>8} | {fps:>7.1f} fps | "
                    f"bitstream {ebytes // 1024:>8} KB | "
                    f"RSS {rss // 1024:>5} MB | GPU {gpu_str}"
                )
            else:
                print(
                    f"submitted: {count:>8}  |  encoded: {enc:>8}  |  "
                    f"fps: {fps:>8.1f}  |  bitstream: {ebytes // 1024} KB  |  "
                    f"RSS: {rss // 1024} MB  |  GPU: {gpu_str}"
                )
            last_count = count
            last_time = now
