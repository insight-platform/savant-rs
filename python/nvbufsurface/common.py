"""Shared helpers for the nvbufsurface encoding pipeline examples.

Provides reusable CLI argument definitions, encoder property builders,
statistics tracking, and the Picasso-based encode-mux-drain lifecycle via
:class:`PicassoSession`.
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

from savant_rs.deepstream import (  # noqa: E402
    DsNvBufSurfaceGstBuffer,
    NvBufSurfaceGenerator,
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

SOURCE_ID = "src-0"


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
# Picasso session — wraps engine + generator + muxer + stats + signal handling
# ---------------------------------------------------------------------------


class PicassoSession:
    """Manages Picasso engine lifecycle, NvBufSurface buffer generation,
    optional MP4 muxer, live stats, and graceful Ctrl-C shutdown.

    Frames are submitted via :meth:`submit`; the engine handles transform,
    rendering, and encoding asynchronously.  Encoded output is pushed to
    an optional MP4 muxer via the ``on_encoded_frame`` callback.

    Optional *on_gpumat* / *on_render* callbacks enable custom GPU drawing
    inside the Picasso worker thread:

    - ``on_gpumat(source_id, frame, data_ptr, pitch, width, height)`` —
      raw CUDA pointer for OpenCV CUDA drawing.
    - ``on_render(source_id, fbo_id, width, height, frame)`` —
      OpenGL FBO for Skia / GL drawing.

    The corresponding ``use_on_gpumat`` / ``use_on_render`` flags on the
    ``SourceSpec`` are set automatically when a callback is provided.

    Usage with NvBufSurface buffers::

        session = PicassoSession(args, video_format=VideoFormat.RGBA,
                                 on_gpumat=my_draw_callback)

        while session.is_running and i < session.limit:
            view = session.acquire_surface_view(frame_id=i)
            session.submit(view, pts_ns=pts_ns,
                           duration_ns=session.frame_duration_ns)
            i += 1

        session.shutdown()

    Usage with ``cv2.cuda.GpuMat`` via ``__cuda_array_interface__``::

        from savant_rs.deepstream import GpuMatCudaArray, make_gpu_mat

        session = PicassoSession(args, video_format=VideoFormat.RGBA,
                                 use_generator=False)

        while session.is_running and i < session.limit:
            mat = make_gpu_mat(args.width, args.height)
            mat.setTo((0, 0, 0, 255))  # draw content
            session.submit(GpuMatCudaArray(mat), pts_ns=...,
                           duration_ns=session.frame_duration_ns)
            i += 1

        session.shutdown()
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
    ) -> None:
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
            f"@ {self._fps} fps, codec={self._codec!r} (gpu {args.gpu_id})"
        )
        print(f"Encoder properties: {enc_props}")

        # -- NvBufSurface generator for buffer acquisition ---------------------
        self._generator: NvBufSurfaceGenerator | None = None
        if use_generator:
            self._generator = NvBufSurfaceGenerator(
                video_format, self._width, self._height, self._fps, 1, args.gpu_id
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
        self._eos_event = threading.Event()
        self._gpu_id = args.gpu_id

        # -- Optional MP4 muxer (direct use from callback) ---------------------
        self._muxer: Mp4Muxer | None = None
        if args.output:
            self._muxer = Mp4Muxer(self._codec, args.output, fps_num=self._fps)
        else:
            print("No output file — encoded frames will be discarded (benchmark mode)")

        # -- Picasso callbacks -------------------------------------------------
        def _on_encoded(output) -> None:
            try:
                if output.is_eos:
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
        self._engine = PicassoEngine(GeneralSpec(idle_timeout_secs=300), callbacks)
        source_spec = SourceSpec(
            codec=CodecSpec.encode(TransformConfig(), enc_cfg),
            draw=draw,
            font_family="sans-serif",
            use_on_gpumat=on_gpumat is not None,
            use_on_render=on_render is not None,
        )
        self._engine.set_source_spec(SOURCE_ID, source_spec)

        # -- Run banner --------------------------------------------------------
        if args.num_frames is not None:
            print(f"Running ({args.num_frames} frames)...\n")
        else:
            print("Running (Ctrl-C to stop)...\n")

        # -- Stats reporter thread ---------------------------------------------
        self._stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        self._stats_thread.start()

    # -- Public API -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``False`` after Ctrl-C or an encoder callback error."""
        return self._running

    def acquire_surface(self, *, frame_id: int) -> DsNvBufSurfaceGstBuffer:
        """Acquire an NvBufSurface GPU buffer from the internal pool.

        Returns a raw ``DsNvBufSurfaceGstBuffer`` guard.  Use this when you need to
        perform GPU operations on the buffer (e.g. ``nvgstbuf_as_gpu_mat``,
        ``nvbuf_as_gpu_mat``) before submitting it to the engine.

        Requires ``use_generator=True`` (the default).
        """
        if self._generator is None:
            raise RuntimeError(
                "NvBufSurfaceGenerator was not created (use_generator=False)"
            )
        return self._generator.acquire_surface(id=frame_id)

    def acquire_surface_view(self, *, frame_id: int) -> SurfaceView:
        """Acquire an NvBufSurface GPU buffer wrapped in a ``SurfaceView``.

        The returned ``SurfaceView`` caches surface parameters (data
        pointer, pitch, width, height, GPU ID) and can be passed directly
        to :meth:`send_frame` or :meth:`submit`.

        Requires ``use_generator=True`` (the default).
        """
        buf = self.acquire_surface(frame_id=frame_id)
        return SurfaceView.from_buffer(buf, 0)

    def make_frame(
        self,
        *,
        pts_ns: int,
        duration_ns: int | None = None,
    ) -> VideoFrame:
        """Create a :class:`VideoFrame` with the session's parameters.

        Use this to add :class:`VideoObject` instances before sending
        via :meth:`send_frame`.
        """
        frame = VideoFrame(
            source_id=SOURCE_ID,
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
        self, frame: VideoFrame, buf: SurfaceView | GstBuffer | int | Any
    ) -> None:
        """Submit a pre-built :class:`VideoFrame` to the Picasso engine.

        *buf* may be a ``SurfaceView``, a ``DsNvBufSurfaceGstBuffer`` guard,
        a raw ``GstBuffer*`` pointer as ``int``, or any object exposing
        ``__cuda_array_interface__`` (e.g. ``savant_rs.deepstream.GpuMatCudaArray``).
        """
        self._engine.send_frame(SOURCE_ID, frame, buf)
        with self._lock:
            self._frame_count += 1

    def submit(
        self,
        buf: SurfaceView | DsNvBufSurfaceGstBuffer | int | Any,
        *,
        pts_ns: int,
        duration_ns: int | None = None,
    ) -> None:
        """Shorthand: create a :class:`VideoFrame` and submit it."""
        frame = self.make_frame(pts_ns=pts_ns, duration_ns=duration_ns)
        self.send_frame(frame, buf)

    def shutdown(self) -> None:
        """Send EOS, wait for encoder drain, finalise muxer, print totals."""
        print("\nStopping...")
        self._running = False

        self._engine.send_eos(SOURCE_ID)
        self._eos_event.wait(timeout=10)
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
            print(
                f"submitted: {count:>8}  |  encoded: {enc:>8}  |  "
                f"fps: {fps:>8.1f}  |  bitstream: {ebytes // 1024} KB  |  "
                f"RSS: {rss // 1024} MB  |  GPU: {gpu_str}"
            )
            last_count = count
            last_time = now
