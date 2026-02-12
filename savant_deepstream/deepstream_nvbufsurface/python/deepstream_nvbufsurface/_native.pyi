"""Type stubs for the deepstream_nvbufsurface._native Rust extension."""

from __future__ import annotations

from typing import Union, final

# ── Enums ────────────────────────────────────────────────────────────────

@final
class Padding:
    """Padding mode for letterboxing.

    - ``NONE`` — scale to fill, may distort aspect ratio.
    - ``RIGHT_BOTTOM`` — image at top-left, padding on right/bottom.
    - ``SYMMETRIC`` — image centered, equal padding on all sides (default).
    """

    NONE: Padding
    RIGHT_BOTTOM: Padding
    SYMMETRIC: Padding
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Interpolation:
    """Interpolation method for scaling.

    - ``NEAREST``  — nearest-neighbor.
    - ``BILINEAR`` — bilinear (default).
    - ``ALGO1``    — GPU: cubic, VIC: 5-tap.
    - ``ALGO2``    — GPU: super, VIC: 10-tap.
    - ``ALGO3``    — GPU: Lanczos, VIC: smart.
    - ``ALGO4``    — GPU: (ignored), VIC: nicest.
    - ``DEFAULT``  — GPU: nearest, VIC: nearest.
    """

    NEAREST: Interpolation
    BILINEAR: Interpolation
    ALGO1: Interpolation
    ALGO2: Interpolation
    ALGO3: Interpolation
    ALGO4: Interpolation
    DEFAULT: Interpolation
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class ComputeMode:
    """Compute backend for transform operations.

    - ``DEFAULT`` — VIC on Jetson, dGPU on x86_64 (default).
    - ``GPU``     — always use GPU compute.
    - ``VIC``     — VIC hardware (Jetson only; raises on dGPU).
    """

    DEFAULT: ComputeMode
    GPU: ComputeMode
    VIC: ComputeMode
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class VideoFormat:
    """Video pixel format.

    - ``RGBA``  — 8-bit RGBA (4 bytes/pixel).
    - ``BGRx``  — 8-bit BGRx (4 bytes/pixel, alpha ignored).
    - ``NV12``  — YUV 4:2:0 semi-planar (default encoder format).
    - ``NV21``  — YUV 4:2:0 semi-planar (UV swapped).
    - ``I420``  — YUV 4:2:0 planar (JPEG encoder format).
    - ``UYVY``  — YUV 4:2:2 packed.
    - ``GRAY8`` — single-channel grayscale.
    """

    RGBA: VideoFormat
    BGRx: VideoFormat
    NV12: VideoFormat
    NV21: VideoFormat
    I420: VideoFormat
    UYVY: VideoFormat
    GRAY8: VideoFormat

    @staticmethod
    def from_name(name: str) -> VideoFormat:
        """Parse a format from a string name.

        Accepted names (case-sensitive): ``RGBA``, ``BGRx``, ``NV12``,
        ``NV21``, ``I420``, ``UYVY``, ``GRAY8``.

        Raises:
            ValueError: If the name is not recognized.
        """
        ...

    def name(self) -> str:
        """Return the canonical name (e.g. ``"NV12"``)."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class MemType:
    """NvBufSurface memory type.

    - ``DEFAULT``       — CUDA Device for dGPU, Surface Array for Jetson.
    - ``CUDA_PINNED``   — CUDA Host (pinned) memory.
    - ``CUDA_DEVICE``   — CUDA Device memory.
    - ``CUDA_UNIFIED``  — CUDA Unified memory.
    - ``SURFACE_ARRAY`` — NVRM Surface Array (Jetson only).
    - ``HANDLE``        — NVRM Handle (Jetson only).
    - ``SYSTEM``        — System memory (malloc).
    """

    DEFAULT: MemType
    CUDA_PINNED: MemType
    CUDA_DEVICE: MemType
    CUDA_UNIFIED: MemType
    SURFACE_ARRAY: MemType
    HANDLE: MemType
    SYSTEM: MemType

    def name(self) -> str:
        """Return the canonical name."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


# ── TransformConfig ──────────────────────────────────────────────────────

class TransformConfig:
    """Configuration for a transform (scale / letterbox) operation.

    All fields have sensible defaults (``Padding.SYMMETRIC``,
    ``Interpolation.BILINEAR``, ``ComputeMode.DEFAULT``, no crop).
    """

    padding: Padding
    interpolation: Interpolation
    src_rect: tuple[int, int, int, int] | None
    compute_mode: ComputeMode

    def __init__(
        self,
        padding: Padding = Padding.SYMMETRIC,
        interpolation: Interpolation = Interpolation.BILINEAR,
        src_rect: tuple[int, int, int, int] | None = None,
        compute_mode: ComputeMode = ComputeMode.DEFAULT,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ── NvBufSurfaceGenerator ────────────────────────────────────────────────

class NvBufSurfaceGenerator:
    """Generates GStreamer buffers with NvBufSurface memory.

    Encapsulates all GStreamer and DeepStream buffer pool interactions.
    """

    def __init__(
        self,
        format: Union[VideoFormat, str],
        width: int,
        height: int,
        fps_num: int = 30,
        fps_den: int = 1,
        gpu_id: int = 0,
        mem_type: Union[MemType, int] | None = None,
        pool_size: int = 4,
    ) -> None: ...
    def nvmm_caps_str(self) -> str:
        """Return the NVMM caps string for configuring an ``appsrc``."""
        ...
    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...
    @property
    def format(self) -> VideoFormat:
        """Video format (e.g. ``VideoFormat.RGBA``, ``VideoFormat.NV12``)."""
        ...
    def acquire_surface(self, id: int | None = None) -> int:
        """Acquire a new NvBufSurface buffer from the pool.

        Returns:
            Raw pointer address of the ``GstBuffer``.
        """
        ...
    def acquire_surface_with_ptr(
        self, id: int | None = None
    ) -> tuple[int, int, int]:
        """Acquire a buffer and return ``(gst_buffer_ptr, data_ptr, pitch)``."""
        ...
    def transform(
        self,
        src_buf_ptr: int,
        config: TransformConfig,
        id: int | None = None,
    ) -> int:
        """Transform (scale + letterbox) a source buffer into a new destination buffer.

        Returns:
            Raw pointer of the destination ``GstBuffer``.
        """
        ...
    def transform_with_ptr(
        self,
        src_buf_ptr: int,
        config: TransformConfig,
        id: int | None = None,
    ) -> tuple[int, int, int]:
        """Like :meth:`transform` but also returns ``(buf_ptr, data_ptr, pitch)``."""
        ...
    def push_to_appsrc(
        self,
        appsrc_ptr: int,
        pts_ns: int,
        duration_ns: int,
        id: int | None = None,
    ) -> None:
        """Acquire a buffer, set timestamps, and push to an ``appsrc``."""
        ...
    @staticmethod
    def send_eos(appsrc_ptr: int) -> None:
        """Send EOS to an ``appsrc``."""
        ...
    def create_surface(
        self, gst_buffer_dest: int, id: int | None = None
    ) -> None:
        """Create a new NvBufSurface and attach it to the given buffer."""
        ...

# ── SkiaContext (only with "skia" feature) ───────────────────────────────

class SkiaContext:
    """GPU-accelerated Skia rendering context backed by CUDA-GL interop.

    Two ways to create:

    - ``SkiaContext(width, height)`` — empty transparent canvas.
    - ``SkiaContext.from_nvbuf(buf_ptr)`` — pre-loaded from an NvBufSurface.
    """

    def __init__(
        self, width: int, height: int, gpu_id: int = 0
    ) -> None: ...
    @staticmethod
    def from_nvbuf(buf_ptr: int, gpu_id: int = 0) -> SkiaContext:
        """Create a context pre-loaded from an NvBufSurface."""
        ...
    @property
    def fbo_id(self) -> int:
        """OpenGL FBO ID backing the canvas."""
        ...
    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        ...
    def render_to_nvbuf(
        self,
        buf_ptr: int,
        config: TransformConfig | None = None,
    ) -> None:
        """Flush Skia and copy the canvas to a destination NvBufSurface."""
        ...

# ── Module-level functions ───────────────────────────────────────────────

def init_cuda(gpu_id: int = 0) -> None:
    """Initialize CUDA context for the given GPU device."""
    ...

def bridge_savant_id_meta_py(element_ptr: int) -> None:
    """Install pad probes to propagate SavantIdMeta across an element."""
    ...

def get_savant_id_meta(buffer_ptr: int) -> list[tuple[str, int]]:
    """Read SavantIdMeta from a GStreamer buffer.

    Returns:
        List of ``(kind, id)`` tuples, e.g. ``[("frame", 42)]``.
    """
    ...

def get_nvbufsurface_info(buffer_ptr: int) -> tuple[int, int, int, int]:
    """Extract NvBufSurface descriptor from a GstBuffer.

    Returns:
        ``(data_ptr, pitch, width, height)`` where ``data_ptr`` is the
        CUDA device pointer, ``pitch`` is the row stride in bytes, and
        ``width``/``height`` are in pixels.
    """
    ...
