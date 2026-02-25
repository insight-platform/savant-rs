"""Type stubs for ``savant_rs.deepstream`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, final

__all__ = [
    "Padding",
    "Interpolation",
    "ComputeMode",
    "VideoFormat",
    "MemType",
    "TransformConfig",
    "NvBufSurfaceGenerator",
    "SkiaContext",
    "init_cuda",
    "bridge_savant_id_meta",
    "get_savant_id_meta",
    "get_nvbufsurface_info",
    "set_buffer_pts",
    "set_buffer_duration",
]

# ── Enums ────────────────────────────────────────────────────────────────

@final
class Padding:
    """Padding mode for letterboxing.

    - ``NONE`` -- scale to fill, may distort aspect ratio.
    - ``RIGHT_BOTTOM`` -- image at top-left, padding on right/bottom.
    - ``SYMMETRIC`` -- image centered, equal padding on all sides (default).
    """

    NONE: Padding
    RIGHT_BOTTOM: Padding
    SYMMETRIC: Padding

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...

@final
class Interpolation:
    """Interpolation method for scaling.

    - ``NEAREST``  -- nearest-neighbor.
    - ``BILINEAR`` -- bilinear (default).
    - ``ALGO1``    -- GPU: cubic, VIC: 5-tap.
    - ``ALGO2``    -- GPU: super, VIC: 10-tap.
    - ``ALGO3``    -- GPU: Lanczos, VIC: smart.
    - ``ALGO4``    -- GPU: (ignored), VIC: nicest.
    - ``DEFAULT``  -- GPU: nearest, VIC: nearest.
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

@final
class ComputeMode:
    """Compute backend for transform operations.

    - ``DEFAULT`` -- VIC on Jetson, dGPU on x86_64 (default).
    - ``GPU``     -- always use GPU compute.
    - ``VIC``     -- VIC hardware (Jetson only, raises error on dGPU).
    """

    DEFAULT: ComputeMode
    GPU: ComputeMode
    VIC: ComputeMode

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...

@final
class VideoFormat:
    """Video pixel format.

    - ``RGBA``  -- 8-bit RGBA (4 bytes/pixel).
    - ``BGRx``  -- 8-bit BGRx (4 bytes/pixel, alpha ignored).
    - ``NV12``  -- YUV 4:2:0 semi-planar (default encoder format).
    - ``NV21``  -- YUV 4:2:0 semi-planar (UV swapped).
    - ``I420``  -- YUV 4:2:0 planar (JPEG encoder format).
    - ``UYVY``  -- YUV 4:2:2 packed.
    - ``GRAY8`` -- single-channel grayscale.
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
        """Parse a video format from a string name.

        Args:
            name: Format name (e.g. ``"NV12"``).

        Returns:
            The parsed video format.

        Raises:
            ValueError: If the name is not recognized.
        """
        ...

    def name(self) -> str:
        """Return the canonical name of this format (e.g. ``"NV12"``)."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class MemType:
    """NvBufSurface memory type.

    - ``DEFAULT``       -- CUDA Device for dGPU, Surface Array for Jetson.
    - ``CUDA_PINNED``   -- CUDA Host (pinned) memory.
    - ``CUDA_DEVICE``   -- CUDA Device memory.
    - ``CUDA_UNIFIED``  -- CUDA Unified memory.
    - ``SURFACE_ARRAY`` -- NVRM Surface Array (Jetson only).
    - ``HANDLE``        -- NVRM Handle (Jetson only).
    - ``SYSTEM``        -- System memory (malloc).
    """

    DEFAULT: MemType
    CUDA_PINNED: MemType
    CUDA_DEVICE: MemType
    CUDA_UNIFIED: MemType
    SURFACE_ARRAY: MemType
    HANDLE: MemType
    SYSTEM: MemType

    def name(self) -> str:
        """Return the canonical name of this memory type."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

# ── TransformConfig ─────────────────────────────────────────────────────

class TransformConfig:
    """Configuration for a transform (scale / letterbox) operation.

    All fields have sensible defaults (``Padding.SYMMETRIC``,
    ``Interpolation.BILINEAR``, ``ComputeMode.DEFAULT``, no crop).
    """

    padding: Padding
    interpolation: Interpolation
    src_rect: Optional[Tuple[int, int, int, int]]
    compute_mode: ComputeMode

    def __init__(
        self,
        padding: Padding = ...,
        interpolation: Interpolation = ...,
        src_rect: Optional[Tuple[int, int, int, int]] = None,
        compute_mode: ComputeMode = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ── NvBufSurfaceGenerator ───────────────────────────────────────────────

class NvBufSurfaceGenerator:
    """GPU buffer pool for allocating NVMM surfaces.

    Args:
        format: Video format (``VideoFormat`` enum or string name).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps_num: Framerate numerator (default 30).
        fps_den: Framerate denominator (default 1).
        gpu_id: GPU device ID (default 0).
        mem_type: Memory type (default ``MemType.DEFAULT``).
        pool_size: Buffer pool size (default 4).
    """

    def __init__(
        self,
        format: Union[VideoFormat, str],
        width: int,
        height: int,
        fps_num: int = 30,
        fps_den: int = 1,
        gpu_id: int = 0,
        mem_type: Union[MemType, int, None] = None,
        pool_size: int = 4,
    ) -> None: ...
    def nvmm_caps_str(self) -> str:
        """Return the NVMM caps string for configuring an ``appsrc``."""
        ...

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def format(self) -> VideoFormat: ...
    def acquire_surface(self, id: Optional[int] = None) -> int:
        """Acquire a new NvBufSurface buffer from the pool.

        Returns:
            Raw pointer address of the GstBuffer.
        """
        ...

    def acquire_surface_with_params(
        self,
        pts_ns: int,
        duration_ns: int,
        id: Optional[int] = None,
    ) -> int:
        """Acquire a buffer and stamp PTS and duration on it.

        Args:
            pts_ns: Presentation timestamp in nanoseconds.
            duration_ns: Frame duration in nanoseconds.
            id: Optional buffer ID / frame index.

        Returns:
            Raw pointer address of the GstBuffer.
        """
        ...

    def acquire_surface_with_ptr(
        self,
        id: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Acquire a buffer and return ``(gst_buffer_ptr, data_ptr, pitch)``."""
        ...

    def transform(
        self,
        src_buf_ptr: int,
        config: TransformConfig,
        id: Optional[int] = None,
    ) -> int:
        """Transform (scale + letterbox) a source buffer into a new destination.

        Returns:
            Raw pointer address of the destination GstBuffer.
        """
        ...

    def transform_with_ptr(
        self,
        src_buf_ptr: int,
        config: TransformConfig,
        id: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Like :meth:`transform` but also returns ``(buf_ptr, data_ptr, pitch)``."""
        ...

    def push_to_appsrc(
        self,
        appsrc_ptr: int,
        pts_ns: int,
        duration_ns: int,
        id: Optional[int] = None,
    ) -> None:
        """Push a new NVMM buffer to an AppSrc element."""
        ...

    @staticmethod
    def send_eos(appsrc_ptr: int) -> None:
        """Send an end-of-stream signal to an AppSrc element."""
        ...

    def create_surface(
        self,
        gst_buffer_dest: int,
        id: Optional[int] = None,
    ) -> None:
        """Create a new NvBufSurface and attach it to the given buffer."""
        ...

# ── SkiaContext ──────────────────────────────────────────────────────────

class SkiaContext:
    """GPU-accelerated Skia rendering context backed by CUDA-GL interop."""

    def __init__(self, width: int, height: int, gpu_id: int = 0) -> None: ...
    @staticmethod
    def from_nvbuf(buf_ptr: int, gpu_id: int = 0) -> SkiaContext:
        """Create a SkiaContext from an existing NvBufSurface buffer."""
        ...

    @property
    def fbo_id(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def render_to_nvbuf(
        self,
        buf_ptr: int,
        config: Optional[TransformConfig] = None,
    ) -> None:
        """Render Skia content onto an NvBufSurface buffer."""
        ...

# ── Module-level functions ──────────────────────────────────────────────

def init_cuda(gpu_id: int = 0) -> None:
    """Initialize CUDA context for the given GPU device.

    Args:
        gpu_id: GPU device ID (default 0).
    """
    ...

def bridge_savant_id_meta(element_ptr: int) -> None:
    """Bridge Savant ID metadata across a GStreamer element."""
    ...

def get_savant_id_meta(buffer_ptr: int) -> List[Tuple[str, int]]:
    """Read Savant ID metadata from a GstBuffer.

    Returns:
        List of ``(source_id, frame_id)`` pairs.
    """
    ...

def get_nvbufsurface_info(buffer_ptr: int) -> Tuple[int, int, int, int]:
    """Get NvBufSurface info from a GstBuffer.

    Returns:
        ``(data_ptr, width, height, pitch)``.
    """
    ...

def set_buffer_pts(buf_ptr: int, pts_ns: int) -> None:
    """Set the presentation timestamp on a raw GstBuffer pointer.

    Args:
        buf_ptr: Raw GstBuffer pointer address.
        pts_ns: Presentation timestamp in nanoseconds.
    """
    ...

def set_buffer_duration(buf_ptr: int, duration_ns: int) -> None:
    """Set the duration on a raw GstBuffer pointer.

    Args:
        buf_ptr: Raw GstBuffer pointer address.
        duration_ns: Duration in nanoseconds.
    """
    ...
