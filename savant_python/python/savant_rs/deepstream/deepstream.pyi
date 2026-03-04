"""Type stubs for ``savant_rs.deepstream`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, final

import cv2
import skia

__all__ = [
    "Padding",
    "Interpolation",
    "ComputeMode",
    "VideoFormat",
    "MemType",
    "Rect",
    "GstBuffer",
    "SurfaceView",
    "TransformConfig",
    "NvBufSurfaceGenerator",
    "BatchedNvBufSurfaceGenerator",
    "BatchedSurface",
    "HeterogeneousBatch",
    "set_num_filled",
    "SkiaContext",
    "SkiaCanvas",
    "init_cuda",
    "gpu_mem_used_mib",
    "bridge_savant_id_meta",
    "get_savant_id_meta",
    "get_nvbufsurface_info",
    "set_buffer_pts",
    "set_buffer_duration",
    "release_buffer",
    "nvgstbuf_as_gpu_mat",
    "nvbuf_as_gpu_mat",
    "from_gpumat",
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

# ── Rect ────────────────────────────────────────────────────────────────

class Rect:
    """A rectangle in pixel coordinates (top, left, width, height).

    Used as an optional source crop region for transform and send_frame.
    """

    top: int
    left: int
    width: int
    height: int

    def __init__(self, top: int, left: int, width: int, height: int) -> None: ...
    def __repr__(self) -> str: ...

# ── GstBuffer ───────────────────────────────────────────────────────────

@final
class GstBuffer:
    """RAII guard owning a ``GstBuffer*`` reference.

    Prevents leaking GStreamer buffers by ensuring ``gst_buffer_unref``
    is called when the guard is garbage-collected or explicitly consumed
    via :meth:`take`.

    Obtain instances from :meth:`NvBufSurfaceGenerator.acquire_surface`,
    :meth:`BatchedSurface.finalize`, :meth:`HeterogeneousBatch.finalize`,
    etc.  Pass them directly to any API that accepts
    ``Union[GstBuffer, int]``.
    """

    @staticmethod
    def from_ptr(ptr: int, add_ref: bool = True) -> GstBuffer:
        """Wrap a raw ``GstBuffer*`` pointer into a guard.

        Args:
            ptr: Raw ``GstBuffer*`` pointer address (must be non-zero).
            add_ref: If ``True`` (default), ``gst_mini_object_ref`` is
                called so the guard owns an additional reference.  Set to
                ``False`` when the caller is transferring ownership.

        Returns:
            A new guard owning the buffer reference.

        Raises:
            ValueError: If *ptr* is 0 (null).
        """
        ...

    @property
    def ptr(self) -> int:
        """Raw ``GstBuffer*`` pointer address.

        Raises:
            RuntimeError: If the guard has been consumed via :meth:`take`.
        """
        ...

    def take(self) -> int:
        """Transfer ownership out of the guard and return the raw pointer.

        After this call the guard is empty (``bool(guard)`` is ``False``)
        and will **not** unref the buffer on destruction.

        Returns:
            Raw ``GstBuffer*`` pointer address.

        Raises:
            RuntimeError: If already consumed.
        """
        ...

    def __repr__(self) -> str: ...
    def __bool__(self) -> bool:
        """``True`` if the guard still owns a buffer, ``False`` if consumed."""
        ...

# ── SurfaceView ─────────────────────────────────────────────────────────

@final
class SurfaceView:
    """Zero-copy view of a single GPU surface.

    Wraps an NvBufSurface-backed buffer or arbitrary CUDA memory with
    cached surface parameters.  Implements ``__cuda_array_interface__``
    for single-plane formats (RGBA, BGRx, GRAY8).

    Construction:

    - ``SurfaceView.from_buffer(buf, slot_index)`` — from a ``GstBuffer``.
    - ``SurfaceView.from_cuda_array(obj)`` — from any object with
      ``__cuda_array_interface__`` (CuPy array, PyTorch CUDA tensor, etc.).
    """

    @staticmethod
    def from_buffer(buf: Union[GstBuffer, int], slot_index: int = 0) -> SurfaceView:
        """Create a view from an NvBufSurface-backed buffer.

        Args:
            buf: Source buffer (``GstBuffer`` guard or raw pointer ``int``).
            slot_index: Zero-based slot index (default 0).

        Raises:
            ValueError: If *buf* is null or *slot_index* is out of bounds.
            RuntimeError: If the buffer is not a valid NvBufSurface or uses
                a multi-plane format (NV12, I420, etc.).
        """
        ...

    @staticmethod
    def from_cuda_array(obj: Any, gpu_id: int = 0) -> SurfaceView:
        """Create a view from any ``__cuda_array_interface__`` object.

        Supported shapes: ``(H, W, C)`` or ``(H, W)`` (grayscale).
        C must be 1 (GRAY8) or 4 (RGBA). dtype must be ``uint8``.

        The source object is kept alive for the lifetime of this view.

        Args:
            obj: CuPy array, PyTorch CUDA tensor, or any object with
                ``__cuda_array_interface__``.
            gpu_id: CUDA device ID (default 0).

        Raises:
            TypeError: If *obj* has no ``__cuda_array_interface__``.
            ValueError: If shape, dtype, or strides are unsupported.
        """
        ...

    @property
    def data_ptr(self) -> int:
        """CUDA data pointer to the first pixel."""
        ...

    @property
    def pitch(self) -> int:
        """Row stride in bytes."""
        ...

    @property
    def width(self) -> int:
        """Surface width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Surface height in pixels."""
        ...

    @property
    def gpu_id(self) -> int:
        """GPU device ID."""
        ...

    @property
    def channels(self) -> int:
        """Number of interleaved channels per pixel."""
        ...

    @property
    def color_format(self) -> int:
        """Raw ``NvBufSurfaceColorFormat`` value."""
        ...

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """CUDA array interface descriptor (v3).

        Allows CuPy, PyTorch, and other CUDA-aware libraries to access
        the surface data without copies.
        """
        ...

    def __repr__(self) -> str: ...
    def __bool__(self) -> bool:
        """``True`` if the view has not been consumed."""
        ...

# ── TransformConfig ─────────────────────────────────────────────────────

class TransformConfig:
    """Configuration for a transform (scale / letterbox) operation.

    All fields have sensible defaults (``Padding.SYMMETRIC``,
    ``Interpolation.BILINEAR``, ``ComputeMode.DEFAULT``).
    """

    padding: Padding
    interpolation: Interpolation
    compute_mode: ComputeMode

    def __init__(
        self,
        padding: Padding = ...,
        interpolation: Interpolation = ...,
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
    def acquire_surface(self, id: Optional[int] = None) -> GstBuffer:
        """Acquire a new NvBufSurface buffer from the pool.

        Returns:
            Guard owning the newly acquired ``GstBuffer``.
        """
        ...

    def acquire_surface_with_params(
        self,
        pts_ns: int,
        duration_ns: int,
        id: Optional[int] = None,
    ) -> GstBuffer:
        """Acquire a buffer and stamp PTS and duration on it.

        Args:
            pts_ns: Presentation timestamp in nanoseconds.
            duration_ns: Frame duration in nanoseconds.
            id: Optional buffer ID / frame index.

        Returns:
            Guard owning the newly acquired ``GstBuffer``.
        """
        ...

    def acquire_surface_with_ptr(
        self,
        id: Optional[int] = None,
    ) -> Tuple[GstBuffer, int, int]:
        """Acquire a buffer and return ``(guard, data_ptr, pitch)``."""
        ...

    def transform(
        self,
        src_buf: Union[GstBuffer, int],
        config: TransformConfig,
        id: Optional[int] = None,
        src_rect: Optional[Rect] = None,
    ) -> GstBuffer:
        """Transform (scale + letterbox) a source buffer into a new destination.

        Returns:
            Guard owning the destination ``GstBuffer``.
        """
        ...

    def transform_with_ptr(
        self,
        src_buf: Union[GstBuffer, int],
        config: TransformConfig,
        id: Optional[int] = None,
        src_rect: Optional[Rect] = None,
    ) -> Tuple[GstBuffer, int, int]:
        """Like :meth:`transform` but also returns ``(guard, data_ptr, pitch)``."""
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
        gst_buffer_dest: Union[GstBuffer, int],
        id: Optional[int] = None,
    ) -> None:
        """Create a new NvBufSurface and attach it to the given buffer."""
        ...

# ── BatchedNvBufSurfaceGenerator ─────────────────────────────────────────

class BatchedNvBufSurfaceGenerator:
    """Homogeneous batched NvBufSurface buffer generator.

    Produces buffers whose ``surfaceList`` is an array of independently
    fillable GPU surfaces, all sharing the same pixel format and
    dimensions.

    Args:
        format: Pixel format (``VideoFormat`` enum or string name,
            e.g. ``"RGBA"``).
        width: Slot width in pixels.
        height: Slot height in pixels.
        max_batch_size: Maximum number of slots per batch.
        pool_size: Number of pre-allocated batched buffers (default 2).
        fps_num: Framerate numerator (default 30).
        fps_den: Framerate denominator (default 1).
        gpu_id: GPU device ID (default 0).
        mem_type: Memory type (default ``MemType.DEFAULT``).

    Raises:
        RuntimeError: If pool creation fails.
    """

    def __init__(
        self,
        format: Union[VideoFormat, str],
        width: int,
        height: int,
        max_batch_size: int,
        pool_size: int = 2,
        fps_num: int = 30,
        fps_den: int = 1,
        gpu_id: int = 0,
        mem_type: Union[MemType, int, None] = None,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def format(self) -> VideoFormat: ...
    @property
    def gpu_id(self) -> int: ...
    @property
    def max_batch_size(self) -> int: ...
    def acquire_batched_surface(self, config: TransformConfig) -> BatchedSurface:
        """Acquire a ``BatchedSurface`` from the pool, ready for slot filling.

        Args:
            config: Scaling / letterboxing configuration applied to every
                :meth:`~BatchedSurface.fill_slot` call on the returned
                surface.

        Returns:
            A fresh batched surface with ``num_filled == 0``.

        Raises:
            RuntimeError: If the pool is exhausted.
        """
        ...

# ── BatchedSurface ───────────────────────────────────────────────────────

class BatchedSurface:
    """Pool-allocated batched NvBufSurface with per-slot fill tracking.

    Obtained from
    :meth:`BatchedNvBufSurfaceGenerator.acquire_batched_surface`.
    Fill individual slots with :meth:`fill_slot`, then call
    :meth:`finalize` to obtain the final ``GstBuffer*`` pointer.
    """

    @property
    def num_filled(self) -> int:
        """Number of slots filled so far."""
        ...

    @property
    def max_batch_size(self) -> int:
        """Maximum number of slots in this batch."""
        ...

    @property
    def buffer_ptr(self) -> int:
        """Raw ``GstBuffer*`` pointer of the underlying buffer.

        Raises:
            RuntimeError: If the surface has been finalized.
        """
        ...

    def slot_ptr(self, index: int) -> Tuple[int, int]:
        """Return ``(data_ptr, pitch)`` for a slot by index.

        Args:
            index: Zero-based slot index (``0 .. max_batch_size - 1``).

        Returns:
            ``(data_ptr, pitch)`` — CUDA device pointer and row stride
            in bytes.

        Raises:
            RuntimeError: If the surface is finalized or *index* is out
                of bounds.
        """
        ...

    def fill_slot(
        self,
        src_buf: Union[GstBuffer, int],
        src_rect: Optional[Rect] = None,
        id: Optional[int] = None,
    ) -> None:
        """Transform a source buffer into the next available batch slot.

        The source surface is scaled (with optional letterboxing) into the
        destination slot according to the ``TransformConfig`` that was passed
        to ``acquire_batched_surface``.  The same source buffer may be used
        for several slots with different *src_rect* regions.

        Args:
            src_buf: ``GstBuffer`` guard or raw ``GstBuffer*`` pointer of the
                source NVMM surface (as returned by
                :meth:`NvBufSurfaceGenerator.acquire_surface`).
            src_rect: Optional crop rectangle applied to the source before
                scaling.  When ``None`` the full source frame is used.
                Coordinates are ``(top, left, width, height)`` in pixels.
            id: Optional frame identifier stored in ``SavantIdMeta``.
                When ``None``, the id is inherited from the source
                buffer's existing ``SavantIdMeta`` (if any).

        Raises:
            ValueError: If the buffer pointer is 0 (null).
            RuntimeError: If the batch is already finalized, the batch
                is full, or the GPU transform fails.
        """
        ...

    def finalize(self) -> GstBuffer:
        """Finalize the batch and return the buffer guard.

        Writes ``SavantIdMeta`` with the collected frame IDs and sets
        ``numFilled`` on the underlying ``NvBufSurface``.  After this
        call the ``BatchedSurface`` is consumed — further method calls
        will raise ``RuntimeError``.

        Returns:
            Guard owning the finalized batched ``GstBuffer``.

        Raises:
            RuntimeError: If already finalized.
        """
        ...

# ── HeterogeneousBatch ───────────────────────────────────────────────────

class HeterogeneousBatch:
    """Zero-copy heterogeneous batch (nvstreammux2-style).

    Assembles individual NvBufSurface buffers of arbitrary dimensions
    and pixel formats into a single batched ``GstBuffer``.

    Args:
        max_batch_size: Maximum number of surfaces in the batch.
        gpu_id: GPU device ID (default 0).

    Raises:
        RuntimeError: If batch creation fails.
    """

    def __init__(self, max_batch_size: int, gpu_id: int = 0) -> None: ...
    @property
    def num_filled(self) -> int:
        """Number of surfaces added so far."""
        ...

    @property
    def max_batch_size(self) -> int:
        """Maximum number of surfaces in this batch."""
        ...

    @property
    def gpu_id(self) -> int:
        """GPU device ID this batch is bound to."""
        ...

    def add(self, src_buf: Union[GstBuffer, int], id: Optional[int] = None) -> None:
        """Add a source buffer to the batch (zero-copy).

        The source buffer's ``NvBufSurface`` is appended to the batch
        without copying pixel data.

        Args:
            src_buf: ``GstBuffer`` guard or raw ``GstBuffer*`` pointer
                of the source NVMM surface.
            id: Optional frame identifier stored in ``SavantIdMeta``.
                When ``None``, the id is inherited from the source
                buffer's existing ``SavantIdMeta`` (if any).

        Raises:
            ValueError: If the buffer pointer is 0 (null).
            RuntimeError: If the batch is already finalized or full.
        """
        ...

    def slot_ptr(self, index: int) -> Tuple[int, int, int, int]:
        """Return ``(data_ptr, pitch, width, height)`` for a slot by index.

        Args:
            index: Zero-based slot index (``0 .. num_filled - 1``).

        Returns:
            ``(data_ptr, pitch, width, height)`` — CUDA device pointer,
            row stride, and the slot's native dimensions in pixels.

        Raises:
            RuntimeError: If the batch is finalized or *index* is out
                of bounds.
        """
        ...

    def finalize(self) -> GstBuffer:
        """Finalize the batch and return the buffer guard.

        Writes ``SavantIdMeta`` with the collected frame IDs and
        assembles the heterogeneous ``NvBufSurface``.  After this call
        the ``HeterogeneousBatch`` is consumed — further method calls
        will raise ``RuntimeError``.

        Returns:
            Guard owning the finalized batch ``GstBuffer``.

        Raises:
            RuntimeError: If already finalized.
        """
        ...

# ── SkiaContext ──────────────────────────────────────────────────────────

class SkiaContext:
    """GPU-accelerated Skia rendering context backed by CUDA-GL interop."""

    def __init__(self, width: int, height: int, gpu_id: int = 0) -> None: ...
    @staticmethod
    def from_nvbuf(buf: Union[GstBuffer, int], gpu_id: int = 0) -> SkiaContext:
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
        buf: Union[GstBuffer, int],
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

def gpu_mem_used_mib(gpu_id: int = 0) -> int:
    """Return GPU memory currently used, in MiB.

    - dGPU (x86_64): Uses NVML to query device gpu_id.
    - Jetson (aarch64): Reads /proc/meminfo (unified memory).

    Args:
        gpu_id: GPU device ID (default 0).

    Returns:
        GPU memory used in MiB.
    """
    ...

def bridge_savant_id_meta(element_ptr: int) -> None:
    """Bridge Savant ID metadata across a GStreamer element."""
    ...

def get_savant_id_meta(buf: Union[GstBuffer, int]) -> List[Tuple[str, int]]:
    """Read Savant ID metadata from a GstBuffer.

    Returns:
        List of ``(source_id, frame_id)`` pairs.
    """
    ...

def get_nvbufsurface_info(buf: Union[GstBuffer, int]) -> Tuple[int, int, int, int]:
    """Get NvBufSurface info from a GstBuffer.

    Returns:
        ``(data_ptr, pitch, width, height)``.
    """
    ...

def set_buffer_pts(buf: Union[GstBuffer, int], pts_ns: int) -> None:
    """Set the presentation timestamp on a GstBuffer.

    Args:
        buf: ``GstBuffer`` guard or raw pointer address.
        pts_ns: Presentation timestamp in nanoseconds.
    """
    ...

def set_num_filled(buf: Union[GstBuffer, int], count: int) -> None:
    """Set numFilled on a batched NvBufSurface GstBuffer.

    Args:
        buf: ``GstBuffer`` guard or raw pointer to a batched NvBufSurface.
        count: Number of filled slots.
    """
    ...

def set_buffer_duration(buf: Union[GstBuffer, int], duration_ns: int) -> None:
    """Set the duration on a GstBuffer.

    Args:
        buf: ``GstBuffer`` guard or raw pointer address.
        duration_ns: Duration in nanoseconds.
    """
    ...

def release_buffer(buf_ptr: int) -> None:
    """Release (unref) a raw ``GstBuffer*`` pointer.

    Call this to free a buffer obtained from ``acquire_surface``,
    ``acquire_surface_with_params``, ``acquire_surface_with_ptr``,
    ``transform``, ``transform_with_ptr``, or ``finalize`` when the
    buffer is no longer needed and is not being passed into a GStreamer
    pipeline.

    Args:
        buf_ptr: Raw ``GstBuffer*`` pointer to release.

    Raises:
        ValueError: If *buf_ptr* is 0 (null).
    """
    ...

# ── SkiaCanvas ──────────────────────────────────────────────────────────

class SkiaCanvas:
    """Convenience wrapper: SkiaContext + skia-python in one object.

    Handles creation of the skia GrDirectContext and Surface backed
    by the SkiaContext's GPU FBO.
    """

    def __init__(self, ctx: SkiaContext) -> None: ...
    @classmethod
    def from_fbo(cls, fbo_id: int, width: int, height: int) -> SkiaCanvas:
        """Create from an existing OpenGL FBO.

        Args:
            fbo_id: OpenGL FBO ID backing the canvas.
            width:  Canvas width in pixels.
            height: Canvas height in pixels.
        """
        ...

    @classmethod
    def create(cls, width: int, height: int, gpu_id: int = 0) -> SkiaCanvas:
        """Create with an empty (transparent) canvas.

        Args:
            width:  Canvas width in pixels.
            height: Canvas height in pixels.
            gpu_id: GPU device ID (default 0).
        """
        ...

    @classmethod
    def from_nvbuf(cls, buf_ptr: int, gpu_id: int = 0) -> SkiaCanvas:
        """Create with canvas pre-loaded from an NvBufSurface.

        Args:
            buf_ptr: Raw pointer of the source GstBuffer.
            gpu_id:  GPU device ID (default 0).
        """
        ...

    @property
    def gr_context(self) -> skia.GrDirectContext:
        """The Skia GPU ``GrDirectContext`` backing this canvas."""
        ...

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        ...

    def canvas(self) -> skia.Canvas:
        """Get the skia-python Canvas for drawing."""
        ...

    def render_to_nvbuf(
        self,
        buf_ptr: int,
        config: Optional[TransformConfig] = None,
    ) -> None:
        """Flush Skia and copy to destination NvBufSurface.

        Args:
            buf_ptr: Raw pointer of the destination GstBuffer.
            config:  Optional ``TransformConfig`` for scaling / letterboxing.
        """
        ...

# ── GpuMat helpers ──────────────────────────────────────────────────────

@contextmanager
def nvgstbuf_as_gpu_mat(
    buf: Union[GstBuffer, int],
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface ``GstBuffer`` as an OpenCV CUDA ``GpuMat``.

    Extracts the CUDA device pointer, pitch, width and height from the
    buffer's NvBufSurface metadata, then creates a zero-copy ``GpuMat``
    together with a CUDA ``Stream``.  When the ``with`` block exits the
    stream is synchronised (``waitForCompletion``).

    Args:
        buf: ``GstBuffer`` guard or raw ``GstBuffer*`` pointer address.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4`` with the
        buffer's native width, height and pitch.
    """
    ...

@contextmanager
def nvbuf_as_gpu_mat(
    data_ptr: int,
    pitch: int,
    width: int,
    height: int,
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Wrap raw CUDA memory as an OpenCV CUDA ``GpuMat``.

    Unlike :func:`nvgstbuf_as_gpu_mat`, this takes the CUDA device
    pointer and layout directly.  Designed for the Picasso ``on_gpumat``
    callback which supplies these values.

    Args:
        data_ptr: CUDA device pointer to the surface data.
        pitch: Row stride in bytes.
        width: Surface width in pixels.
        height: Surface height in pixels.

    Yields:
        ``(gpumat, stream)`` -- the ``GpuMat`` is ``CV_8UC4``.
    """
    ...

def from_gpumat(
    gen: NvBufSurfaceGenerator,
    gpumat: cv2.cuda.GpuMat,
    *,
    interpolation: int = ...,
    id: Optional[int] = None,
) -> GstBuffer:
    """Acquire a buffer from the pool and fill it from a ``GpuMat``.

    If the source ``GpuMat`` dimensions differ from the generator's
    dimensions the image is scaled using :func:`cv2.cuda.resize` with
    the given *interpolation* method.  When sizes match the data is
    copied directly (zero-overhead ``copyTo``).

    Args:
        gen: Surface generator (determines destination dimensions and
            format).
        gpumat: Source ``GpuMat`` (must be ``CV_8UC4``).
        interpolation: OpenCV interpolation flag (default
            ``cv2.INTER_LINEAR``).
        id: Optional frame identifier for ``SavantIdMeta``.

    Returns:
        Guard owning the newly acquired ``GstBuffer``.
    """
    ...
