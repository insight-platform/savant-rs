"""Type stubs for ``savant_rs.deepstream`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, final

import cv2
import numpy as np
import skia

__all__ = [
    "Padding",
    "DstPadding",
    "Interpolation",
    "ComputeMode",
    "VideoFormat",
    "MemType",
    "Rect",
    "SharedBuffer",
    "SurfaceView",
    "TransformConfig",
    "BufferGenerator",
    "UniformBatchGenerator",
    "SurfaceBatch",
    "NonUniformBatch",
    "SavantIdMetaKind",
    "set_num_filled",
    "SkiaContext",
    "SkiaCanvas",
    "init_cuda",
    "gpu_mem_used_mib",
    "jetson_model",
    "is_jetson_kernel",
    "has_nvenc",
    "get_savant_id_meta",
    "get_nvbufsurface_info",
    "GpuMatCudaArray",
    "make_gpu_mat",
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
class DstPadding:
    """Optional per-side destination padding for letterboxing.

    When set in ``TransformConfig.dst_padding``, reduces the effective
    destination area before the letterbox rect is computed.
    """

    left: int
    top: int
    right: int
    bottom: int

    def __init__(
        self,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
    ) -> None: ...
    def __repr__(self) -> str: ...

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
class SavantIdMetaKind:
    """Kind tag for ``SavantIdMeta`` entries.

    - ``FRAME`` -- per-frame identifier.
    - ``BATCH`` -- per-batch identifier.
    """

    FRAME: SavantIdMetaKind
    BATCH: SavantIdMetaKind

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

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

# ── SharedBuffer ──────────────────────────────────────────────────────────

@final
class SharedBuffer:
    """Safe wrapper for a shared GStreamer buffer.

    Uses the Option<T> pattern to emulate Rust move semantics.
    After a consuming Rust method (e.g. NvInfer.submit) consumes
    the buffer, all subsequent property access raises RuntimeError.

    Cannot be constructed, cloned, or deconstructed from Python.
    """

    @property
    def strong_count(self) -> int:
        """Number of strong Arc references to the underlying buffer."""
        ...

    @property
    def pts_ns(self) -> Optional[int]:
        """Buffer PTS in nanoseconds, or None if unset."""
        ...

    @pts_ns.setter
    def pts_ns(self, value: int) -> None: ...
    @property
    def duration_ns(self) -> Optional[int]:
        """Buffer duration in nanoseconds, or None if unset."""
        ...

    @duration_ns.setter
    def duration_ns(self, value: int) -> None: ...
    def savant_ids(self) -> List[Tuple[SavantIdMetaKind, int]]:
        """Read SavantIdMeta from the buffer.

        Returns:
            list of (kind, id) tuples, e.g.
            ``[(SavantIdMetaKind.FRAME, 42)]``.
        """
        ...

    def set_savant_ids(self, ids: List[Tuple[SavantIdMetaKind, int]]) -> None:
        """Replace SavantIdMeta on the buffer.

        Args:
            ids: list of (kind, id) tuples.
        """
        ...

    @property
    def is_consumed(self) -> bool:
        """True if the buffer has been consumed."""
        ...

    def __bool__(self) -> bool:
        """True if the buffer has not been consumed."""
        ...

    def __repr__(self) -> str: ...

# ── SurfaceView ─────────────────────────────────────────────────────────

@final
class SurfaceView:
    """Zero-copy view of a single GPU surface.

    Wraps an NvBufSurface-backed buffer or arbitrary CUDA memory with
    cached surface parameters.  Implements ``__cuda_array_interface__``
    for single-plane formats (RGBA, BGRx, GRAY8).

    Construction:

    - ``SurfaceView.from_buffer(buf, slot_index)`` — from a ``SharedBuffer`` or raw pointer.
    - ``SurfaceView.from_cuda_array(obj)`` — from any object with
      ``__cuda_array_interface__`` (CuPy array, PyTorch CUDA tensor, etc.).
    """

    @staticmethod
    def from_buffer(
        buf: Union[SharedBuffer, int], slot_index: int = 0, cuda_stream: int = 0
    ) -> SurfaceView:
        """Create a view from an NvBufSurface-backed buffer.

        Args:
            buf: Source buffer (``SharedBuffer`` or raw pointer ``int``).
            slot_index: Zero-based slot index (default 0).

        Raises:
            ValueError: If *buf* is null or *slot_index* is out of bounds.
            RuntimeError: If the buffer is not a valid NvBufSurface or uses
                a multi-plane format (NV12, I420, etc.).
        """
        ...

    @staticmethod
    def from_cuda_array(obj: Any, gpu_id: int = 0, cuda_stream: int = 0) -> SurfaceView:
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
    def cuda_stream(self) -> int:
        """CUDA stream handle associated with this view (as an integer pointer).

        Returns 0 for the default (legacy) stream.
        """
        ...

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """CUDA array interface descriptor (v3).

        Allows CuPy, PyTorch, and other CUDA-aware libraries to access
        the surface data without copies.
        """
        ...

    def memset(self, value: int) -> None:
        """Fill the surface with a constant byte value.

        Every byte of the surface (up to ``pitch × height``) is set to
        *value*.  Use :meth:`fill` for arbitrary per-channel colours.

        Args:
            value: Byte value (0–255) to fill every byte with.

        Raises:
            RuntimeError: If the view has been consumed or the GPU operation fails.
        """
        ...

    def fill(self, color: List[int]) -> None:
        """Fill the surface with a repeating pixel colour.

        *color* must have exactly as many elements as the surface's
        channel count (e.g. ``[R, G, B, A]`` for RGBA, ``[Y]`` for GRAY8).

        Example::

            view.fill([128, 0, 255, 255])   # semi-blue, opaque RGBA

        Args:
            color: Per-channel byte values (0–255 each).

        Raises:
            ValueError: If *color* length does not match channel count.
            RuntimeError: If the view has been consumed or the GPU operation fails.
        """
        ...

    def upload(self, data: np.ndarray) -> None:
        """Upload pixel data from a NumPy array to the surface.

        Args:
            data: A 3-D ``uint8`` array with shape ``(height, width, channels)``
                matching the surface dimensions and color format (e.g. 4 channels for RGBA).

        Raises:
            ValueError: If *data* has wrong shape, dtype, or dimensions.
            RuntimeError: If the view has been consumed or the GPU operation fails.
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
    dst_padding: Optional[DstPadding]
    interpolation: Interpolation
    compute_mode: ComputeMode

    def __init__(
        self,
        padding: Padding = ...,
        dst_padding: Optional[DstPadding] = None,
        interpolation: Interpolation = ...,
        compute_mode: ComputeMode = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ── BufferGenerator ───────────────────────────────────────────────────────

class BufferGenerator:
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
    def acquire(self, id: Optional[int] = None) -> SharedBuffer:
        """Acquire a new NvBufSurface buffer from the pool.

        Returns:
            SharedBuffer owning the newly acquired ``GstBuffer``.
        """
        ...

    def acquire_with_params(
        self,
        pts_ns: int,
        duration_ns: int,
        id: Optional[int] = None,
    ) -> SharedBuffer:
        """Acquire a buffer and stamp PTS and duration on it.

        Args:
            pts_ns: Presentation timestamp in nanoseconds.
            duration_ns: Frame duration in nanoseconds.
            id: Optional buffer ID / frame index.

        Returns:
            SharedBuffer owning the newly acquired ``GstBuffer``.
        """
        ...

    def transform(
        self,
        src_buf: Union[SharedBuffer, int],
        config: TransformConfig,
        id: Optional[int] = None,
        src_rect: Optional[Rect] = None,
    ) -> SharedBuffer:
        """Transform (scale + letterbox) a source buffer into a new destination.

        Returns:
            SharedBuffer owning the destination ``GstBuffer``.
        """
        ...

    @staticmethod
    def send_eos(appsrc_ptr: int) -> None:
        """Send an end-of-stream signal to an AppSrc element."""
        ...

# ── UniformBatchGenerator ────────────────────────────────────────────────

class UniformBatchGenerator:
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
    def acquire_batch(
        self,
        config: TransformConfig,
        ids: Optional[List[Tuple[SavantIdMetaKind, int]]] = None,
    ) -> SurfaceBatch:
        """Acquire a ``SurfaceBatch`` from the pool, ready for slot filling.

        Args:
            config: Scaling / letterboxing configuration applied to every
                :meth:`~SurfaceBatch.transform_slot` call on the returned
                surface.
            ids: Optional list of ``(SavantIdMetaKind, id)`` tuples.

        Returns:
            A fresh batched surface with ``num_filled == 0``.

        Raises:
            RuntimeError: If the pool is exhausted.
        """
        ...

# ── SurfaceBatch ──────────────────────────────────────────────────────────

class SurfaceBatch:
    """Pool-allocated batched NvBufSurface with per-slot fill tracking.

    Obtained from
    :meth:`UniformBatchGenerator.acquire_batch`.
    Fill individual slots with :meth:`transform_slot`, then call
    :meth:`finalize`, then :meth:`shared_buffer` to access the buffer.
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
    def is_finalized(self) -> bool:
        """Whether the batch has been finalized."""
        ...

    def shared_buffer(self) -> SharedBuffer:
        """Return the underlying SharedBuffer. Available only after finalize.

        Returns:
            SharedBuffer for the finalized batched buffer.

        Raises:
            RuntimeError: If not yet finalized.
        """
        ...

    def view(self, slot_index: int) -> SurfaceView:
        """Create a zero-copy single-frame view of one filled slot.

        Available only after finalize.

        Raises:
            RuntimeError: If not yet finalized or slot index out of bounds.
        """
        ...

    def transform_slot(
        self,
        slot: int,
        src_buf: Union[SharedBuffer, int],
        src_rect: Optional[Rect] = None,
    ) -> None:
        """Transform a source buffer into the specified batch slot.

        The source surface is scaled (with optional letterboxing) into the
        destination slot according to the ``TransformConfig`` that was passed
        to ``acquire_batch``.  The same source buffer may be used
        for several slots with different *src_rect* regions.

        Args:
            slot: Zero-based slot index to fill.
            src_buf: ``SharedBuffer`` or raw ``GstBuffer*`` pointer of the
                source NVMM surface (as returned by
                :meth:`BufferGenerator.acquire`).
            src_rect: Optional crop rectangle applied to the source before
                scaling.  When ``None`` the full source frame is used.
                Coordinates are ``(top, left, width, height)`` in pixels.

        Raises:
            ValueError: If the buffer pointer is 0 (null).
            RuntimeError: If the batch is already finalized, the slot
                is out of bounds, or the GPU transform fails.
        """
        ...

    def finalize(self) -> None:
        """Finalize the batch (non-consuming).

        Writes ``SavantIdMeta`` with the collected frame IDs and sets
        ``numFilled`` on the underlying ``NvBufSurface``.  Call
        :meth:`shared_buffer` afterward to access the buffer.

        Raises:
            RuntimeError: If already finalized.
        """
        ...

    def memset_slot(self, index: int, value: int) -> None:
        """Fill a slot's surface with a constant byte value.

        Args:
            index: Zero-based slot index.
            value: Byte value (0–255).

        Raises:
            RuntimeError: If the batch is not finalized, *index* is out of
                bounds, or the GPU operation fails.
        """
        ...

    def upload_slot(self, index: int, data: np.ndarray) -> None:
        """Upload pixel data from a NumPy array into a batch slot.

        Args:
            index: Zero-based slot index.
            data: A 3-D ``uint8`` array with shape ``(height, width, channels)``
                matching the slot dimensions.

        Raises:
            ValueError: If *data* has wrong shape, dtype, or dimensions.
            RuntimeError: If the batch is not finalized, *index* is out of
                bounds, or the GPU operation fails.
        """
        ...

# ── NonUniformBatch ───────────────────────────────────────────────────────

class NonUniformBatch:
    """Zero-copy heterogeneous batch (nvstreammux2-style).

    Assembles individual SurfaceView surfaces of arbitrary dimensions
    and pixel formats into a single batched ``GstBuffer``.

    Args:
        gpu_id: GPU device ID (default 0).

    Raises:
        RuntimeError: If batch creation fails.
    """

    def __init__(self, gpu_id: int = 0) -> None: ...
    @property
    def num_filled(self) -> int:
        """Number of surfaces added so far.

        Raises:
            RuntimeError: If the batch has been finalized.
        """
        ...

    @property
    def gpu_id(self) -> int:
        """GPU device ID this batch is bound to."""
        ...

    def add(self, src_view: SurfaceView) -> None:
        """Add a source SurfaceView to the batch (zero-copy).

        The source view's surface is appended to the batch
        without copying pixel data.

        Args:
            src_view: SurfaceView of the source surface.

        Raises:
            RuntimeError: If the batch is already finalized.
        """
        ...

    def finalize(
        self,
        ids: Optional[List[Tuple[SavantIdMetaKind, int]]] = None,
    ) -> SharedBuffer:
        """Finalize the batch (consuming).

        Writes ``SavantIdMeta`` with the collected frame IDs and
        assembles the heterogeneous ``NvBufSurface``.  Consumes the
        batch and returns the resulting SharedBuffer.

        Args:
            ids: Optional list of (kind, id) tuples where kind is
                "frame" or "batch".

        Returns:
            SharedBuffer for the finalized batch buffer.

        Raises:
            RuntimeError: If already finalized.
        """
        ...

# ── SkiaContext ──────────────────────────────────────────────────────────

class SkiaContext:
    """GPU-accelerated Skia rendering context backed by CUDA-GL interop."""

    def __init__(self, width: int, height: int, gpu_id: int = 0) -> None: ...
    @staticmethod
    def from_nvbuf(
        buf: Union[SurfaceView, SharedBuffer, int], gpu_id: int = 0
    ) -> SkiaContext:
        """Create a SkiaContext from an existing NvBufSurface buffer.

        Accepts a ``SurfaceView`` (preferred — the CUDA pointer is already
        resolved), a ``SharedBuffer``, or a raw ``GstBuffer*``
        pointer as ``int``.
        """
        ...

    @property
    def fbo_id(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def render_to_nvbuf(
        self,
        buf: Union[SharedBuffer, int],
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

def jetson_model(gpu_id: int = 0) -> Optional[str]:
    """Return the Jetson model name if running on a Jetson device, or None if not.

    Uses CUDA SM count and /proc/meminfo MemTotal to identify the model.
    Works inside containers where /proc/device-tree is typically not mounted.
    Requires uname -r to contain "tegra" and a working CUDA.

    Args:
        gpu_id: GPU device ID (default 0).

    Returns:
        Model name (e.g. "Orin Nano 8GB") or None if not Jetson.
    """
    ...

def is_jetson_kernel() -> bool:
    """Return True if the kernel is a Jetson (Tegra) kernel.

    Checks uname -r for the "tegra" suffix.
    """
    ...

def gpu_architecture(gpu_id: int = 0) -> Optional[str]:
    """Return the GPU architecture family name (x86_64 dGPU only, via NVML).

    Returns a lowercase architecture name such as ``"ampere"``, ``"ada"``,
    ``"hopper"``, ``"turing"``, etc.  Returns ``None`` on Jetson/aarch64.

    Args:
        gpu_id: GPU device ID (default 0).

    Returns:
        Architecture name or None if not on x86_64.
    """
    ...

def gpu_platform_tag(gpu_id: int = 0) -> str:
    """Return a directory-safe platform tag for TensorRT engine caching.

    - Jetson: Jetson model name (e.g. ``"agx_orin_64gb"``, ``"orin_nano_8gb"``).
    - dGPU (x86_64): GPU architecture family (e.g. ``"ampere"``, ``"ada"``).
    - Unknown: ``"unknown"`` if the platform cannot be determined.

    Args:
        gpu_id: GPU device ID (default 0).

    Returns:
        Platform tag string.
    """
    ...

def has_nvenc(gpu_id: int = 0) -> bool:
    """Return True if the GPU has NVENC hardware encoding support.

    - Jetson: Orin Nano is the only Jetson without NVENC; all others have it.
      Unknown models conservatively return False.
    - dGPU (x86_64): Uses NVML encoder_capacity(H264) — returns False for
      datacenter GPUs without NVENC (H100, A100, A30, etc.).

    Args:
        gpu_id: GPU device ID (default 0).

    Returns:
        True if NVENC is available.
    """
    ...

def get_savant_id_meta(
    buf: Union[SharedBuffer, int],
) -> List[Tuple[SavantIdMetaKind, int]]:
    """Read Savant ID metadata from a GstBuffer.

    Returns:
        List of ``(source_id, frame_id)`` pairs.
    """
    ...

def get_nvbufsurface_info(buf: Union[SharedBuffer, int]) -> Tuple[int, int, int, int]:
    """Get NvBufSurface info from a GstBuffer.

    Returns:
        ``(data_ptr, pitch, width, height)``.
    """
    ...

def set_num_filled(buf: Union[SharedBuffer, int], count: int) -> None:
    """Set numFilled on a batched NvBufSurface GstBuffer.

    Args:
        buf: ``SharedBuffer`` or raw pointer to a batched NvBufSurface.
        count: Number of filled slots.
    """
    ...

def _test_consume_shared_buffer(buf: SharedBuffer) -> None:
    """Consume a SharedBuffer for testing (debug builds only).

    Args:
        buf: The buffer to consume.
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

class GpuMatCudaArray:
    """Exposes ``__cuda_array_interface__`` (v3) for a ``cv2.cuda.GpuMat``.

    OpenCV's ``GpuMat`` does not implement the protocol natively; this
    wrapper bridges it to consumers like Picasso ``send_frame``.

    Only ``CV_8UC1`` (GRAY8) and ``CV_8UC4`` (RGBA) mats are supported.
    """

    __slots__ = ("_mat", "__cuda_array_interface__")

    def __init__(self, mat: cv2.cuda.GpuMat) -> None: ...

def make_gpu_mat(width: int, height: int, channels: int = 4) -> cv2.cuda.GpuMat:
    """Allocate a zero-initialised ``cv2.cuda.GpuMat`` of the given size."""
    ...

@contextmanager
def nvgstbuf_as_gpu_mat(
    buf: Union[SharedBuffer, int],
) -> Generator[tuple[cv2.cuda.GpuMat, cv2.cuda.Stream], None, None]:
    """Expose an NvBufSurface ``GstBuffer`` as an OpenCV CUDA ``GpuMat``.

    Extracts the CUDA device pointer, pitch, width and height from the
    buffer's NvBufSurface metadata, then creates a zero-copy ``GpuMat``
    together with a CUDA ``Stream``.  When the ``with`` block exits the
    stream is synchronised (``waitForCompletion``).

    Args:
        buf: ``SharedBuffer`` or raw ``GstBuffer*`` pointer address.

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
    gen: BufferGenerator,
    gpumat: cv2.cuda.GpuMat,
    *,
    interpolation: int = ...,
    id: Optional[int] = None,
) -> SharedBuffer:
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
