"""Type stubs for ``savant_rs.deepstream`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, final

import cv2
import numpy as np
import skia

from savant_rs.gstreamer import Codec
from savant_rs.primitives import EndOfStream, VideoFrame

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
    "MetaClearPolicy",
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
    "FlexibleDecoderConfig",
    "DecoderParameters",
    "SkipReason",
    "DecodedFrame",
    "SealedDelivery",
    "FrameOutput",
    "ParameterChangeOutput",
    "SkippedOutput",
    "OrphanFrameOutput",
    "SourceEosOutput",
    "EventOutput",
    "ErrorOutput",
    "RestartedOutput",
    "FlexibleDecoderOutput",
    "FlexibleDecoder",
    "EvictionDecision",
    "FlexibleDecoderPoolConfig",
    "FlexibleDecoderPool",
    "H264StreamFormat",
    "HevcStreamFormat",
    "JpegBackend",
    "CudadecMemtype",
    "H264DecoderConfig",
    "HevcDecoderConfig",
    "Vp8DecoderConfig",
    "Vp9DecoderConfig",
    "Av1DecoderConfig",
    "JpegDecoderConfig",
    "PngDecoderConfig",
    "RawRgbaDecoderConfig",
    "RawRgbDecoderConfig",
    "DecoderConfig",
    "nvgstbuf_as_gpu_mat",
    "nvbuf_as_gpu_mat",
    "from_gpumat",
    # encoder enums
    "Platform",
    "RateControl",
    "H264Profile",
    "HevcProfile",
    "DgpuPreset",
    "TuningPreset",
    "JetsonPresetLevel",
    # encoder property structs
    "H264DgpuProps",
    "HevcDgpuProps",
    "H264JetsonProps",
    "HevcJetsonProps",
    "JpegProps",
    "PngProps",
    "Av1DgpuProps",
    "Av1JetsonProps",
    "EncoderProperties",
    "EncoderConfig",
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

    @staticmethod
    def from_name(name: str) -> Padding:
        """Parse a padding mode from a string name.

        Accepts ``"none"``, ``"right_bottom"`` / ``"rightbottom"``,
        ``"symmetric"``. Case-insensitive.

        Args:
            name: Padding mode name.

        Returns:
            The parsed padding mode.

        Raises:
            ValueError: If the name is not recognized.
        """
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Interpolation:
    """Interpolation method for scaling.

    Variants whose behaviour differs between GPU (dGPU / x86_64) and VIC
    (Video Image Compositor / Jetson) carry compound names.

    - ``NEAREST``                -- nearest-neighbor (same on both).
    - ``BILINEAR``               -- bilinear (default, same on both).
    - ``GPU_CUBIC_VIC_5TAP``     -- GPU: cubic, VIC: 5-tap.
    - ``GPU_SUPER_VIC_10TAP``    -- GPU: super-sampling, VIC: 10-tap.
    - ``GPU_LANCZOS_VIC_SMART``  -- GPU: Lanczos, VIC: smart.
    - ``GPU_IGNORED_VIC_NICEST`` -- GPU: ignored (no-op), VIC: nicest.
    - ``DEFAULT``                -- platform default (nearest on both).
    """

    NEAREST: Interpolation
    BILINEAR: Interpolation
    GPU_CUBIC_VIC_5TAP: Interpolation
    GPU_SUPER_VIC_10TAP: Interpolation
    GPU_LANCZOS_VIC_SMART: Interpolation
    GPU_IGNORED_VIC_NICEST: Interpolation
    DEFAULT: Interpolation

    @staticmethod
    def from_name(name: str) -> Interpolation:
        """Parse an interpolation method from a string name.

        Accepts canonical names (``"gpu_cubic_vic_5tap"``, ``"gpu_lanczos_vic_smart"``,
        etc.), legacy short names (``"cubic"``, ``"lanczos"``, ``"nicest"``),
        and DeepStream names (``"algo1"``–``"algo4"``). Case-insensitive,
        underscores are stripped before matching.

        Args:
            name: Interpolation method name.

        Returns:
            The parsed interpolation method.

        Raises:
            ValueError: If the name is not recognized.
        """
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

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
    @staticmethod
    def uniform(value: int) -> DstPadding:
        """Create destination padding with equal values on all sides.

        Args:
            value: Padding value applied to left, top, right, and bottom.

        Returns:
            A new ``DstPadding`` with all sides set to *value*.
        """
        ...

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
class MetaClearPolicy:
    """Controls when ``NvDsObjectMeta`` entries are erased from the batch buffer.

    Shared by both :mod:`savant_rs.nvinfer` and :mod:`savant_rs.nvtracker`
    pipelines; re-exported from those modules for convenience.

    - ``NONE`` -- never clear automatically.
    - ``BEFORE`` -- clear stale objects before attaching new objects (default).
    - ``AFTER`` -- clear all objects when the output is dropped.
    - ``BOTH`` -- clear before submission **and** after the output is dropped.
    """

    NONE: MetaClearPolicy
    BEFORE: MetaClearPolicy
    AFTER: MetaClearPolicy
    BOTH: MetaClearPolicy

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class SavantIdMetaKind:
    """Kind tag for ``SavantIdMeta`` entries.

    - ``FRAME`` -- per-frame identifier.
    - ``BATCH`` -- per-batch identifier.

    The associated numeric id is an unsigned 128-bit value (``0 <= id < 2**128``)
    at the Rust boundary; Python passes it as ``int``.
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

# ── FlexibleDecoder types ─────────────────────────────────────────────────

@final
class FlexibleDecoderConfig:
    """Configuration for a :class:`FlexibleDecoder`.

    Args:
        source_id: Bound source_id; frames with a different source_id are rejected.
        gpu_id: GPU device ordinal.
        pool_size: Number of RGBA buffers per internal decoder pool.
    """

    def __init__(
        self, source_id: str, gpu_id: int, pool_size: int
    ) -> None: ...

    def with_idle_timeout_ms(self, ms: int) -> FlexibleDecoderConfig:
        """Set the idle timeout for graceful drain (milliseconds).  Returns a new config."""
        ...

    def with_detect_buffer_limit(self, n: int) -> FlexibleDecoderConfig:
        """Set the max frames buffered during H.264/HEVC stream detection."""
        ...

    def with_decoder_config_callback(
        self,
        cb: Callable[[DecoderConfig, VideoFrame], DecoderConfig],
    ) -> FlexibleDecoderConfig:
        """Install a decoder-config transformation callback.

        The callback is invoked each time the underlying decoder is
        (re-)activated (first submit and every subsequent codec / resolution
        change).  Exceptions and non-:class:`DecoderConfig` return values are
        logged and fall back to the original config.
        """
        ...

    @property
    def source_id(self) -> str: ...
    @property
    def gpu_id(self) -> int: ...
    @property
    def pool_size(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class DecoderParameters:
    """Codec, width and height snapshot for a decoder session."""

    @property
    def codec(self) -> Codec: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class SkipReason:
    """Why a frame was not decoded.

    Use the ``is_*`` properties to determine the variant, and :attr:`detail`
    for the human-readable payload of string-carrying variants.
    """

    @property
    def is_source_id_mismatch(self) -> bool: ...
    @property
    def is_unsupported_codec(self) -> bool: ...
    @property
    def is_waiting_for_keyframe(self) -> bool: ...
    @property
    def is_detection_buffer_overflow(self) -> bool: ...
    @property
    def is_no_payload(self) -> bool: ...
    @property
    def is_invalid_payload(self) -> bool: ...
    @property
    def is_decoder_creation_failed(self) -> bool: ...
    @property
    def is_decoder_restarted(self) -> bool: ...
    @property
    def detail(self) -> Optional[str]:
        """Human-readable detail for string-carrying variants, or ``None``."""
        ...

    def __repr__(self) -> str: ...

@final
class DecodedFrame:
    """Scalar metadata from a decoded frame.

    The GPU buffer (if not yet taken via
    :meth:`FlexibleDecoderOutput.take_delivery`) is indicated by
    :attr:`has_buffer`.
    """

    @property
    def frame_id(self) -> Optional[int]: ...
    @property
    def pts_ns(self) -> int: ...
    @property
    def dts_ns(self) -> Optional[int]: ...
    @property
    def duration_ns(self) -> Optional[int]: ...
    @property
    def codec(self) -> Codec: ...
    @property
    def format(self) -> VideoFormat: ...
    @property
    def has_buffer(self) -> bool: ...
    def __repr__(self) -> str: ...

@final
class SealedDelivery:
    """A ``(VideoFrame, SharedBuffer)`` pair sealed until the associated
    :class:`FlexibleDecoderOutput` is dropped.

    Call :meth:`unseal` (blocking, GIL released) or :meth:`try_unseal`
    (non-blocking) to obtain the pair.
    """

    def is_released(self) -> bool:
        """Whether the seal has been released (non-blocking check)."""
        ...

    def unseal(
        self, timeout_ms: Optional[int] = None
    ) -> Tuple[VideoFrame, SharedBuffer]:
        """Block until the :class:`FlexibleDecoderOutput` is dropped, then
        return the ``(VideoFrame, SharedBuffer)`` pair.

        The GIL is released during the blocking wait.

        Args:
            timeout_ms: Optional timeout in milliseconds.

        Raises:
            RuntimeError: If already consumed by a previous call.
            TimeoutError: If the timeout expires before the seal is released.
        """
        ...

    def try_unseal(self) -> Optional[Tuple[VideoFrame, SharedBuffer]]:
        """Non-blocking attempt to unseal.

        Returns ``(VideoFrame, SharedBuffer)`` if the seal has been
        released, or ``None`` if still sealed.

        Raises:
            RuntimeError: If already consumed by a previous call.
        """
        ...

    def __repr__(self) -> str: ...

@final
class FrameOutput:
    """Decoded frame paired with the submitted :class:`VideoFrame`.

    Owns the underlying Rust output so that its ``Drop`` releases the
    seal when this object is garbage-collected.

    Call :meth:`take_delivery` to extract a :class:`SealedDelivery`.
    """

    @property
    def frame(self) -> VideoFrame:
        """The submitted :class:`VideoFrame`."""
        ...

    @property
    def decoded_frame(self) -> DecodedFrame:
        """Scalar metadata of the decoded frame."""
        ...

    def take_delivery(self) -> Optional[SealedDelivery]:
        """Extract the sealed ``(VideoFrame, SharedBuffer)`` delivery.

        Returns :class:`SealedDelivery` on the first call.
        Subsequent calls return ``None``.
        """
        ...

    def __repr__(self) -> str: ...

@final
class ParameterChangeOutput:
    """Codec or resolution changed between two decoder sessions."""

    @property
    def old(self) -> DecoderParameters:
        """Previous decoder parameters."""
        ...

    @property
    def new(self) -> DecoderParameters:
        """New decoder parameters."""
        ...

    def __repr__(self) -> str: ...

@final
class SkippedOutput:
    """A frame that was rejected (not submitted to the decoder)."""

    @property
    def frame(self) -> VideoFrame:
        """The rejected :class:`VideoFrame`."""
        ...

    @property
    def data(self) -> Optional[bytes]:
        """Raw payload bytes (if available)."""
        ...

    @property
    def reason(self) -> SkipReason:
        """Why the frame was skipped."""
        ...

    def __repr__(self) -> str: ...

@final
class OrphanFrameOutput:
    """A decoded frame whose ``frame_id`` had no matching submitted
    :class:`VideoFrame`."""

    @property
    def decoded_frame(self) -> DecodedFrame:
        """Decoded frame metadata."""
        ...

    def __repr__(self) -> str: ...

@final
class SourceEosOutput:
    """Logical per-source end-of-stream."""

    @property
    def source_id(self) -> str:
        """Source identifier."""
        ...

    def __repr__(self) -> str: ...

@final
class EventOutput:
    """A GStreamer event captured at the pipeline output."""

    @property
    def summary(self) -> str:
        """Debug summary of the GStreamer event."""
        ...

    def __repr__(self) -> str: ...

@final
class ErrorOutput:
    """An error from the underlying decoder."""

    @property
    def message(self) -> str:
        """Error message."""
        ...

    def __repr__(self) -> str: ...

@final
class RestartedOutput:
    """Aggregate signal emitted when the FlexibleDecoder transparently
    restarted after the underlying NvDecoder worker died (e.g. a watchdog
    trip)."""

    @property
    def source_id(self) -> str:
        """Source id of the FlexibleDecoder that restarted."""
        ...

    @property
    def reason(self) -> str:
        """Human-readable reason for the restart."""
        ...

    @property
    def lost_frames(self) -> int:
        """Number of in-flight frames lost because of the restart.

        Each is also surfaced separately as a :class:`SkippedOutput` with
        :class:`SkipReason` ``DecoderRestarted``.
        """
        ...

    def __repr__(self) -> str: ...

@final
class FlexibleDecoderOutput:
    """Callback payload from :class:`FlexibleDecoder`.

    Use the ``is_*`` properties to determine the variant, then call the
    corresponding ``as_*`` method to get a typed output object.

    For ``Frame`` outputs, call ``as_frame().take_delivery()`` to extract a
    sealed ``(VideoFrame, SharedBuffer)`` pair.  When the :class:`FrameOutput`
    is dropped (garbage-collected), the seal is released and downstream can
    unseal.
    """

    @property
    def is_frame(self) -> bool: ...
    @property
    def is_parameter_change(self) -> bool: ...
    @property
    def is_skipped(self) -> bool: ...
    @property
    def is_orphan_frame(self) -> bool: ...
    @property
    def is_source_eos(self) -> bool: ...
    @property
    def is_event(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
    @property
    def is_restarted(self) -> bool: ...

    def as_frame(self) -> Optional[FrameOutput]:
        """Downcast to :class:`FrameOutput`, or ``None``."""
        ...

    def as_parameter_change(self) -> Optional[ParameterChangeOutput]:
        """Downcast to :class:`ParameterChangeOutput`, or ``None``."""
        ...

    def as_skipped(self) -> Optional[SkippedOutput]:
        """Downcast to :class:`SkippedOutput`, or ``None``."""
        ...

    def as_orphan_frame(self) -> Optional[OrphanFrameOutput]:
        """Downcast to :class:`OrphanFrameOutput`, or ``None``."""
        ...

    def as_source_eos(self) -> Optional[SourceEosOutput]:
        """Downcast to :class:`SourceEosOutput`, or ``None``."""
        ...

    def as_event(self) -> Optional[EventOutput]:
        """Downcast to :class:`EventOutput`, or ``None``."""
        ...

    def as_error(self) -> Optional[ErrorOutput]:
        """Downcast to :class:`ErrorOutput`, or ``None``."""
        ...

    def as_restarted(self) -> Optional[RestartedOutput]:
        """Downcast to :class:`RestartedOutput`, or ``None``."""
        ...

    def __repr__(self) -> str: ...

@final
class FlexibleDecoder:
    """Single-stream adaptive GPU decoder.

    Wraps the Rust ``FlexibleDecoder`` and delivers all output through the
    ``result_callback`` supplied at construction.

    Args:
        config: Decoder configuration.
        result_callback: Called for every decoded frame, parameter change,
            skip, EOS, or error.
    """

    def __init__(
        self,
        config: FlexibleDecoderConfig,
        result_callback: Callable[[FlexibleDecoderOutput], None],
    ) -> None: ...

    def submit(
        self, frame: VideoFrame, data: Optional[bytes] = None
    ) -> None:
        """Submit an encoded frame for decoding.

        Args:
            frame: Video frame with metadata.
            data: Encoded payload.  If ``None``, the frame's internal content
                is used.

        Raises:
            RuntimeError: If shut down or an infrastructure error occurs.
        """
        ...

    def source_eos(self, source_id: str) -> None:
        """Inject a logical per-source EOS.

        Raises:
            RuntimeError: If shut down.
        """
        ...

    def graceful_shutdown(self) -> None:
        """Drain the current decoder and shut down.

        Raises:
            RuntimeError: If already shut down or a drain error occurs.
        """
        ...

    def shutdown(self) -> None:
        """Immediate teardown — frames in flight are lost."""
        ...

    def __repr__(self) -> str: ...


@final
class EvictionDecision:
    """Decision returned by an eviction callback.

    Use the class-level constants :attr:`EVICT` and :attr:`KEEP`.
    """

    EVICT: EvictionDecision
    """Remove the decoder from the pool."""

    KEEP: EvictionDecision
    """Keep the decoder alive (reset TTL)."""

    @property
    def is_evict(self) -> bool:
        """``True`` if this is :attr:`EVICT`."""
        ...

    @property
    def is_keep(self) -> bool:
        """``True`` if this is :attr:`KEEP`."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...


@final
class FlexibleDecoderPoolConfig:
    """Configuration for a :class:`FlexibleDecoderPool`.

    Args:
        gpu_id: GPU device ordinal.
        pool_size: Number of RGBA buffers per internal decoder pool.
        eviction_ttl_ms: Idle stream TTL in milliseconds before eviction
            is considered.
    """

    def __init__(
        self, gpu_id: int, pool_size: int, eviction_ttl_ms: int
    ) -> None: ...

    def with_idle_timeout_ms(self, ms: int) -> FlexibleDecoderPoolConfig:
        """Return a new config with the given idle timeout (milliseconds)."""
        ...

    def with_detect_buffer_limit(self, n: int) -> FlexibleDecoderPoolConfig:
        """Return a new config with the given detection buffer limit."""
        ...

    def with_decoder_config_callback(
        self,
        cb: Callable[[DecoderConfig, VideoFrame], DecoderConfig],
    ) -> FlexibleDecoderPoolConfig:
        """Install a decoder-config transformation callback applied to every
        per-stream decoder created by the pool. See
        :meth:`FlexibleDecoderConfig.with_decoder_config_callback`.
        """
        ...

    @property
    def gpu_id(self) -> int: ...

    @property
    def pool_size(self) -> int: ...

    @property
    def idle_timeout_ms(self) -> int: ...

    @property
    def detect_buffer_limit(self) -> int: ...

    @property
    def eviction_ttl_ms(self) -> int: ...

    def __repr__(self) -> str: ...


@final
class FlexibleDecoderPool:
    """Multi-stream pool of :class:`FlexibleDecoder` instances.

    Routes incoming frames by ``source_id`` to per-stream decoders,
    creating them on demand.  Idle streams are evicted after
    ``eviction_ttl_ms``.

    Args:
        config: Pool configuration.
        result_callback: Called for every decoded output from any stream.
        eviction_callback: Optional.  Called when a stream's TTL expires.
            Return :attr:`EvictionDecision.KEEP` to reset the TTL or
            :attr:`EvictionDecision.EVICT` to remove the stream.
            When ``None``, all expired streams are evicted automatically.
    """

    def __init__(
        self,
        config: FlexibleDecoderPoolConfig,
        result_callback: Callable[[FlexibleDecoderOutput], None],
        eviction_callback: Optional[Callable[[str], EvictionDecision]] = None,
    ) -> None: ...

    def submit(
        self, frame: VideoFrame, data: Optional[bytes] = None
    ) -> None:
        """Submit an encoded frame for decoding.

        The frame is routed to the per-stream decoder for
        ``frame.source_id``.  If none exists, one is created transparently.

        Args:
            frame: Video frame with metadata.
            data: Encoded payload.  If ``None``, the frame's internal content
                is used.

        Raises:
            RuntimeError: If shut down or an infrastructure error occurs.
        """
        ...

    def source_eos(self, source_id: str) -> None:
        """Inject a logical per-source EOS.

        Raises:
            RuntimeError: If shut down.
        """
        ...

    def graceful_shutdown(self) -> None:
        """Drain every decoder in the pool and shut down.

        Raises:
            RuntimeError: If already shut down.
        """
        ...

    def shutdown(self) -> None:
        """Immediate teardown — frames in flight are lost."""
        ...

    def __repr__(self) -> str: ...


# ── DecoderConfig types ─────────────────────────────────────────────────

@final
class H264StreamFormat:
    """H.264 bitstream format carried in the GStreamer caps."""

    BYTE_STREAM: H264StreamFormat
    AVC: H264StreamFormat
    AVC3: H264StreamFormat

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class HevcStreamFormat:
    """HEVC bitstream format carried in the GStreamer caps."""

    BYTE_STREAM: HevcStreamFormat
    HVC1: HevcStreamFormat
    HEV1: HevcStreamFormat

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class JpegBackend:
    """Backend used by the JPEG decoder."""

    GPU: JpegBackend
    CPU: JpegBackend

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class CudadecMemtype:
    """CUDA memory type for the ``nvv4l2decoder`` ``cudadec-memtype`` property.

    **Platform: dGPU only.** This class is not exposed on Jetson
    (``aarch64``); ``nvv4l2decoder`` does not expose the
    ``cudadec-memtype`` property on that platform. Importing the
    symbol on Jetson raises ``ImportError``.
    """

    DEVICE: CudadecMemtype
    PINNED: CudadecMemtype
    UNIFIED: CudadecMemtype

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class H264DecoderConfig:
    """H.264 decoder configuration.

    ``codec_data`` is derived by the pipeline from the bitstream and is
    intentionally not exposed.

    Tunable availability (per DeepStream 7.1 Gst-nvvideo4linux2 docs):

    * ``num_extra_surfaces`` -- both platforms.
    * ``drop_frame_interval`` -- both platforms.
    * ``cudadec_memtype``, ``low_latency_mode`` -- dGPU only.
    * ``enable_max_performance``, ``low_latency`` -- Jetson only.
    """

    def __init__(self, stream_format: H264StreamFormat) -> None: ...
    def with_num_extra_surfaces(self, n: int) -> H264DecoderConfig: ...
    def with_drop_frame_interval(self, n: int) -> H264DecoderConfig: ...
    def with_cudadec_memtype(self, t: CudadecMemtype) -> H264DecoderConfig:
        """**dGPU only** -- maps to ``cudadec-memtype``."""
        ...

    def with_low_latency_mode(self, v: bool) -> H264DecoderConfig:
        """**dGPU only** -- maps to ``low-latency-mode`` (no frame
        reordering; requires IDR-only or low-delay-encoded
        bitstreams).
        """
        ...

    def with_enable_max_performance(self, v: bool) -> H264DecoderConfig:
        """**Jetson only** -- maps to ``enable-max-performance``."""
        ...

    def with_low_latency(self, v: bool) -> H264DecoderConfig:
        """**Jetson only** -- maps to ``disable-dpb`` (low-latency mode
        for IDR-only or IPPP bitstreams).
        """
        ...

    @property
    def stream_format(self) -> H264StreamFormat: ...
    @property
    def num_extra_surfaces(self) -> Optional[int]: ...
    @property
    def drop_frame_interval(self) -> Optional[int]: ...
    @property
    def cudadec_memtype(self) -> Optional[CudadecMemtype]:
        """**dGPU only.**"""
        ...

    @property
    def low_latency_mode(self) -> Optional[bool]:
        """**dGPU only.**"""
        ...

    @property
    def enable_max_performance(self) -> Optional[bool]:
        """**Jetson only.**"""
        ...

    @property
    def low_latency(self) -> Optional[bool]:
        """**Jetson only** -- ``disable-dpb``."""
        ...

    def __repr__(self) -> str: ...


@final
class HevcDecoderConfig:
    """HEVC decoder configuration (``codec_data`` not exposed).

    See :class:`H264DecoderConfig` for the per-platform tunable matrix.
    """

    def __init__(self, stream_format: HevcStreamFormat) -> None: ...
    def with_num_extra_surfaces(self, n: int) -> HevcDecoderConfig: ...
    def with_drop_frame_interval(self, n: int) -> HevcDecoderConfig: ...
    def with_cudadec_memtype(self, t: CudadecMemtype) -> HevcDecoderConfig:
        """**dGPU only** -- maps to ``cudadec-memtype``."""
        ...

    def with_low_latency_mode(self, v: bool) -> HevcDecoderConfig:
        """**dGPU only** -- maps to ``low-latency-mode``."""
        ...

    def with_enable_max_performance(self, v: bool) -> HevcDecoderConfig:
        """**Jetson only** -- maps to ``enable-max-performance``."""
        ...

    def with_low_latency(self, v: bool) -> HevcDecoderConfig:
        """**Jetson only** -- maps to ``disable-dpb``."""
        ...

    @property
    def stream_format(self) -> HevcStreamFormat: ...
    @property
    def num_extra_surfaces(self) -> Optional[int]: ...
    @property
    def drop_frame_interval(self) -> Optional[int]: ...
    @property
    def cudadec_memtype(self) -> Optional[CudadecMemtype]:
        """**dGPU only.**"""
        ...

    @property
    def low_latency_mode(self) -> Optional[bool]:
        """**dGPU only.**"""
        ...

    @property
    def enable_max_performance(self) -> Optional[bool]:
        """**Jetson only.**"""
        ...

    @property
    def low_latency(self) -> Optional[bool]:
        """**Jetson only** -- ``disable-dpb``."""
        ...

    def __repr__(self) -> str: ...


@final
class Vp8DecoderConfig:
    """VP8 decoder configuration.

    See :class:`H264DecoderConfig` for the per-platform tunable matrix.
    """

    def __init__(self) -> None: ...
    def with_num_extra_surfaces(self, n: int) -> Vp8DecoderConfig: ...
    def with_drop_frame_interval(self, n: int) -> Vp8DecoderConfig: ...
    def with_cudadec_memtype(self, t: CudadecMemtype) -> Vp8DecoderConfig:
        """**dGPU only** -- maps to ``cudadec-memtype``."""
        ...

    def with_low_latency_mode(self, v: bool) -> Vp8DecoderConfig:
        """**dGPU only** -- maps to ``low-latency-mode``."""
        ...

    def with_enable_max_performance(self, v: bool) -> Vp8DecoderConfig:
        """**Jetson only** -- maps to ``enable-max-performance``."""
        ...

    def with_low_latency(self, v: bool) -> Vp8DecoderConfig:
        """**Jetson only** -- maps to ``disable-dpb``."""
        ...

    @property
    def num_extra_surfaces(self) -> Optional[int]: ...
    @property
    def drop_frame_interval(self) -> Optional[int]: ...
    @property
    def cudadec_memtype(self) -> Optional[CudadecMemtype]:
        """**dGPU only.**"""
        ...

    @property
    def low_latency_mode(self) -> Optional[bool]:
        """**dGPU only.**"""
        ...

    @property
    def enable_max_performance(self) -> Optional[bool]:
        """**Jetson only.**"""
        ...

    @property
    def low_latency(self) -> Optional[bool]:
        """**Jetson only** -- ``disable-dpb``."""
        ...

    def __repr__(self) -> str: ...


@final
class Vp9DecoderConfig:
    """VP9 decoder configuration.

    See :class:`H264DecoderConfig` for the per-platform tunable matrix.
    """

    def __init__(self) -> None: ...
    def with_num_extra_surfaces(self, n: int) -> Vp9DecoderConfig: ...
    def with_drop_frame_interval(self, n: int) -> Vp9DecoderConfig: ...
    def with_cudadec_memtype(self, t: CudadecMemtype) -> Vp9DecoderConfig:
        """**dGPU only** -- maps to ``cudadec-memtype``."""
        ...

    def with_low_latency_mode(self, v: bool) -> Vp9DecoderConfig:
        """**dGPU only** -- maps to ``low-latency-mode``."""
        ...

    def with_enable_max_performance(self, v: bool) -> Vp9DecoderConfig:
        """**Jetson only** -- maps to ``enable-max-performance``."""
        ...

    def with_low_latency(self, v: bool) -> Vp9DecoderConfig:
        """**Jetson only** -- maps to ``disable-dpb``."""
        ...

    @property
    def num_extra_surfaces(self) -> Optional[int]: ...
    @property
    def drop_frame_interval(self) -> Optional[int]: ...
    @property
    def cudadec_memtype(self) -> Optional[CudadecMemtype]:
        """**dGPU only.**"""
        ...

    @property
    def low_latency_mode(self) -> Optional[bool]:
        """**dGPU only.**"""
        ...

    @property
    def enable_max_performance(self) -> Optional[bool]:
        """**Jetson only.**"""
        ...

    @property
    def low_latency(self) -> Optional[bool]:
        """**Jetson only** -- ``disable-dpb``."""
        ...

    def __repr__(self) -> str: ...


@final
class Av1DecoderConfig:
    """AV1 decoder configuration.

    See :class:`H264DecoderConfig` for the per-platform tunable matrix.
    """

    def __init__(self) -> None: ...
    def with_num_extra_surfaces(self, n: int) -> Av1DecoderConfig: ...
    def with_drop_frame_interval(self, n: int) -> Av1DecoderConfig: ...
    def with_cudadec_memtype(self, t: CudadecMemtype) -> Av1DecoderConfig:
        """**dGPU only** -- maps to ``cudadec-memtype``."""
        ...

    def with_low_latency_mode(self, v: bool) -> Av1DecoderConfig:
        """**dGPU only** -- maps to ``low-latency-mode``."""
        ...

    def with_enable_max_performance(self, v: bool) -> Av1DecoderConfig:
        """**Jetson only** -- maps to ``enable-max-performance``."""
        ...

    def with_low_latency(self, v: bool) -> Av1DecoderConfig:
        """**Jetson only** -- maps to ``disable-dpb``."""
        ...

    @property
    def num_extra_surfaces(self) -> Optional[int]: ...
    @property
    def drop_frame_interval(self) -> Optional[int]: ...
    @property
    def cudadec_memtype(self) -> Optional[CudadecMemtype]:
        """**dGPU only.**"""
        ...

    @property
    def low_latency_mode(self) -> Optional[bool]:
        """**dGPU only.**"""
        ...

    @property
    def enable_max_performance(self) -> Optional[bool]:
        """**Jetson only.**"""
        ...

    @property
    def low_latency(self) -> Optional[bool]:
        """**Jetson only** -- ``disable-dpb``."""
        ...

    def __repr__(self) -> str: ...


@final
class JpegDecoderConfig:
    """JPEG decoder configuration."""

    @staticmethod
    def gpu() -> JpegDecoderConfig:
        """Use the GPU JPEG decoder (``nvjpegdec``)."""
        ...

    @staticmethod
    def cpu() -> JpegDecoderConfig:
        """Use the CPU JPEG decoder (``jpegdec``)."""
        ...

    @property
    def backend(self) -> JpegBackend: ...
    def __repr__(self) -> str: ...


@final
class PngDecoderConfig:
    """PNG decoder configuration (no tunable parameters)."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


@final
class RawRgbaDecoderConfig:
    """Raw RGBA decoder configuration (dimensions from the frame)."""

    def __init__(self, width: int, height: int) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class RawRgbDecoderConfig:
    """Raw RGB decoder configuration (dimensions from the frame)."""

    def __init__(self, width: int, height: int) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def __repr__(self) -> str: ...


@final
class DecoderConfig:
    """Umbrella wrapper for the per-codec decoder configurations.

    Use :meth:`codec` to inspect the variant and ``as_*`` /
    ``with_*`` / ``from_*`` methods to convert between typed inner
    classes and this umbrella type.
    """

    @staticmethod
    def from_h264(cfg: H264DecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_hevc(cfg: HevcDecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_vp8(cfg: Vp8DecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_vp9(cfg: Vp9DecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_av1(cfg: Av1DecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_jpeg(cfg: JpegDecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_png(cfg: PngDecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_raw_rgba(cfg: RawRgbaDecoderConfig) -> DecoderConfig: ...
    @staticmethod
    def from_raw_rgb(cfg: RawRgbDecoderConfig) -> DecoderConfig: ...

    def codec(self) -> Codec: ...
    def as_h264(self) -> Optional[H264DecoderConfig]: ...
    def as_hevc(self) -> Optional[HevcDecoderConfig]: ...
    def as_vp8(self) -> Optional[Vp8DecoderConfig]: ...
    def as_vp9(self) -> Optional[Vp9DecoderConfig]: ...
    def as_av1(self) -> Optional[Av1DecoderConfig]: ...
    def as_jpeg(self) -> Optional[JpegDecoderConfig]: ...
    def as_png(self) -> Optional[PngDecoderConfig]: ...
    def as_raw_rgba(self) -> Optional[RawRgbaDecoderConfig]: ...
    def as_raw_rgb(self) -> Optional[RawRgbDecoderConfig]: ...

    def with_h264(self, cfg: H264DecoderConfig) -> DecoderConfig: ...
    def with_hevc(self, cfg: HevcDecoderConfig) -> DecoderConfig: ...
    def with_vp8(self, cfg: Vp8DecoderConfig) -> DecoderConfig: ...
    def with_vp9(self, cfg: Vp9DecoderConfig) -> DecoderConfig: ...
    def with_av1(self, cfg: Av1DecoderConfig) -> DecoderConfig: ...
    def with_jpeg(self, cfg: JpegDecoderConfig) -> DecoderConfig: ...
    def with_png(self, cfg: PngDecoderConfig) -> DecoderConfig: ...
    def with_raw_rgba(self, cfg: RawRgbaDecoderConfig) -> DecoderConfig: ...
    def with_raw_rgb(self, cfg: RawRgbDecoderConfig) -> DecoderConfig: ...

    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Encoder enums
#
# Symmetric counterparts to the decoder enums above. Use these to assemble
# per-codec encoder properties (`H264DgpuProps`, `JpegProps`, etc.) which
# are then wrapped in `EncoderProperties` and attached to `EncoderConfig`.
# ═══════════════════════════════════════════════════════════════════════════

@final
class Platform:
    DGPU: Platform
    JETSON: Platform

    @staticmethod
    def from_name(name: str) -> Platform: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class RateControl:
    VARIABLE_BITRATE: RateControl
    CONSTANT_BITRATE: RateControl
    CONSTANT_QP: RateControl

    @staticmethod
    def from_name(name: str) -> RateControl: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class H264Profile:
    BASELINE: H264Profile
    MAIN: H264Profile
    HIGH: H264Profile
    HIGH444: H264Profile

    @staticmethod
    def from_name(name: str) -> H264Profile: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class HevcProfile:
    MAIN: HevcProfile
    MAIN10: HevcProfile
    FREXT: HevcProfile

    @staticmethod
    def from_name(name: str) -> HevcProfile: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class DgpuPreset:
    P1: DgpuPreset
    P2: DgpuPreset
    P3: DgpuPreset
    P4: DgpuPreset
    P5: DgpuPreset
    P6: DgpuPreset
    P7: DgpuPreset

    @staticmethod
    def from_name(name: str) -> DgpuPreset: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class TuningPreset:
    HIGH_QUALITY: TuningPreset
    LOW_LATENCY: TuningPreset
    ULTRA_LOW_LATENCY: TuningPreset
    LOSSLESS: TuningPreset

    @staticmethod
    def from_name(name: str) -> TuningPreset: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class JetsonPresetLevel:
    DISABLED: JetsonPresetLevel
    ULTRA_FAST: JetsonPresetLevel
    FAST: JetsonPresetLevel
    MEDIUM: JetsonPresetLevel
    SLOW: JetsonPresetLevel

    @staticmethod
    def from_name(name: str) -> JetsonPresetLevel: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# Per-codec encoder property structs
# ═══════════════════════════════════════════════════════════════════════════

class H264DgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[H264Profile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    const_qp: Optional[str]
    init_qp: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]
    extended_colorformat: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[H264Profile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
        qp_range: Optional[str] = None,
        const_qp: Optional[str] = None,
        init_qp: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
        extended_colorformat: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class HevcDgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[HevcProfile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    const_qp: Optional[str]
    init_qp: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]
    extended_colorformat: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[HevcProfile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
        qp_range: Optional[str] = None,
        const_qp: Optional[str] = None,
        init_qp: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
        extended_colorformat: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class H264JetsonProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[H264Profile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset_level: Optional[JetsonPresetLevel]
    peak_bitrate: Optional[int]
    vbv_size: Optional[int]
    qp_range: Optional[str]
    quant_i_frames: Optional[int]
    quant_p_frames: Optional[int]
    ratecontrol_enable: Optional[bool]
    maxperf_enable: Optional[bool]
    two_pass_cbr: Optional[bool]
    num_ref_frames: Optional[int]
    insert_sps_pps: Optional[bool]
    insert_aud: Optional[bool]
    insert_vui: Optional[bool]
    disable_cabac: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[H264Profile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[JetsonPresetLevel] = None,
        peak_bitrate: Optional[int] = None,
        vbv_size: Optional[int] = None,
        qp_range: Optional[str] = None,
        quant_i_frames: Optional[int] = None,
        quant_p_frames: Optional[int] = None,
        ratecontrol_enable: Optional[bool] = None,
        maxperf_enable: Optional[bool] = None,
        two_pass_cbr: Optional[bool] = None,
        num_ref_frames: Optional[int] = None,
        insert_sps_pps: Optional[bool] = None,
        insert_aud: Optional[bool] = None,
        insert_vui: Optional[bool] = None,
        disable_cabac: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class HevcJetsonProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    profile: Optional[HevcProfile]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset_level: Optional[JetsonPresetLevel]
    peak_bitrate: Optional[int]
    vbv_size: Optional[int]
    qp_range: Optional[str]
    quant_i_frames: Optional[int]
    quant_p_frames: Optional[int]
    ratecontrol_enable: Optional[bool]
    maxperf_enable: Optional[bool]
    two_pass_cbr: Optional[bool]
    num_ref_frames: Optional[int]
    enable_lossless: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        profile: Optional[HevcProfile] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[JetsonPresetLevel] = None,
        peak_bitrate: Optional[int] = None,
        vbv_size: Optional[int] = None,
        qp_range: Optional[str] = None,
        quant_i_frames: Optional[int] = None,
        quant_p_frames: Optional[int] = None,
        ratecontrol_enable: Optional[bool] = None,
        maxperf_enable: Optional[bool] = None,
        two_pass_cbr: Optional[bool] = None,
        num_ref_frames: Optional[int] = None,
        enable_lossless: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class JpegProps:
    quality: Optional[int]

    def __init__(self, quality: Optional[int] = None) -> None: ...
    def __repr__(self) -> str: ...

class PngProps:
    """PNG encoder properties (CPU-based, ``pngenc`` from gst-plugins-good)."""

    compression_level: Optional[int]

    def __init__(self, compression_level: Optional[int] = None) -> None: ...
    def __repr__(self) -> str: ...

class Av1DgpuProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset: Optional[DgpuPreset]
    tuning_info: Optional[TuningPreset]
    qp_range: Optional[str]
    max_bitrate: Optional[int]
    vbv_buf_size: Optional[int]
    vbv_init: Optional[int]
    cq: Optional[int]
    aq: Optional[int]
    temporal_aq: Optional[bool]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset: Optional[DgpuPreset] = None,
        tuning_info: Optional[TuningPreset] = None,
        qp_range: Optional[str] = None,
        max_bitrate: Optional[int] = None,
        vbv_buf_size: Optional[int] = None,
        vbv_init: Optional[int] = None,
        cq: Optional[int] = None,
        aq: Optional[int] = None,
        temporal_aq: Optional[bool] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Av1JetsonProps:
    bitrate: Optional[int]
    control_rate: Optional[RateControl]
    iframeinterval: Optional[int]
    idrinterval: Optional[int]
    preset_level: Optional[JetsonPresetLevel]
    peak_bitrate: Optional[int]
    vbv_size: Optional[int]
    qp_range: Optional[str]
    quant_i_frames: Optional[int]
    quant_p_frames: Optional[int]
    quant_b_frames: Optional[int]
    ratecontrol_enable: Optional[bool]
    maxperf_enable: Optional[bool]
    two_pass_cbr: Optional[bool]
    num_ref_frames: Optional[int]
    insert_seq_hdr: Optional[bool]
    tiles: Optional[str]

    def __init__(
        self,
        bitrate: Optional[int] = None,
        control_rate: Optional[RateControl] = None,
        iframeinterval: Optional[int] = None,
        idrinterval: Optional[int] = None,
        preset_level: Optional[JetsonPresetLevel] = None,
        peak_bitrate: Optional[int] = None,
        vbv_size: Optional[int] = None,
        qp_range: Optional[str] = None,
        quant_i_frames: Optional[int] = None,
        quant_p_frames: Optional[int] = None,
        quant_b_frames: Optional[int] = None,
        ratecontrol_enable: Optional[bool] = None,
        maxperf_enable: Optional[bool] = None,
        two_pass_cbr: Optional[bool] = None,
        num_ref_frames: Optional[int] = None,
        insert_seq_hdr: Optional[bool] = None,
        tiles: Optional[str] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# EncoderProperties (tagged union via factory statics)
# ═══════════════════════════════════════════════════════════════════════════

class EncoderProperties:
    @staticmethod
    def h264_dgpu(props: H264DgpuProps) -> EncoderProperties: ...
    @staticmethod
    def h264_jetson(props: H264JetsonProps) -> EncoderProperties: ...
    @staticmethod
    def hevc_dgpu(props: HevcDgpuProps) -> EncoderProperties: ...
    @staticmethod
    def hevc_jetson(props: HevcJetsonProps) -> EncoderProperties: ...
    @staticmethod
    def jpeg(props: JpegProps) -> EncoderProperties: ...
    @staticmethod
    def av1_dgpu(props: Av1DgpuProps) -> EncoderProperties: ...
    @staticmethod
    def av1_jetson(props: Av1JetsonProps) -> EncoderProperties: ...
    @staticmethod
    def png(props: PngProps) -> EncoderProperties: ...
    def __repr__(self) -> str: ...

# ═══════════════════════════════════════════════════════════════════════════
# EncoderConfig
# ═══════════════════════════════════════════════════════════════════════════

class EncoderConfig:
    """Encoder configuration for the Picasso/NvEncoder pipeline.

    Uses a builder pattern -- chain ``.format(...).fps(...).gpu_id(...)``
    etc. after construction.

    **Important:** The builder methods ``format()``, ``gpu_id()`` shadow
    the underlying property setters. At runtime, property assignment
    (``cfg.gpu_id = 0``) raises ``AttributeError: read-only``.
    Always use the builder method call form::

        cfg = EncoderConfig(Codec.H264, 1280, 720)
        cfg.format(VideoFormat.RGBA)   # builder call -- OK
        cfg.gpu_id(0)                  # builder call -- OK
        cfg.fps(30, 1)
        cfg.properties(props)
    """

    def __init__(self, codec: Codec, width: int, height: int) -> None: ...

    # ── read-only property getters ──
    # Note: ``format`` and ``gpu_id`` have builder methods with the same
    # name that shadow the property setter; use the builder call form.
    @property
    def fps_num(self) -> int: ...
    @property
    def fps_den(self) -> int: ...
    @property
    def mem_type(self) -> MemType: ...
    @property
    def encoder_params(self) -> Optional[EncoderProperties]: ...

    # ── builder methods (return self for chaining) ──
    def format(self, fmt: VideoFormat) -> EncoderConfig: ...
    def fps(self, num: int, den: int) -> EncoderConfig: ...
    def gpu_id(self, id: int) -> EncoderConfig: ...
    def properties(self, props: EncoderProperties) -> EncoderConfig: ...
    def __repr__(self) -> str: ...
