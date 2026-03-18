"""Type stubs for ``savant_rs.nvinfer`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union, final

from savant_rs.deepstream import SharedBuffer
from savant_rs.primitives.geometry import RBBox

__all__ = [
    "MetaClearPolicy",
    "DataType",
    "Roi",
    "NvInferConfig",
    "InferDims",
    "TensorView",
    "ElementOutput",
    "BatchInferenceOutput",
    "NvInfer",
]

# ── Enums ────────────────────────────────────────────────────────────────

@final
class MetaClearPolicy:
    """Controls when object metadata is erased from the batch buffer.

    - ``NONE`` -- never clear automatically.
    - ``BEFORE`` -- clear stale objects before attaching ROI objects (default).
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

@final
class DataType:
    """Data type of a tensor element.

    - ``FLOAT`` -- 32-bit floating point (4 bytes).
    - ``HALF`` -- 16-bit floating point (2 bytes).
    - ``INT8`` -- 8-bit signed integer (1 byte).
    - ``INT32`` -- 32-bit signed integer (4 bytes).
    """

    FLOAT: DataType
    HALF: DataType
    INT8: DataType
    INT32: DataType

    def element_size(self) -> int:
        """Size in bytes of a single element of this type."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

# ── ROI ──────────────────────────────────────────────────────────────────

@final
class Roi:
    """A region of interest: an identifier paired with a bounding box.

    Args:
        id: Caller-defined identifier returned in ``ElementOutput.roi_id``.
        bbox: Bounding box (center-based, optionally rotated).
    """

    def __init__(self, id: int, bbox: RBBox) -> None: ...

    @property
    def id(self) -> int:
        """Caller-defined identifier."""
        ...

    @property
    def bbox(self) -> RBBox:
        """Bounding box (center-based, optionally rotated)."""
        ...

    def __repr__(self) -> str: ...

# ── Configuration ────────────────────────────────────────────────────────

@final
class NvInferConfig:
    """Configuration for the NvInfer pipeline.

    Args:
        nvinfer_properties: NvInfer config keys.  Use dotted notation
            ``section.key`` for per-class sections.  Bare keys go to
            ``[property]``.
        input_format: Input format for appsrc caps (e.g. ``"RGBA"``).
        input_width: Input width for appsrc caps.
        input_height: Input height for appsrc caps.
        name: Optional instance name for logging.
        element_properties: Additional GStreamer element properties.
        gpu_id: GPU device ID.
        queue_depth: GStreamer queue max-size-buffers (0 = no queue).
        meta_clear_policy: When to clear object metadata.
    """

    def __init__(
        self,
        nvinfer_properties: Dict[str, str],
        input_format: str,
        input_width: int,
        input_height: int,
        *,
        name: str = "",
        element_properties: Optional[Dict[str, str]] = None,
        gpu_id: int = 0,
        queue_depth: int = 0,
        meta_clear_policy: MetaClearPolicy = MetaClearPolicy.BEFORE,
    ) -> None: ...

    @staticmethod
    def new_flexible(
        nvinfer_properties: Dict[str, str],
        input_format: str,
        *,
        name: str = "",
        element_properties: Optional[Dict[str, str]] = None,
        gpu_id: int = 0,
        queue_depth: int = 0,
        meta_clear_policy: MetaClearPolicy = MetaClearPolicy.BEFORE,
    ) -> NvInferConfig:
        """Create a config without fixed input dimensions.

        Required for non-uniform batches where each frame may have a
        different resolution.
        """
        ...

    @property
    def name(self) -> str: ...
    @property
    def gpu_id(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def input_format(self) -> str: ...
    @property
    def input_width(self) -> Optional[int]: ...
    @property
    def input_height(self) -> Optional[int]: ...
    @property
    def meta_clear_policy(self) -> MetaClearPolicy: ...

    def __repr__(self) -> str: ...

# ── Output types ─────────────────────────────────────────────────────────

@final
class InferDims:
    """Tensor dimensions and total element count."""

    @property
    def dimensions(self) -> List[int]:
        """Shape along each axis."""
        ...

    @property
    def num_elements(self) -> int:
        """Total number of elements (product of dimensions)."""
        ...

    def __repr__(self) -> str: ...

@final
class TensorView:
    """Zero-copy view into a single output tensor.

    Exposes ``host_ptr`` and ``device_ptr`` as plain integer addresses so
    that Python callers can construct framework-native tensors (NumPy via
    ``ctypes``, CuPy, PyTorch) without any data copy on the Rust side.

    Valid while the parent ``BatchInferenceOutput`` is alive.
    """

    @property
    def name(self) -> str:
        """Output layer name."""
        ...

    @property
    def dims(self) -> InferDims:
        """Tensor dimensions."""
        ...

    @property
    def data_type(self) -> DataType:
        """Data type of tensor elements."""
        ...

    @property
    def byte_length(self) -> int:
        """Byte length of the tensor."""
        ...

    @property
    def host_ptr(self) -> int:
        """Host (CPU) memory address of the tensor data, or 0 if unavailable."""
        ...

    @property
    def device_ptr(self) -> int:
        """Device (GPU) memory address of the tensor data, or 0 if unavailable."""
        ...

    @property
    def numpy_dtype(self) -> str:
        """NumPy-compatible dtype string (``"float32"``, ``"float16"``,
        ``"int8"``, ``"int32"``)."""
        ...

    def __repr__(self) -> str: ...

@final
class ElementOutput:
    """Per-element inference output for one ROI in one frame."""

    @property
    def frame_id(self) -> Optional[int]:
        """User-provided frame ID (if present)."""
        ...

    @property
    def roi_id(self) -> Optional[int]:
        """ROI identifier from ``Roi.id``.  ``None`` when the full frame was used."""
        ...

    @property
    def tensors(self) -> List[TensorView]:
        """Output tensors for this element."""
        ...

    def __repr__(self) -> str: ...

@final
class BatchInferenceOutput:
    """Owns the GStreamer sample and exposes per-ROI inference outputs.

    Tensor data remains valid as long as this object (or any child
    ``TensorView``) is alive.
    """

    @property
    def batch_id(self) -> int:
        """User-provided batch ID."""
        ...

    @property
    def num_elements(self) -> int:
        """Number of elements in the batch."""
        ...

    @property
    def elements(self) -> List[ElementOutput]:
        """Per-element outputs (one per ROI per frame)."""
        ...

    def buffer(self) -> SharedBuffer:
        """Get the output GStreamer buffer.

        Returns:
            SharedBuffer for the inference output.
        """
        ...

    def __repr__(self) -> str: ...

# ── Engine ───────────────────────────────────────────────────────────────

@final
class NvInfer:
    """NvInfer inference engine.

    Wraps a DeepStream ``nvinfer`` element in an ``appsrc -> [queue] ->
    nvinfer -> appsink`` GStreamer pipeline.

    Args:
        config: Engine configuration.
        callback: Callback invoked when asynchronous inference completes.
    """

    def __init__(
        self,
        config: NvInferConfig,
        callback: Callable[[BatchInferenceOutput], None],
    ) -> None: ...

    def submit(
        self,
        batch: Union[SharedBuffer, int],
        batch_id: int,
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> None:
        """Submit a batched buffer for asynchronous inference.

        Args:
            batch: Batched NvBufSurface buffer.
            batch_id: User-chosen identifier (must not be ``2**64 - 1``).
            rois: Per-slot ROI lists.
        """
        ...

    def infer_sync(
        self,
        batch: Union[SharedBuffer, int],
        batch_id: int,
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> BatchInferenceOutput:
        """Synchronous inference -- blocks until results arrive (up to 30 s).

        Args:
            batch: Batched NvBufSurface buffer.
            batch_id: User-chosen identifier (must not be ``2**64 - 1``).
            rois: Per-slot ROI lists.

        Returns:
            Inference results.
        """
        ...

    def shutdown(self) -> None:
        """Graceful shutdown: send EOS, drain, stop pipeline."""
        ...

    def __repr__(self) -> str: ...
