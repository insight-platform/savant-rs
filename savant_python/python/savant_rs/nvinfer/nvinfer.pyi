"""Type stubs for ``savant_rs.nvinfer`` submodule.

Only available when ``savant_rs`` is built with the ``deepstream`` Cargo feature.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union, final

import numpy as np
from numpy.typing import NDArray
from savant_rs.deepstream import SharedBuffer, SavantIdMetaKind, VideoFormat
from savant_rs.primitives import VideoFrame
from savant_rs.primitives.geometry import RBBox

__all__ = [
    "MetaClearPolicy",
    "ModelInputScaling",
    "DataType",
    "Roi",
    "RoiKind",
    "NvInferConfig",
    "InferDims",
    "TensorView",
    "ElementOutput",
    "BatchInferenceOutput",
    "NvInferOutput",
    "NvInfer",
    "NvInferBatchingOperatorConfig",
    "BatchFormationResult",
    "OperatorTensorView",
    "OperatorElementOutput",
    "OperatorFrameOutput",
    "SealedDeliveries",
    "OperatorOutput",
    "OperatorInferenceOutput",
    "NvInferBatchingOperator",
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
class ModelInputScaling:
    """How input frames are scaled to the model's fixed input dimensions.

    Maps to nvinfer ``maintain-aspect-ratio`` / ``symmetric-padding``; do not
    set those keys in ``nvinfer_properties``.

    - ``FILL`` -- stretch to model input (default).
    - ``KEEP_ASPECT_RATIO`` -- preserve aspect ratio, padding on the right/bottom.
    - ``KEEP_ASPECT_RATIO_SYMMETRIC`` -- preserve aspect ratio, symmetric padding.
    """

    FILL: ModelInputScaling
    KEEP_ASPECT_RATIO: ModelInputScaling
    KEEP_ASPECT_RATIO_SYMMETRIC: ModelInputScaling

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class ModelColorFormat:
    """Color space the model expects for its input tensor.

    Maps to nvinfer ``model-color-format``; do not set that key in
    ``nvinfer_properties``.

    - ``RGB``  -- 3-channel RGB input (default).
    - ``BGR``  -- 3-channel BGR input.
    - ``GRAY`` -- single-channel grayscale input.
    """

    RGB: ModelColorFormat
    BGR: ModelColorFormat
    GRAY: ModelColorFormat

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

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

@final
class RoiKind:
    """Per-frame ROI specification for the batching operator.

    Use ``RoiKind.full_frame()`` when the entire frame should be inferred,
    or ``RoiKind.rois(list_of_roi)`` when specific regions are provided.
    """

    @staticmethod
    def full_frame() -> RoiKind:
        """Create a ``FullFrame`` variant -- infer on the whole frame."""
        ...

    @staticmethod
    def rois(rois: List[Roi]) -> RoiKind:
        """Create a ``Rois`` variant with a list of :class:`Roi`."""
        ...

    @property
    def is_full_frame(self) -> bool:
        """``True`` when this is the ``FullFrame`` variant."""
        ...

    def get_rois(self) -> List[Roi]:
        """Return the ROI list (empty for ``FullFrame``)."""
        ...

    def __repr__(self) -> str: ...

# ── Configuration ────────────────────────────────────────────────────────

@final
class NvInferConfig:
    """Configuration for the NvInfer pipeline.

    Args:
        nvinfer_properties: NvInfer config keys.  Use dotted notation
            ``section.key`` for per-class sections.  Bare keys go to
            ``[property]``.  ``infer-dims``, ``model-color-format`` must
            **not** be set here; they are auto-injected.
        input_format: Pixel format for appsrc caps.
        model_width: Model input tensor width in pixels.
        model_height: Model input tensor height in pixels.
        model_color_format: Model input color space.
        name: Optional instance name for logging.
        element_properties: Additional GStreamer element properties.
        gpu_id: GPU device ID.
        queue_depth: GStreamer queue max-size-buffers (0 = no queue).
        input_channel_capacity: Bounded input channel size; when full,
            ``NvInfer.submit`` blocks. Default: ``16``.
        output_channel_capacity: Bounded output channel size. Default: ``16``.
        drain_poll_interval_ms: How often the framework drain thread polls
            appsink. Default: ``100``.
        meta_clear_policy: When to clear object metadata.
        disable_output_host_copy: When ``True``, skip device-to-host
            copy of output tensors. Default: ``False``.
        scaling: How frames are scaled to model input size.
        operation_timeout_ms: Framework watchdog / in-flight deadline (ms).
            On timeout, the pipeline enters a terminal failed state.
            Default: ``30000``.
    """

    def __init__(
        self,
        nvinfer_properties: Dict[str, str],
        input_format: VideoFormat,
        model_width: int,
        model_height: int,
        *,
        model_color_format: ModelColorFormat = ModelColorFormat.RGB,
        name: str = "",
        element_properties: Optional[Dict[str, str]] = None,
        gpu_id: int = 0,
        queue_depth: int = 0,
        input_channel_capacity: int = 16,
        output_channel_capacity: int = 16,
        drain_poll_interval_ms: int = 100,
        meta_clear_policy: MetaClearPolicy = MetaClearPolicy.BEFORE,
        disable_output_host_copy: bool = False,
        scaling: ModelInputScaling = ModelInputScaling.FILL,
        operation_timeout_ms: int = 30000,
    ) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def gpu_id(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def input_channel_capacity(self) -> int: ...
    @property
    def output_channel_capacity(self) -> int: ...
    @property
    def drain_poll_interval_ms(self) -> int: ...
    @property
    def nvinfer_properties(self) -> Dict[str, str]: ...
    @property
    def element_properties(self) -> Dict[str, str]: ...
    def model_input_dimensions(self) -> Tuple[int, int]: ...
    @property
    def input_format(self) -> VideoFormat: ...
    @property
    def model_width(self) -> int: ...
    @property
    def model_height(self) -> int: ...
    @property
    def model_color_format(self) -> ModelColorFormat: ...
    @property
    def meta_clear_policy(self) -> MetaClearPolicy: ...
    @property
    def disable_output_host_copy(self) -> bool:
        """Whether the device-to-host copy of output tensors is disabled."""
        ...

    @property
    def scaling(self) -> ModelInputScaling:
        """How input frames are scaled to the model input size."""
        ...

    @property
    def operation_timeout_ms(self) -> int:
        """Operation timeout in milliseconds."""
        ...

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
    def has_host_data(self) -> bool:
        """Whether host (CPU) tensor data is valid.

        Returns ``False`` when ``disable_output_host_copy`` was set on the
        config, meaning only ``device_ptr`` is usable.
        """
        ...

    @property
    def numpy_dtype(self) -> str:
        """NumPy-compatible dtype string (``"float32"``, ``"float16"``,
        ``"int8"``, ``"int32"``)."""
        ...

    def as_numpy(self) -> np.ndarray:
        """Return tensor data as a NumPy array (zero-copy view).

        The returned array shares memory with the inference output buffer;
        it is valid as long as the parent ``BatchInferenceOutput`` is alive.

        Raises:
            RuntimeError: If host data is unavailable (``has_host_data`` is
                ``False``) or the host pointer is null.
        """
        ...

    def __repr__(self) -> str: ...

@final
class ElementOutput:
    """Per-element inference output for one ROI in one frame.

    User frame ids are on ``BatchInferenceOutput.buffer()`` (see
    ``SharedBuffer.savant_ids()``), not on this object.
    """

    @property
    def roi_id(self) -> Optional[int]:
        """ROI identifier from ``Roi.id``.  ``None`` when the full frame was used."""
        ...

    @property
    def slot_number(self) -> int:
        """DeepStream surface slot index (``NvDsFrameMeta.batch_id``)."""
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
    def has_host_data(self) -> bool:
        """Whether host (CPU) tensor data is valid for all tensors.

        Returns ``False`` when ``disable_output_host_copy`` was set on the
        config, meaning only device (GPU) pointers in ``TensorView`` are usable.
        """
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

@final
class NvInferOutput:
    """One item from :meth:`NvInfer.recv` / :meth:`NvInfer.recv_timeout` / :meth:`NvInfer.try_recv`.

    Discriminate with ``is_inference``, ``is_event``, ``is_eos``, ``is_error`` and read
    ``as_inference()``, ``event_summary``, ``eos_source_id``, ``error_message``.
    """

    @property
    def is_inference(self) -> bool: ...
    @property
    def is_event(self) -> bool: ...
    @property
    def is_eos(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
    def as_inference(self) -> Optional[BatchInferenceOutput]: ...
    @property
    def event_summary(self) -> Optional[str]: ...
    @property
    def eos_source_id(self) -> Optional[str]: ...
    @property
    def error_message(self) -> Optional[str]: ...
    def __repr__(self) -> str: ...

# ── Engine ───────────────────────────────────────────────────────────────

@final
class NvInfer:
    """NvInfer inference engine.

    Wraps a DeepStream ``nvinfer`` element in an ``appsrc -> [queue] ->
    nvinfer -> appsink`` GStreamer pipeline.  Outputs are **pulled** with
    :meth:`recv`, :meth:`recv_timeout`, or :meth:`try_recv` (matches the Rust API).

    Args:
        config: Engine configuration.
    """

    def __init__(self, config: NvInferConfig) -> None: ...
    def submit(
        self,
        batch: Union[SharedBuffer, int],
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> None:
        """Submit a batched buffer for inference.

        Args:
            batch: Batched NvBufSurface buffer.
            rois: Per-slot ROI lists.
        """
        ...

    def recv(self) -> NvInferOutput:
        """Block until the next output item (inference, event, EOS, or error).

        Raises:
            RuntimeError: On channel disconnect.
        """
        ...

    def recv_timeout(self, timeout_ms: int) -> Optional[NvInferOutput]:
        """Block until the next output or ``timeout_ms`` elapses.

        Returns:
            ``None`` on timeout.
        """
        ...

    def try_recv(self) -> Optional[NvInferOutput]:
        """Non-blocking peek for the next output; ``None`` if none ready."""
        ...

    def send_eos(self, source_id: str) -> None:
        """Send logical per-source EOS (custom downstream event)."""
        ...

    def send_custom_downstream_event(
        self,
        structure_name: str,
        string_fields: Optional[Dict[str, str]] = None,
    ) -> None:
        """Send a custom downstream GStreamer event (string fields only).

        Wraps Rust ``NvInfer::send_event`` for ``CustomDownstream`` structures.
        Other ``gst::Event`` types require the Rust API.
        """
        ...

    def is_failed(self) -> bool:
        """Whether the pipeline is in a terminal failed state."""
        ...

    def shutdown(self) -> None:
        """Graceful shutdown: send EOS, drain, stop pipeline."""
        ...

    def __repr__(self) -> str: ...

# ── Batching operator layer ──────────────────────────────────────────────

@final
class NvInferBatchingOperatorConfig:
    """Configuration for the NvInferBatchingOperator batching layer.

    Embeds a full ``NvInferConfig`` which is forwarded to the inner NvInfer
    pipeline.  The GPU device ID for batch construction is taken from
    ``nvinfer_config.gpu_id``.

    Args:
        max_batch_size: Maximum batch size; triggers inference when reached.
        max_batch_wait_ms: Maximum time in milliseconds to wait before
            submitting a partial batch.
        nvinfer_config: Configuration forwarded to the inner NvInfer engine.
        pending_batch_timeout_ms: Maximum time (ms) a submitted batch can
            remain pending. On timeout, the operator enters a terminal
            failed state. Default: ``60000``.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_batch_wait_ms: int,
        nvinfer_config: NvInferConfig,
        *,
        pending_batch_timeout_ms: int = 60000,
    ) -> None: ...
    @property
    def max_batch_size(self) -> int: ...
    @property
    def max_batch_wait_ms(self) -> int: ...
    @property
    def pending_batch_timeout_ms(self) -> int:
        """Pending batch timeout in milliseconds."""
        ...
    @property
    def nvinfer_config(self) -> NvInferConfig:
        """The embedded NvInfer engine configuration."""
        ...

    def __repr__(self) -> str: ...

@final
class BatchFormationResult:
    """Result returned by the batch formation callback.

    Args:
        ids: Per-frame Savant IDs as ``(SavantIdMetaKind, int)`` pairs.
        rois: Per-frame ROI specification.
    """

    def __init__(
        self,
        ids: List[Tuple[SavantIdMetaKind, int]],
        rois: List[RoiKind],
    ) -> None: ...
    def __repr__(self) -> str: ...

@final
class OperatorTensorView:
    """Zero-copy view into a single output tensor from an operator slot.

    Exposes ``host_ptr`` and ``device_ptr`` as plain integer addresses so
    that Python callers can construct framework-native tensors (NumPy via
    ``ctypes``, CuPy, PyTorch) without any data copy on the Rust side.

    Valid while the parent ``OperatorInferenceOutput`` is alive.
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
    def has_host_data(self) -> bool:
        """Whether host (CPU) tensor data is valid.

        Returns ``False`` when ``disable_output_host_copy`` was set on the
        config, meaning only ``device_ptr`` is usable.
        """
        ...

    @property
    def numpy_dtype(self) -> str:
        """NumPy-compatible dtype string (``"float32"``, ``"float16"``,
        ``"int8"``, ``"int32"``)."""
        ...

    def as_numpy(self) -> np.ndarray:
        """Return tensor data as a NumPy array (zero-copy view).

        The returned array shares memory with the inference output buffer;
        it is valid as long as the parent ``OperatorInferenceOutput`` is alive.

        Raises:
            RuntimeError: If host data is unavailable (``has_host_data`` is
                ``False``) or the host pointer is null.
        """
        ...

    def __repr__(self) -> str: ...

@final
class OperatorElementOutput:
    """Per-element inference output for one ROI in one operator frame.

    Provides access to output tensors and coordinate scaling methods that
    transform model-space predictions back to absolute frame coordinates.
    The coordinate scaler is lazily initialized on first use.

    Tensor pointers are valid while the parent ``OperatorInferenceOutput``
    (or any sibling ``OperatorTensorView``) is alive.
    """

    @property
    def roi_id(self) -> Optional[int]:
        """ROI identifier from ``Roi.id``.  ``None`` when the full frame was used."""
        ...

    @property
    def slot_number(self) -> int:
        """DeepStream surface slot index (``NvDsFrameMeta.batch_id``)."""
        ...

    @property
    def tensors(self) -> List[OperatorTensorView]:
        """Output tensors for this element."""
        ...

    def scale_points(
        self,
        data: Union[NDArray[np.float32], List[Tuple[float, float]]],
    ) -> NDArray[np.float32]:
        """Transform points from model-input space to absolute frame coordinates.

        Args:
            data: ``ndarray[float32]`` of shape ``(N, 2)`` **or**
                ``list[tuple[float, float]]`` of ``(x, y)`` points.

        Returns:
            ``ndarray[float32]`` of shape ``(N, 2)``.
        """
        ...

    def scale_ltwh(
        self,
        data: Union[NDArray[np.float32], List[Tuple[float, float, float, float]]],
    ) -> NDArray[np.float32]:
        """Transform axis-aligned boxes ``(left, top, width, height)`` from
        model-input space to absolute frame coordinates.

        Args:
            data: ``ndarray[float32]`` of shape ``(N, 4)`` **or**
                ``list[tuple[float, float, float, float]]``.

        Returns:
            ``ndarray[float32]`` of shape ``(N, 4)``.
        """
        ...

    def scale_ltrb(
        self,
        data: Union[NDArray[np.float32], List[Tuple[float, float, float, float]]],
    ) -> NDArray[np.float32]:
        """Transform axis-aligned boxes ``(left, top, right, bottom)`` from
        model-input space to absolute frame coordinates.

        Args:
            data: ``ndarray[float32]`` of shape ``(N, 4)`` **or**
                ``list[tuple[float, float, float, float]]``.

        Returns:
            ``ndarray[float32]`` of shape ``(N, 4)``.
        """
        ...

    def scale_rbboxes(self, boxes: List[RBBox]) -> List[RBBox]:
        """Transform rotated bounding boxes from model-input space to absolute
        frame coordinates.

        Args:
            boxes: ``list[RBBox]``.

        Returns:
            ``list[RBBox]``.
        """
        ...

    def __repr__(self) -> str: ...

@final
class OperatorFrameOutput:
    """Per-frame inference result (callback view — no buffer access).

    The per-frame buffer is held internally and only accessible after
    calling ``OperatorInferenceOutput.take_deliveries()`` and then
    ``SealedDeliveries.unseal()``.

    Tensor data remains valid as long as the parent
    ``OperatorInferenceOutput`` is alive.
    """

    @property
    def frame(self) -> VideoFrame:
        """The original VideoFrame submitted for this frame."""
        ...

    @property
    def elements(self) -> List[OperatorElementOutput]:
        """Inference results for this frame."""
        ...

    def __repr__(self) -> str: ...

@final
class SealedDeliveries:
    """A batch of ``(VideoFrame, SharedBuffer)`` pairs sealed until the
    associated ``OperatorInferenceOutput`` is dropped.

    Individual buffers are inaccessible while sealed.  Call :meth:`unseal`
    (blocking) or :meth:`try_unseal` (non-blocking) to obtain the pairs.

    **Drop safety**: dropping ``SealedDeliveries`` without calling
    ``unseal()`` is safe — contained buffers are freed and no deadlock
    can occur.
    """

    def __len__(self) -> int:
        """Number of frames in the sealed batch."""
        ...

    def is_empty(self) -> bool:
        """Whether the batch is empty."""
        ...

    def is_released(self) -> bool:
        """Whether the seal has been released (non-blocking check).

        Returns ``True`` once the ``OperatorInferenceOutput`` has been
        dropped.
        """
        ...

    def unseal(
        self, timeout_ms: Optional[int] = None
    ) -> List[Tuple[VideoFrame, SharedBuffer]]:
        """Block until the ``OperatorInferenceOutput`` is dropped, then
        return all deliveries as ``list[tuple[VideoFrame, SharedBuffer]]``.

        The GIL is released during the blocking wait so the callback
        thread (which needs the GIL to drop the output) can proceed.

        Args:
            timeout_ms: Optional timeout in milliseconds.  When ``None``
                (default), blocks indefinitely.  When the timeout expires,
                raises ``TimeoutError``.

        Raises:
            RuntimeError: If already consumed by a previous call.
            TimeoutError: If the timeout expires before the seal is released.
        """
        ...

    def try_unseal(self) -> Optional[List[Tuple[VideoFrame, SharedBuffer]]]:
        """Non-blocking attempt to unseal.

        Returns ``list[tuple[VideoFrame, SharedBuffer]]`` if the seal
        has been released, or ``None`` if still sealed.

        Raises:
            RuntimeError: If already consumed by a previous call.
        """
        ...

    def __repr__(self) -> str: ...

@final
class OperatorOutput:
    """Discriminated callback payload from :class:`NvInferBatchingOperator`.

    Use ``is_inference`` / ``is_eos`` / ``is_error`` and
    ``as_operator_inference_output()``, ``eos_source_id``, ``error_message``.
    """

    @property
    def is_inference(self) -> bool: ...
    @property
    def is_eos(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
    def as_operator_inference_output(self) -> Optional[OperatorInferenceOutput]: ...
    @property
    def eos_source_id(self) -> Optional[str]: ...
    @property
    def error_message(self) -> Optional[str]: ...
    def __repr__(self) -> str: ...

@final
class OperatorInferenceOutput:
    """Full batch inference result from NvInferBatchingOperator.

    Takes over lifetime management of tensor data from the underlying
    NvInfer output. Tensor pointers in ``OperatorFrameOutput.elements``
    remain valid as long as this object is alive.

    Call :meth:`take_deliveries` to extract a :class:`SealedDeliveries`
    containing the ``(VideoFrame, SharedBuffer)`` pairs for downstream.
    """

    @property
    def frames(self) -> List[OperatorFrameOutput]:
        """Per-frame outputs (inference results only — no buffer access)."""
        ...

    @property
    def host_copy_enabled(self) -> bool:
        """Whether host (CPU) tensor data is valid."""
        ...

    @property
    def num_frames(self) -> int:
        """Number of frames in the batch."""
        ...

    def take_deliveries(self) -> Optional[SealedDeliveries]:
        """Extract sealed deliveries while keeping tensor data alive.

        Returns a :class:`SealedDeliveries` on the first call.
        Subsequent calls return ``None``.
        """
        ...

    def __repr__(self) -> str: ...

@final
class NvInferBatchingOperator:
    """Higher-level batching layer over NvInfer.

    Accepts individual ``(VideoFrame, SharedBuffer)`` pairs, accumulates them
    into batches according to configurable policies, and delegates inference
    to the underlying NvInfer pipeline. Results are delivered via a callback
    with per-frame outputs mapped back to the original frame/buffer pairs.

    Args:
        config: Batching policy and NvInfer engine configuration (the
            ``NvInferConfig`` is embedded in ``NvInferBatchingOperatorConfig``).
        batch_formation_callback: Called when a batch is ready. Receives a
            list of VideoFrames and must return a ``BatchFormationResult``
            with per-frame ROIs and Savant IDs.
        result_callback: Called for each operator result: ``OperatorOutput``
            (inference batch, per-source EOS, or pipeline error).
    """

    def __init__(
        self,
        config: NvInferBatchingOperatorConfig,
        batch_formation_callback: Callable[
            [List[VideoFrame]], BatchFormationResult
        ],
        result_callback: Callable[[OperatorOutput], None],
    ) -> None: ...
    def add_frame(
        self, frame: VideoFrame, buffer: Union[SharedBuffer, int]
    ) -> None:
        """Add a single frame for batched inference.

        If adding this frame fills the batch to ``max_batch_size``, the
        batch is submitted immediately.

        Args:
            frame: Video frame metadata.
            buffer: Individual frame buffer (NvBufSurface).
        """
        ...

    def flush(self) -> None:
        """Submit the current partial batch immediately (if non-empty)."""
        ...

    def send_eos(self, source_id: str) -> None:
        """Send logical per-source EOS to the inner NvInfer pipeline."""
        ...

    def shutdown(self) -> None:
        """Flush pending frames, stop the timer, and shut down NvInfer."""
        ...

    def __repr__(self) -> str: ...
