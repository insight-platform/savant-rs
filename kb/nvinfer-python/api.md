# NvInfer KB — Python API Reference

## Module: `savant_rs.nvinfer`

All types live in `savant_rs.nvinfer`. External dependencies are in `savant_rs.deepstream`
and `savant_rs.primitives.geometry` (see `deps.md`).

---

## MetaClearPolicy

Enum controlling when object metadata is erased from the batch buffer.

```python
class MetaClearPolicy:
    NONE: MetaClearPolicy      # never clear
    BEFORE: MetaClearPolicy    # clear stale objects before attaching ROI objects (DEFAULT)
    AFTER: MetaClearPolicy     # clear all objects when the output is dropped
    BOTH: MetaClearPolicy      # clear before AND after
```

Supports `==`, `!=`, `int()`, `hash()`.

---

## ModelInputScaling

How input frames are scaled to the model's fixed input dimensions. Backed by
nvinfer `maintain-aspect-ratio` / `symmetric-padding` (injected by the Rust
config builder). Do **not** set those keys in `nvinfer_properties`.

```python
class ModelInputScaling:
    FILL: ModelInputScaling                        # stretch to model input (DEFAULT)
    KEEP_ASPECT_RATIO: ModelInputScaling         # preserve AR, padding right/bottom
    KEEP_ASPECT_RATIO_SYMMETRIC: ModelInputScaling  # preserve AR, symmetric padding
```

Supports `==`, `!=`, `int()`, `hash()`, `repr()`.

---

## ModelColorFormat

Color format the model expects for its input tensor.

```python
class ModelColorFormat:
    RGB: ModelColorFormat
    BGR: ModelColorFormat
    GRAY: ModelColorFormat
```

Supports `==`, `!=`, `int()`, `hash()`, `repr()`.

---

## DataType

Enum for tensor element types.

```python
class DataType:
    FLOAT: DataType   # 32-bit float  (4 bytes)
    HALF: DataType    # 16-bit float  (2 bytes)
    INT8: DataType    # 8-bit int     (1 byte)
    INT32: DataType   # 32-bit int    (4 bytes)

    def element_size(self) -> int: ...
```

---

## Roi

Region of interest: identifier + bounding box (center-based, optionally rotated).

```python
class Roi:
    def __init__(self, id: int, bbox: RBBox) -> None: ...

    @property
    def id(self) -> int: ...        # read-only, int

    @property
    def bbox(self) -> RBBox: ...    # read-only, savant_rs.primitives.geometry.RBBox
```

`RBBox` is `savant_rs.primitives.geometry.RBBox(xc, yc, width, height, angle=None)`.
When the angle is non-zero, the batch-meta builder computes the axis-aligned wrapping
box (via `get_wrapping_bbox`) and clamps it to `[0, max_w] × [0, max_h]` before
passing to DeepStream. Both rotated and axis-aligned paths use the same clamping
logic.

---

## RoiKind

Specifies how ROIs are provided for a frame: either the entire frame or an
explicit list of `Roi` objects.

```python
class RoiKind:
    @staticmethod
    def full_frame() -> RoiKind: ...

    @staticmethod
    def rois(rois: List[Roi]) -> RoiKind: ...

    @property
    def is_full_frame(self) -> bool: ...

    def get_rois(self) -> List[Roi]: ...
```

- `full_frame()` — the entire frame is the ROI (no crop).
- `rois(...)` — explicit list of per-frame ROIs.
- `is_full_frame` — `True` when constructed via `full_frame()`.
- `get_rois()` — returns the ROI list; empty for `full_frame()`.

---

## NvInferConfig

```python
class NvInferConfig:
    def __init__(
        self,
        nvinfer_properties: Dict[str, str],   # REQ
        input_format: VideoFormat,             # REQ (e.g. VideoFormat.RGBA)
        model_width: int,                      # REQ
        model_height: int,                     # REQ
        *,
        name: str = "",                                           # OPT
        element_properties: Optional[Dict[str, str]] = None,     # OPT
        gpu_id: int = 0,                                         # OPT
        queue_depth: int = 0,                                    # OPT — GStreamer queue max-size-buffers
        input_channel_capacity: int = 16,                       # OPT — framework input channel (backpressure)
        output_channel_capacity: int = 16,                        # OPT — framework output channel
        drain_poll_interval_ms: int = 100,                        # OPT — appsink poll interval in framework drain
        meta_clear_policy: MetaClearPolicy = MetaClearPolicy.BEFORE, # OPT
        disable_output_host_copy: bool = False,                  # OPT
        scaling: ModelInputScaling = ModelInputScaling.FILL,      # OPT
        model_color_format: ModelColorFormat = ModelColorFormat.RGB, # OPT
        operation_timeout_ms: int = 30000,                         # OPT — framework watchdog / in-flight deadline
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
    def model_input_dimensions(self) -> tuple[int, int]: ...
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
    def disable_output_host_copy(self) -> bool: ... # True = skip D2H copy
    @property
    def scaling(self) -> ModelInputScaling: ...
    @property
    def operation_timeout_ms(self) -> int: ...  # timeout in ms (Rust pipeline watchdog)
```

**Forbidden in `nvinfer_properties`:** `maintain-aspect-ratio`, `symmetric-padding`
(use `scaling` on `NvInferConfig` instead).

### nvinfer_properties dict

Use dotted keys for per-class sections:
```python
{
    "model-engine-file": "/models/foo.engine",
    "config-file-path": "/configs/foo.txt",
    "class-attrs-all.pre-cluster-threshold": "0.2",
}
```

Bare keys are written to `[property]`. `process-mode`,
`output-tensor-meta`, `maintain-aspect-ratio`, and (when applicable)
`symmetric-padding` are set by the Rust config builder from `NvInferConfig`
fields — do not duplicate them in the dict.

---

## InferDims

```python
class InferDims:
    @property
    def dimensions(self) -> List[int]: ...     # shape per axis
    @property
    def num_elements(self) -> int: ...         # product of dimensions
```

---

## TensorView

Zero-copy view into a single output tensor. Valid while parent
`BatchInferenceOutput` is alive.

```python
class TensorView:
    @property
    def name(self) -> str: ...
    @property
    def dims(self) -> InferDims: ...
    @property
    def data_type(self) -> DataType: ...
    @property
    def byte_length(self) -> int: ...
    @property
    def host_ptr(self) -> int: ...       # CPU address, 0 if unavailable
    @property
    def device_ptr(self) -> int: ...     # GPU address, 0 if unavailable
    @property
    def has_host_data(self) -> bool: ... # False when disable_output_host_copy is set
    @property
    def numpy_dtype(self) -> str: ...    # "float32", "float16", "int8", "int32"
```

`host_ptr` / `device_ptr` are plain integer addresses. Build framework-native
tensors (NumPy via `ctypes`, CuPy, PyTorch) without any data copy on the Rust
side — see `patterns.md` for `tensor_to_numpy` / `tensor_to_cupy` helpers.

`numpy_dtype` maps `data_type` to a NumPy-compatible string:
FLOAT → `"float32"`, HALF → `"float16"`, INT8 → `"int8"`, INT32 → `"int32"`.

### `as_numpy()`

```python
def as_numpy(self) -> numpy.ndarray:
    """Return tensor data as a NumPy array (zero-copy view).

    Automatically selects dtype from ``data_type``, reshapes to
    ``dims.dimensions``, and validates host data availability.

    Raises:
        RuntimeError: If host data is unavailable (host copy disabled
            or null pointer).
    """
```

Preferred over the manual `ctypes` approach for typical use — see `patterns.md`.

⚠ Raises `RuntimeError` if `BatchInferenceOutput` has been dropped.

---

## ElementOutput

```python
class ElementOutput:
    @property
    def roi_id(self) -> Optional[int]: ...
    @property
    def slot_number(self) -> int: ...  # NvDsFrameMeta.batch_id / surface index
    @property
    def tensors(self) -> List[TensorView]: ...
```

---

## BatchInferenceOutput

Holds the output `SharedBuffer`. Tensor data stays valid as long as this
object (or any child `TensorView`) is alive.

```python
class BatchInferenceOutput:
    @property
    def has_host_data(self) -> bool: ...  # False when disable_output_host_copy is set
    @property
    def num_elements(self) -> int: ...
    @property
    def elements(self) -> List[ElementOutput]: ...

    def buffer(self) -> SharedBuffer: ...
```

`buffer()` returns the output GStreamer buffer as a `SharedBuffer`.
`SavantIdMeta` attached to the input buffer is preserved through the pipeline
and readable from this output buffer (`savant_ids()`). Each `ElementOutput`
exposes the DeepStream surface slot as `slot_number`; correlate with user ids
by indexing `savant_ids()` with that slot.

---

## NvInferOutput

Discriminated item from `NvInfer.recv()` / `recv_timeout()` / `try_recv()`:

```python
class NvInferOutput:
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
```

Pipeline/framework errors surface as `is_error` with `error_message` (not as
Python exceptions from `recv`, except `RuntimeError` on channel disconnect).

---

## NvInfer

Pull-based API matching Rust `nvinfer::NvInfer` (no constructor callback).

```python
class NvInfer:
    def __init__(self, config: NvInferConfig) -> None: ...

    def submit(
        self,
        batch: Union[SharedBuffer, int],
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> None: ...

    def recv(self) -> NvInferOutput: ...
    def recv_timeout(self, timeout_ms: int) -> Optional[NvInferOutput]: ...
    def try_recv(self) -> Optional[NvInferOutput]: ...
    def send_eos(self, source_id: str) -> None: ...
    def send_custom_downstream_event(
        self,
        structure_name: str,
        string_fields: Optional[Dict[str, str]] = None,
    ) -> None: ...
    def is_failed(self) -> bool: ...
    def graceful_shutdown(self, timeout_ms: int) -> list[NvInferOutput]: ...
    def shutdown(self) -> None: ...
```

- `submit()` — pushes a batch; buffer is consumed (`SharedBuffer` → `gst::Buffer`).
- `recv()` — blocks until the next output item (inference, GStreamer event,
  logical EOS, or error payload). Raises `RuntimeError` only on channel disconnect.
- `recv_timeout(timeout_ms)` — same as `recv` but returns `None` on timeout.
- `try_recv()` — non-blocking; `None` if no item ready.
- `send_eos(source_id)` — logical per-source EOS (custom downstream event).
- `send_custom_downstream_event(structure_name, string_fields=None)` — custom
  downstream event with string fields only (subset of Rust `send_event`).
- `is_failed()` — terminal failed state from the underlying pipeline.
- `graceful_shutdown(timeout_ms)` — GIL released; returns drained `NvInferOutput` list; takes inner engine.
- `shutdown()` — abrupt shutdown. Raises if already shut down.

### Rust API

In the Rust API, `submit()` accepts `SharedBuffer` directly
(not `gst::Buffer`). The `into_buffer()` deconstruction happens inside these
methods. If the `SharedBuffer` has outstanding references, an `NvInferError`
is returned.

`BatchInferenceOutput` holds both a `gstreamer::Sample` and a `SharedBuffer`.
The `SharedBuffer` is a **ref-counted handle** to the same `GstBuffer` held by
the sample (obtained via `gst_mini_object_ref`, NOT `gst_mini_object_copy`).
Access it via `buffer()`, which returns a clone of the `SharedBuffer`.

**Critical**: never use `BufferRef::to_owned()` on DeepStream buffers — it
triggers `gst_mini_object_copy()` which deep-copies `NvDsBatchMeta`.
DeepStream's `batch_meta_copy` calls `nvds_acquire_meta_from_pool()` which
crashes (SIGSEGV) when the meta pool doesn't support cloning outside the DS
pipeline context. Always use `from_glib_none(buffer.as_ptr())` to get a
ref-counted owned `gst::Buffer` without copying.

Pipeline correlation (PTS) is managed via the `batch_id` parameter on
`submit()`. The Python API also accepts a raw `int` (GstBuffer pointer)
in addition to `SharedBuffer`.

---

# Batching Operator Layer

## NvInferBatchingOperatorConfig

```python
class NvInferBatchingOperatorConfig:
    def __init__(
        self,
        max_batch_size: int,
        max_batch_wait_ms: int,
        nvinfer_config: NvInferConfig,
        pending_batch_timeout_ms: int = 60000,  # OPT — timeout for pending batches
    ) -> None: ...

    @property
    def max_batch_size(self) -> int: ...
    @property
    def max_batch_wait_ms(self) -> int: ...
    @property
    def pending_batch_timeout_ms(self) -> int: ...  # timeout in ms for pending batches
    @property
    def nvinfer_config(self) -> NvInferConfig: ...
```

---

## BatchFormationResult

```python
class BatchFormationResult:
    def __init__(
        self,
        ids: list[tuple[SavantIdMetaKind, int]],
        rois: list[RoiKind],
    ) -> None: ...
```

Returned by the batch formation callback.  `ids` is per-frame Savant IDs;
`rois` is per-frame ROI specification (parallel to the frames list).

---

## NvInferBatchingOperator

```python
class NvInferBatchingOperator:
    def __init__(
        self,
        config: NvInferBatchingOperatorConfig,
        batch_formation_callback: Callable[[list[VideoFrame]], BatchFormationResult],
        result_callback: Callable[[OperatorOutput], None],
    ) -> None: ...

    def add_frame(self, frame: VideoFrame, buffer: SharedBuffer) -> None: ...
    def flush(self) -> None: ...
    def send_eos(self, source_id: str) -> None: ...
    def graceful_shutdown(self, timeout_ms: int) -> list[OperatorOutput]: ...
    def shutdown(self) -> None: ...
```

Higher-level batching layer over `NvInfer`.  Accepts individual
`(VideoFrame, SharedBuffer)` pairs, accumulates them into batches,
and delivers each result via `result_callback` as `OperatorOutput`
(inference batch, per-source EOS, or pipeline error).

---

## OperatorOutput

```python
class OperatorOutput:
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
```

---

## OperatorInferenceOutput

```python
class OperatorInferenceOutput:
    @property
    def frames(self) -> list[OperatorFrameOutput]: ...
    @property
    def host_copy_enabled(self) -> bool: ...
    @property
    def num_frames(self) -> int: ...

    def take_deliveries(self) -> Optional[SealedDeliveries]: ...
```

Full batch result.  `take_deliveries()` returns `SealedDeliveries` on
the first call, `None` on subsequent calls.  Tensor pointers in
`frames[].elements[].tensors` remain valid while this object is alive.

---

## OperatorFrameOutput

```python
class OperatorFrameOutput:
    @property
    def frame(self) -> VideoFrame: ...
    @property
    def elements(self) -> list[OperatorElementOutput]: ...
```

Per-frame inference result (callback view — **no buffer access**).
The per-frame buffer is held internally and only accessible after
`take_deliveries()` + `unseal()`.

---

## OperatorElementOutput

```python
class OperatorElementOutput:
    @property
    def roi_id(self) -> Optional[int]: ...
    @property
    def slot_number(self) -> int: ...
    @property
    def tensors(self) -> list[OperatorTensorView]: ...

    def scale_points(self, data) -> NDArray[float32]: ...
    def scale_ltwh(self, data) -> NDArray[float32]: ...
    def scale_ltrb(self, data) -> NDArray[float32]: ...
    def scale_rbboxes(self, boxes: list[RBBox]) -> list[RBBox]: ...
```

Per-element output with coordinate scaling.  `scale_*` methods transform
model-space predictions to absolute frame coordinates.

---

## OperatorTensorView

Same interface as `TensorView` — `name`, `dims`, `data_type`, `byte_length`,
`host_ptr`, `device_ptr`, `has_host_data`, `numpy_dtype`, `as_numpy()`.

---

## SealedDeliveries

```python
class SealedDeliveries:
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def is_released(self) -> bool: ...
    def unseal(self, timeout_ms: Optional[int] = None) -> list[tuple[VideoFrame, SharedBuffer]]: ...
    def try_unseal(self) -> Optional[list[tuple[VideoFrame, SharedBuffer]]]: ...
```

A batch of `(VideoFrame, SharedBuffer)` pairs sealed until the
`OperatorInferenceOutput` is dropped.

- `unseal(timeout_ms=None)` — **blocking**, releases the GIL internally
  during the wait.  When `timeout_ms` is `None` (default), blocks
  indefinitely.  When a timeout is given, raises `TimeoutError` if the seal
  is not released within the specified duration.
- `try_unseal()` — non-blocking, returns `None` if still sealed.
- Dropping without calling `unseal()` is safe.

⚠ `unseal()` releases the GIL during the blocking wait.  This is critical
because the callback thread (Rust) needs the GIL to drop
`OperatorInferenceOutput`.  If the GIL were held during `unseal()`,
deadlock would result.
