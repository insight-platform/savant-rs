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

## NvInferConfig

```python
class NvInferConfig:
    def __init__(
        self,
        nvinfer_properties: Dict[str, str],   # REQ
        input_format: str,                     # REQ (e.g. "RGBA")
        input_width: int,                      # REQ
        input_height: int,                     # REQ
        *,
        name: str = "",                                           # OPT
        element_properties: Optional[Dict[str, str]] = None,     # OPT
        gpu_id: int = 0,                                         # OPT
        queue_depth: int = 0,                                    # OPT
        meta_clear_policy: MetaClearPolicy = MetaClearPolicy.BEFORE, # OPT
        disable_output_host_copy: bool = False,                  # OPT
        scaling: ModelInputScaling = ModelInputScaling.FILL,      # OPT
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
        disable_output_host_copy: bool = False,
        scaling: ModelInputScaling = ModelInputScaling.FILL,
    ) -> NvInferConfig: ...

    @property
    def name(self) -> str: ...
    @property
    def gpu_id(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def input_format(self) -> str: ...
    @property
    def input_width(self) -> Optional[int]: ...    # None for flexible configs
    @property
    def input_height(self) -> Optional[int]: ...   # None for flexible configs
    @property
    def meta_clear_policy(self) -> MetaClearPolicy: ...
    @property
    def disable_output_host_copy(self) -> bool: ... # True = skip D2H copy
    @property
    def scaling(self) -> ModelInputScaling: ...
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

## NvInfer

```python
class NvInfer:
    def __init__(
        self,
        config: NvInferConfig,
        callback: Callable[[BatchInferenceOutput], None],
    ) -> None: ...

    def submit(
        self,
        batch: SharedBuffer,
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> None: ...

    def infer_sync(
        self,
        batch: SharedBuffer,
        rois: Optional[Dict[int, List[Roi]]] = None,
        timeout_ms: int = 30000,
    ) -> BatchInferenceOutput: ...

    def shutdown(self) -> None: ...
```

- `submit()` — async, results arrive via callback. The buffer is consumed
  (internally deconstructed from `SharedBuffer` to `gst::Buffer`).
- `infer_sync()` — blocks up to `timeout_ms` milliseconds (default 30 000),
  returns result directly. Same buffer consumption semantics.
- `shutdown()` — sends EOS, drains, stops pipeline. Raises if already shut down.

### Rust API

In the Rust API, `submit()` and `infer_sync()` accept `SharedBuffer` directly
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

Pipeline correlation (PTS) is auto-generated internally — callers do not
provide a `batch_id`.

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
    ) -> None: ...

    @property
    def max_batch_size(self) -> int: ...
    @property
    def max_batch_wait_ms(self) -> int: ...
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
        result_callback: Callable[[OperatorInferenceOutput], None],
    ) -> None: ...

    def add_frame(self, frame: VideoFrame, buffer: SharedBuffer) -> None: ...
    def flush(self) -> None: ...
    def shutdown(self) -> None: ...
```

Higher-level batching layer over `NvInfer`.  Accepts individual
`(VideoFrame, SharedBuffer)` pairs, accumulates them into batches,
and delivers per-frame results via `result_callback`.

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
