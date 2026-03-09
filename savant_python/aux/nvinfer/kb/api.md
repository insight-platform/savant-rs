# NvInfer KB — Python API Reference

## Module: `savant_rs.nvinfer`

All types live in `savant_rs.nvinfer`. External dependencies are in `savant_rs.deepstream`
(see `deps.md`).

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

Region of interest: identifier + bounding box.

```python
class Roi:
    def __init__(self, id: int, rect: Rect) -> None: ...

    @property
    def id(self) -> int: ...        # read-only, int

    @property
    def rect(self) -> Rect: ...     # read-only, savant_rs.deepstream.Rect
```

`Rect` is `savant_rs.deepstream.Rect(top, left, width, height)`.

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
```

### nvinfer_properties dict

Use dotted keys for per-class sections:
```python
{
    "model-engine-file": "/models/foo.engine",
    "config-file-path": "/configs/foo.txt",
    "class-attrs-all.pre-cluster-threshold": "0.2",
}
```

Bare keys are written to `[property]`. `process-mode` and
`output-tensor-meta` are auto-injected by the Rust config builder.

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

    def as_bytes(self) -> bytes: ...
    def as_numpy(self) -> numpy.ndarray: ...
```

`as_numpy()` returns a **1-D** ndarray with dtype matching `data_type`:
FLOAT → float32, HALF → float16, INT8 → int8, INT32 → int32.

⚠ Raises `RuntimeError` if `BatchInferenceOutput` has been dropped.

---

## ElementOutput

```python
class ElementOutput:
    @property
    def frame_id(self) -> Optional[int]: ...
    @property
    def roi_id(self) -> Optional[int]: ...
    @property
    def tensors(self) -> List[TensorView]: ...
```

---

## BatchInferenceOutput

Owns the GStreamer sample. Tensor data stays valid as long as this object
(or any child `TensorView`) is alive.

```python
class BatchInferenceOutput:
    @property
    def batch_id(self) -> int: ...
    @property
    def num_elements(self) -> int: ...
    @property
    def elements(self) -> List[ElementOutput]: ...
```

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
        batch: DsNvBufSurfaceGstBuffer,
        batch_id: int,
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> None: ...

    def infer_sync(
        self,
        batch: DsNvBufSurfaceGstBuffer,
        batch_id: int,
        rois: Optional[Dict[int, List[Roi]]] = None,
    ) -> BatchInferenceOutput: ...

    def shutdown(self) -> None: ...
```

- `submit()` — async, results arrive via callback
- `infer_sync()` — blocks up to 30 s, returns result directly
- `shutdown()` — sends EOS, drains, stops pipeline. Raises if already shut down.

⚠ `batch_id` must not equal `2**64 - 1`.
