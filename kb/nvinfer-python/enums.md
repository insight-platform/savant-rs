# NvInfer KB — Enums

## MetaClearPolicy

| Variant | int | Meaning |
|---|---|---|
| `NONE` | 0 | Never clear object metadata automatically |
| `BEFORE` | 1 | Clear stale objects before attaching ROI objects (default) |
| `AFTER` | 2 | Clear all objects when `BatchInferenceOutput` is dropped |
| `BOTH` | 3 | Clear before **and** after |

Default for `NvInferConfig` is `BEFORE`.

When `BEFORE` is active, `attach_batch_meta_with_rois` removes existing
`NvDsObjectMeta` from each frame before writing the new ROI objects.

When `AFTER` is active, the `Drop` impl of `BatchInferenceOutput` calls
`clear_all_frame_objects` on the underlying GStreamer sample.

### Usage

```python
from savant_rs.nvinfer import MetaClearPolicy

policy = MetaClearPolicy.BOTH
assert int(policy) == 3
assert policy == MetaClearPolicy.BOTH
```

---

## DataType

| Variant | int | element_size() | numpy dtype |
|---|---|---|---|
| `FLOAT` | 0 | 4 | `float32` |
| `HALF` | 1 | 2 | `float16` |
| `INT8` | 2 | 1 | `int8` |
| `INT32` | 3 | 4 | `int32` |

### Usage

```python
from savant_rs.nvinfer import DataType

dt = DataType.HALF
assert dt.element_size() == 2
assert repr(dt) == "DataType.HALF"
```

### Mapping to numpy dtypes

```python
DTYPE_MAP = {
    DataType.FLOAT: np.float32,
    DataType.HALF:  np.float16,
    DataType.INT8:  np.int8,
    DataType.INT32: np.int32,
}
```

`TensorView.numpy_dtype` returns the corresponding dtype string
(`"float32"`, `"float16"`, `"int8"`, `"int32"`).
