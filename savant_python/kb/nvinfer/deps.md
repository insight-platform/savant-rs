# NvInfer KB — Dependencies & Import Map

## savant_rs.nvinfer (main module)

```python
from savant_rs.nvinfer import (
    MetaClearPolicy,
    DataType,
    Roi,
    NvInferConfig,
    InferDims,
    TensorView,
    ElementOutput,
    BatchInferenceOutput,
    NvInfer,
)
```

## savant_rs.primitives.geometry (Roi bounding box)

```python
from savant_rs.primitives.geometry import (
    RBBox,                             # Roi bounding box (center-based, optionally rotated)
)
```

## savant_rs.deepstream (buffer types)

```python
from savant_rs.deepstream import (
    SharedBuffer,                      # batched GPU buffer (shared via Arc)
    SurfaceView,                       # unified GPU surface descriptor (resolves CUDA ptr on Jetson)
    BufferGenerator,        # single-surface generator
    UniformBatchGenerator, # multi-surface batch generator
    init_cuda,                         # call once before any GPU work
)
```

## Rust crate dependencies (savant_core_py)

When built with the `deepstream` feature, `savant_core_py` adds `numpy = "0.28"` for
GPU surface memory operations exposed to Python (`memset`, `upload`, `memset_slot`,
`upload_slot`).

## Third-party

| Package | Usage |
|---|---|
| `numpy` | `tensor.as_numpy()` returns ndarray; build canvases; decode outputs |
| `PIL` (Pillow) | Load JPEG/PNG test images, convert to RGBA numpy |
| `ctypes` | CUDA runtime calls: `cuMemcpyHtoD_v2`, `cuMemsetD8_v2` |
| `json` | Load ground-truth data |

## Buffer Interop

### Creating a single NvBufSurface

```python
gen = BufferGenerator(format="RGBA", width=1920, height=1080, gpu_id=0, pool_size=1)
buf = gen.acquire(id=0)
view = SurfaceView.from_buffer(buf, cuda_stream=0)
# view.data_ptr is a CUDA device pointer — use nvbuf_as_gpu_mat for OpenCV upload
```

### Building a batched buffer

```python
batched_gen = UniformBatchGenerator(
    format="RGBA", width=1920, height=1080,
    max_batch_size=32, pool_size=2, gpu_id=0,
)
config = TransformConfig()
batch = batched_gen.acquire_batch(config, ids=[(SavantIdMetaKind.FRAME, 0)])
batch.transform_slot(0, src_buf)
batch.finalize()
gst_buffer = batch.shared_buffer()
del batch, view, src_buf  # release Arc refs before consumption
```

The `gst_buffer` (a `SharedBuffer`) is then passed to
`engine.submit()` or `engine.infer_sync()`.

### CUDA memory operations (ctypes)

```python
import ctypes

libcuda = ctypes.cdll.LoadLibrary("libcuda.so")
cuMemcpyHtoD_v2 = libcuda.cuMemcpyHtoD_v2
cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
cuMemcpyHtoD_v2.restype = ctypes.c_int

data = numpy_array.ctypes.data_as(ctypes.c_void_p)
result = cuMemcpyHtoD_v2(gpu_ptr, data, numpy_array.nbytes)
assert result == 0, f"cuMemcpyHtoD_v2 failed: {result}"
```
