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
    DsNvBufSurfaceGstBuffer,           # batched GPU buffer guard
    DsNvSurfaceBufferGenerator,        # single-surface generator
    DsNvUniformSurfaceBufferGenerator, # multi-surface batch generator
    init_cuda,                         # call once before any GPU work
)
```

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
gen = DsNvSurfaceBufferGenerator(gpu_id=0)
surface = gen.create(width=1920, height=1080, format="RGBA", batch_size=1)
ptr = surface.acquire_surface_with_ptr(0)
# ptr is a raw GPU pointer (int) — write pixels via ctypes CUDA memcpy
```

### Building a batched buffer

```python
batch_gen = DsNvUniformSurfaceBufferGenerator(
    gpu_id=0, width=1920, height=1080, format="RGBA", batch_size=N
)
batch_gen.fill_slot(0, surface)
buf = batch_gen.finalize()
gst_buffer = buf.as_gst_buffer()
```

The `gst_buffer` (a `DsNvBufSurfaceGstBuffer`) is then passed to
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
