# savant_python — Dependencies

## Rust Dependencies

`savant_python` has minimal direct Rust dependencies:

| Crate | Purpose |
|-------|---------|
| `savant_core_py` | All PyO3 class/function implementations |
| `pyo3` | Python ↔ Rust bridge |
| `pretty_env_logger` | Logging initialization |
| `pyo3-build-config` | Build-time link args (build-dep) |

All other Rust dependencies are transitive through `savant_core_py`.

## Feature Chains

```
savant_python::deepstream
  → savant_python::gst
  → savant_core_py::deepstream
    → savant_core_py::gst
    → nvinfer, picasso, deepstream_buffers, deepstream_encoders,
      nvidia_gpu_utils, skia-safe
```

## Python Import Map

### Core (always available)
```python
from savant_rs import version
from savant_rs.primitives import VideoFrame, Attribute, ...
from savant_rs.primitives.geometry import RBBox, Point, ...
from savant_rs.draw_spec import ColorDraw, ObjectDraw, ...
from savant_rs.utils import eval_expr, gen_frame, ByteBuffer, ...
from savant_rs.utils.serialization import save_message, load_message
from savant_rs.utils.symbol_mapper import register_model_objects, ...
from savant_rs.pipeline import VideoPipeline, VideoPipelineConfiguration, ...
from savant_rs.match_query import MatchQuery, FloatExpression, ...
from savant_rs.logging import LogLevel, set_log_level
from savant_rs.zmq import WriterConfig, ReaderConfig, ...
from savant_rs.telemetry import init, shutdown, TracerConfiguration, ...
from savant_rs.webserver import init_webserver, stop_webserver, ...
from savant_rs.webserver.kvs import set_attributes, get_attribute, ...
from savant_rs.metrics import CounterFamily, GaugeFamily, ...
```

### GStreamer (feature=gst)
```python
from savant_rs.gstreamer import FlowResult, InvocationReason, Codec, Mp4Muxer
```

### DeepStream (feature=deepstream)
```python
from savant_rs.deepstream import (
    SharedBuffer, SurfaceView,
    BufferGenerator, UniformBatchGenerator,
    init_cuda, gpu_mem_used_mib, jetson_model, is_jetson_kernel, has_nvenc,
)
from savant_rs.nvinfer import NvInfer, NvInferConfig, Roi, ...
from savant_rs.picasso import PicassoEngine, Callbacks, OutputMessage, ...
```

### Pure-Python (injected into deepstream module)
```python
from savant_rs.deepstream import GpuMatCudaArray, make_gpu_mat, SkiaCanvas
```

## Third-Party Python Dependencies (test time)

| Package | Usage |
|---------|-------|
| `pytest` | Test framework |
| `numpy` | Tensor access, image manipulation |
| `Pillow` (PIL) | Image loading for test assets |
| `cupy` | GPU tensor access (zero-copy) |
| `ctypes` | CUDA runtime calls |
| `opencv-python` | GpuMat operations (via `_ds_gpumat.py`) |

## Core Python Dependencies

From `pyproject.toml`:

| Package | Version | Usage |
|---------|---------|-------|
| `pretty-traceback` | `2024.1021` | Enhanced Python tracebacks (runtime dep) |

### Optional: `clientsdk`

| Package | Version | Usage |
|---------|---------|-------|
| `python-magic` | `~0.4.27` | MIME type detection |
| `requests` | `~2.32.5` | HTTP client |
| `numpy` | `>=1.26` | Tensor / image manipulation |
| `opencv-python` | `~4.12.0` | GpuMat operations |

## Sub-crate KB Locations

For detailed API docs on DeepStream submodules, see:
- **nvinfer**: `kb/nvinfer-python/`
- **picasso**: `kb/picasso-python/`
