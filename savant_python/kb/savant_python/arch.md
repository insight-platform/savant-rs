# savant_python Architecture

## Overview

`savant_python` is a **thin maturin wrapper** that produces the `savant_rs`
Python wheel. It contains no business logic — all class and function
implementations live in `savant_core_py`.

```
savant_python/
├── src/
│   ├── lib.rs       # PyO3 module definition: submodule registration + sys.modules injection
│   └── build.rs     # pyo3-build-config link args
├── python/
│   └── savant_rs/   # pure-Python source root (merged into wheel)
│       ├── __init__.py          # re-exports native .so + injects pure-Python helpers
│       ├── savant_rs.pyi        # top-level type stubs
│       ├── _ds_gpumat.py        # GpuMat helpers (injected into savant_rs.deepstream)
│       ├── _ds_skia_canvas.py   # SkiaCanvas helper (injected into savant_rs.deepstream)
│       ├── primitives/          # .pyi stubs for geometry, frame, object, etc.
│       ├── picasso/             # .pyi stubs for picasso
│       ├── nvinfer/             # .pyi stubs for nvinfer
│       ├── deepstream/          # .pyi stubs for deepstream
│       ├── draw_spec/           # .pyi stubs
│       ├── utils/               # .pyi stubs + serialization/symbol_mapper
│       ├── zmq/                 # .pyi stubs
│       ├── pipeline/            # .pyi stubs
│       ├── match_query/         # .pyi stubs
│       ├── logging/             # .pyi stubs
│       ├── telemetry/           # .pyi stubs
│       ├── webserver/           # .pyi stubs + kvs/
│       ├── metrics/             # .pyi stubs
│       ├── gstreamer/           # .pyi stubs
│       ├── test/                # .pyi stubs
│       ├── atomic_counter/      # .pyi stubs
│       └── py/                  # pure-Python packages (api, client, log, utils)
├── pytests/                     # Python tests (35 test files)
├── Cargo.toml                   # maturin metadata, features, deps
└── pyproject.toml               # Python build config
```

## Dependency Chain

```
savant_python (savant_rs wheel)
 └── savant_core_py (PyO3 bindings)
      ├── savant_core (Rust library)
      ├── pyo3
      └── [feature=deepstream]
           ├── nvinfer (DeepStream nvinfer plugin)
           ├── picasso (frame processing engine)
           ├── deepstream_buffers (NvBufSurface wrappers)
           ├── deepstream_encoders (H264/H265/JPEG/PNG encoders)
           ├── nvidia_gpu_utils (GPU memory queries, Jetson model detection, NVENC capability)
           └── skia-safe (Skia rendering)
```

## Feature Flags

| Feature | Cargo.toml | What it enables |
|---------|-----------|-----------------|
| `default` | `[]` | Core modules only (no GPU) |
| `gst` | `savant_core_py/gst` | GStreamer types (`Codec`, `Mp4Muxer`, `FlowResult`) |
| `deepstream` | `gst` + `savant_core_py/deepstream` | All of `gst` + DeepStream, nvinfer, picasso |

## Module Registration (`lib.rs`)

`savant_python/src/lib.rs` defines PyO3 submodules and delegates to
`savant_core_py` for class registration:

```
savant_rs (root module)
 ├── primitives        → savant_core_py::primitives (Attribute, VideoFrame, etc.)
 │   └── geometry      → savant_core_py::primitives::bbox + point/segment/area (RBBox, BBox, Point, Segment, etc.)
 ├── draw_spec         → savant_core_py::draw_spec
 ├── utils             → savant_core_py::utils
 │   ├── symbol_mapper → savant_core_py::utils::symbol_mapper
 │   └── serialization → savant_core_py::primitives::message
 ├── pipeline          → savant_core_py::pipeline (Python names: VideoPipeline, VideoPipelineConfiguration)
 ├── pipeline2         → alias for pipeline (sys.modules["savant_rs.pipeline2"] = pipeline)
 ├── match_query       → savant_core_py::match_query
 ├── logging           → savant_core_py::logging
 ├── zmq               → savant_core_py::zmq
 ├── telemetry         → savant_core_py::telemetry
 ├── webserver         → savant_core_py::webserver
 │   └── kvs           → savant_core_py::webserver::kvs
 ├── metrics           → savant_core_py::metrics
 ├── gstreamer         → savant_core_py::gstreamer [feature=gst]
 ├── deepstream        → savant_core_py::deepstream [feature=deepstream]
 ├── nvinfer           → savant_core_py::nvinfer [feature=deepstream]
 └── picasso           → savant_core_py::picasso [feature=deepstream]
```

Each submodule is also injected into `sys.modules` so that
`from savant_rs.<submod> import <Class>` works.

## sys.modules Injection

After all submodules are created, `init_all()` injects them into
`sys.modules`:

```python
sys.modules["savant_rs.primitives"] = primitives
sys.modules["savant_rs.primitives.geometry"] = geometry
sys.modules["savant_rs.deepstream"] = deepstream
sys.modules["savant_rs.nvinfer"] = nvinfer
sys.modules["savant_rs.picasso"] = picasso
# ... etc.
```

This is necessary because PyO3 submodules are not automatically importable
as Python packages.

## Pure-Python Injection (`__init__.py`)

After the native module loads, `__init__.py` monkey-patches pure-Python
helpers into native submodules:

- `GpuMatCudaArray`, `make_gpu_mat`, `nvgstbuf_as_gpu_mat`, `nvbuf_as_gpu_mat`,
  `from_gpumat` → injected into `savant_rs.deepstream`
- `SkiaCanvas` → injected into `savant_rs.deepstream`

These are imported from `_ds_gpumat.py` and `_ds_skia_canvas.py` respectively.
Import errors are silently caught (expected when DeepStream is not available).

## Type Stubs (.pyi)

Every Python-visible class and function has a corresponding `.pyi` stub:

| Module | Stub file |
|--------|-----------|
| `savant_rs` | `savant_rs.pyi` |
| `savant_rs.primitives.geometry` | `primitives/geometry/geometry.pyi` |
| `savant_rs.primitives` | `primitives/video_frame.pyi`, `primitives/video_object.pyi`, etc. |
| `savant_rs.nvinfer` | `nvinfer/nvinfer.pyi` |
| `savant_rs.picasso` | `picasso/picasso.pyi` |
| `savant_rs.deepstream` | `deepstream/deepstream.pyi` |
| ... | ... |

**Rule**: When a class/function signature changes in `savant_core_py`, the
corresponding `.pyi` file in `savant_python/python/savant_rs/` MUST be updated.
