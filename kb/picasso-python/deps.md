# External Dependencies for Picasso Tests

## Import Map

### savant_rs.picasso (all picasso types)
```python
from savant_rs.picasso import (
    # Encoder enums
    Platform, RateControl, H264Profile, HevcProfile,
    DgpuPreset, TuningPreset, JetsonPresetLevel,
    # Encoder property structs
    H264DgpuProps, HevcDgpuProps, H264JetsonProps, HevcJetsonProps,
    JpegProps, PngProps, Av1DgpuProps, EncoderProperties, EncoderConfig,
    # Spec types
    GeneralSpec, EvictionDecision, ConditionalSpec, ObjectDrawSpec,
    CodecSpec, SourceSpec,
    # PTS reset & callback ordering
    PtsResetPolicy, StreamResetReason, CallbackInvocationOrder,
    # Output messages
    OutputMessage,
    # Callbacks & engine
    Callbacks, PicassoEngine,
)
```

### savant_rs.deepstream (GPU/transform, required for encode/bypass)
```python
from savant_rs.deepstream import (
    SharedBuffer,  # Python wrapper for Arc-backed NvBufSurface GStreamer buffer (Option<SharedBuffer> for move semantics)
    SurfaceView,          # unified GPU surface descriptor (preferred buf for send_frame)
    MemType,              # memory type enum
    BufferGenerator,# GPU buffer pool
    TransformConfig,      # transform config for CodecSpec.encode()
    VideoFormat,          # pixel format enum
    init_cuda,            # CUDA context init
    Padding,              # NONE, RIGHT_BOTTOM, SYMMETRIC
    Interpolation,        # NEAREST, BILINEAR, ALGO1-4, DEFAULT
    ComputeMode,          # DEFAULT, GPU, VIC
    # Pure-Python helpers (injected at import time, see note below)
    GpuMatCudaArray,      # __cuda_array_interface__ wrapper for cv2.cuda.GpuMat
    make_gpu_mat,         # allocate a zero-initialised GpuMat
    nvgstbuf_as_gpu_mat,  # context manager: SharedBuffer (or int) → (GpuMat, Stream)
    nvbuf_as_gpu_mat,     # context manager: raw CUDA params → (GpuMat, Stream)
    from_gpumat,          # GpuMat → SharedBuffer via generator pool
    SkiaCanvas,           # convenience Skia wrapper for SkiaContext FBO
)
```

### savant_rs.draw_spec (for ObjectDrawSpec entries)
```python
from savant_rs.draw_spec import (
    BoundingBoxDraw,  # (border_color, background_color, thickness, padding)
    ColorDraw,        # (red, green, blue, alpha) all int 0-255
    DotDraw,          # (color: ColorDraw, radius: int)
    LabelDraw,        # (font_color, background_color, border_color, font_scale, thickness, position, padding, format)
    LabelPosition,    # (position: LabelPositionKind, margin_x, margin_y)
    ObjectDraw,       # (bounding_box?, central_dot?, label?, blur=False, bbox_source=DetectionBox)
    PaddingDraw,      # (left, top, right, bottom) all int
)
```

### savant_rs.gstreamer
```python
from savant_rs.gstreamer import Codec  # H264, HEVC, JPEG, AV1, PNG, RAW_RGBA, RAW_RGB
```

### savant_rs.primitives (frames, objects, geometry)
```python
from savant_rs.primitives import (
    VideoFrame,        # frame container
    VideoFrameContent, # .none(), .external(url, method?)
    VideoObject,       # detection object
    IdCollisionResolutionPolicy,  # .GenerateNewId
    EndOfStream,       # EOS signal
)
from savant_rs.primitives.geometry import RBBox  # (xc, yc, width, height, [angle])
from savant_rs.match_query import MatchQuery     # .idle() for querying objects
```

---

## Key Type Constructors (quick ref)

### VideoFrame
```python
VideoFrame(
    source_id="src-0",
    framerate="30/1",
    width=1280, height=720,
    content=VideoFrameContent.none(),
    time_base=(1, 1_000_000_000),
    pts=0,
)
```

### VideoObject
```python
VideoObject(
    id=0, namespace="detector", label="person",
    detection_box=RBBox(cx, cy, w, h),
    attributes=[], confidence=None,
    track_id=None, track_box=None,
)
# Add to frame:
frame.add_object(obj, IdCollisionResolutionPolicy.GenerateNewId)
```

### BufferGenerator
```python
gen = BufferGenerator(VideoFormat.RGBA, width, height, fps_num, fps_den, gpu_id)
buf = gen.acquire(id=frame_idx)  # returns SharedBuffer
# pts/duration are taken from the VideoFrame; set frame.pts and frame.duration before send_frame.
# For raw buffer info (data_ptr, pitch, width, height) use get_nvbufsurface_info(buf).
```

### Rect
```python
Rect(top, left, width, height)  # optional per-call crop for transform/send_frame
```

### TransformConfig
```python
TransformConfig()  # all defaults: Padding.SYMMETRIC, Interpolation.BILINEAR, ComputeMode.DEFAULT
# src_rect removed — pass Rect to engine.send_frame(..., src_rect=...) or generator.transform(..., src_rect=...) per call
```

### ColorDraw
```python
ColorDraw(red=255, green=0, blue=0, alpha=255)
ColorDraw.transparent()  # (0,0,0,0)
```

### PaddingDraw
```python
PaddingDraw(left=0, top=0, right=0, bottom=0)
PaddingDraw.default_padding()  # all zeros
```

### LabelPosition
```python
LabelPosition.default_position()  # TopLeftOutside, margin_x=0, margin_y=-10
```

---

## Rust/Python Mixed-Import Architecture

`savant_rs` is a **maturin-built PyO3 extension module** (`savant_rs.cpython-*.so`).
At import time the native `.so` registers submodules directly into `sys.modules`:

```rust
// savant_python/src/lib.rs
sys_modules.set_item("savant_rs.deepstream", m.getattr("deepstream")?)?;
sys_modules.set_item("savant_rs.picasso",    m.getattr("picasso")?)?;
// ... etc.
```

This means **`savant_rs.deepstream` resolves to the native submodule, NOT the
Python `deepstream/` package directory** — even though both exist on disk.
The `deepstream/__init__.py` is never executed at runtime.

### How pure-Python helpers are exposed

Pure-Python helpers (`nvgstbuf_as_gpu_mat`, `nvbuf_as_gpu_mat`, `from_gpumat`,
`SkiaCanvas`) cannot live
inside the native submodule.  They are exposed via **attribute injection** in
`savant_rs/__init__.py`:

```python
# savant_rs/__init__.py (simplified)
from .savant_rs import *                     # loads .so, registers submodules in sys.modules

import sys as _sys
_ds = _sys.modules.get("savant_rs.deepstream")
if _ds is not None:
    try:
        from savant_rs._ds_gpumat import nvgstbuf_as_gpu_mat, nvbuf_as_gpu_mat, from_gpumat
        _ds.nvgstbuf_as_gpu_mat = nvgstbuf_as_gpu_mat
        _ds.nvbuf_as_gpu_mat = nvbuf_as_gpu_mat
        _ds.from_gpumat = from_gpumat
    except ImportError:
        pass
    try:
        from savant_rs._ds_skia_canvas import SkiaCanvas
        _ds.SkiaCanvas = SkiaCanvas
    except ImportError:
        pass
```

### Rules for adding new pure-Python helpers to a native submodule

1. **Place the file at the `savant_rs` package root** as `_<prefix>_<name>.py`
   (e.g. `_ds_gpumat.py`).  Do NOT put it inside the shadowed `deepstream/`
   directory — it will never be loaded.

2. **Use absolute imports** to reference native symbols:
   ```python
   from savant_rs.deepstream import BufferGenerator, get_buffers_info
   ```
   By the time `savant_rs/__init__.py` imports your file, the native `.so` has
   already registered `savant_rs.deepstream` in `sys.modules`, so this resolves
   correctly.

3. **Inject into the native module** from `savant_rs/__init__.py`:
   ```python
   _ds = _sys.modules.get("savant_rs.deepstream")
   if _ds is not None:
       try:
           from savant_rs._my_helpers import my_func
           _ds.my_func = my_func
       except ImportError:
           pass
   ```

4. **Wrap with `try/except ImportError`** when the helper has optional
   dependencies (`cv2`, `skia`, etc.) so that users who don't need
   those features aren't broken.

5. **Update the `.pyi` type stubs** in `deepstream/deepstream.pyi` to include
   the injected symbols — type checkers read the `.pyi`, not the runtime module.

### File layout

```
savant_python/python/savant_rs/
├── __init__.py              ← imports .so + injects pure-Python helpers
├── _ds_gpumat.py            ← nvgstbuf_as_gpu_mat, nvbuf_as_gpu_mat, from_gpumat (requires cv2)
├── _ds_skia_canvas.py       ← SkiaCanvas (requires skia-python)
├── deepstream/
│   ├── __init__.py          ← dead at runtime (shadowed by native module)
│   └── deepstream.pyi       ← type stubs used by IDEs/mypy
├── picasso/
│   └── picasso.pyi          ← type stubs for savant_rs.picasso
└── ...
```

⚠ After adding/modifying helpers, the wheel must be rebuilt and reinstalled
(`SAVANT_FEATURES=deepstream make build_savant install`) for changes to take
effect.  For quick iteration, copy changed files directly into
`venv/lib/python3.12/site-packages/savant_rs/` and clear `__pycache__`.

---

## Third-Party GPU Drawing (for on_gpumat / on_render callbacks)

### OpenCV CUDA — GpuMat helpers

Two context managers for different call sites:

```python
from savant_rs.deepstream import nvgstbuf_as_gpu_mat, nvbuf_as_gpu_mat, from_gpumat

# nvgstbuf_as_gpu_mat: takes a SharedBuffer (or raw int ptr), extracts NvBufSurface info.
# Use outside callbacks (e.g. pre-filling backgrounds before send_frame).
buf = gen.acquire(id=i)  # SharedBuffer
with nvgstbuf_as_gpu_mat(buf) as (mat, stream):
    mat.setTo((20, 20, 28, 255), stream=stream)
# stream is synchronised on exit; buf safe to push downstream

# nvbuf_as_gpu_mat: takes raw CUDA params directly.
# Use inside the on_gpumat callback which already provides these values.
def on_gpumat(source_id, frame, data_ptr, pitch, width, height, cuda_stream):
    with nvbuf_as_gpu_mat(data_ptr, pitch, width, height) as (mat, stream):
        mat.setTo((20, 20, 28, 255), stream=stream)

# from_gpumat: copy a GpuMat into a new buffer (with optional scaling)
new_buf = from_gpumat(gen, some_gpumat, interpolation=cv2.INTER_LINEAR)  # returns SharedBuffer
```

### skia-python — SkiaCanvas helper (on_render)
```python
from savant_rs.deepstream import SkiaCanvas

# SkiaCanvas wraps SkiaContext + skia GrDirectContext + Surface in one object.
# from_fbo() is designed for Picasso's on_render callback:
class MyRenderer:
    def __init__(self):
        self._canvas = None

    def __call__(self, source_id, fbo_id, width, height, frame):
        if self._canvas is None:
            self._canvas = SkiaCanvas.from_fbo(fbo_id, width, height)
        self._canvas.gr_context.resetContext()  # ← CRITICAL with draw spec
        c = self._canvas.canvas()
        # ... draw on c (skia.Canvas) ...
        self._canvas.gr_context.flushAndSubmit()
```

### skia-python — raw API (on_render)
```python
import skia

GL_RGBA8 = 0x8058

# ⚠ On headless EGL (DeepStream), plain MakeGL() returns None.
# Always use MakeEGL() for the GL interface:
interface = skia.GrGLInterface.MakeEGL()
gr_context = skia.GrDirectContext.MakeGL(interface)

# Wrap Picasso's FBO as a Skia surface:
fb_info = skia.GrGLFramebufferInfo(fbo_id, GL_RGBA8)
target = skia.GrBackendRenderTarget(width, height, 0, 8, fb_info)
surface = skia.Surface.MakeFromBackendRenderTarget(
    gr_context, target,
    skia.kTopLeft_GrSurfaceOrigin,
    skia.kRGBA_8888_ColorType,
    None,
)
canvas = surface.getCanvas()
# ... draw ...
surface.flushAndSubmit()
```

### savant_gstreamer (MP4 muxing)
```python
from savant_gstreamer import Mp4Muxer, Codec
# Mp4Muxer is Send — safe to use directly from callbacks.
muxer = Mp4Muxer(Codec.H264, "/tmp/out.mp4", fps_num=30)
muxer.push(data_bytes, pts_ns, dts_ns, duration_ns)
muxer.finish()
```
