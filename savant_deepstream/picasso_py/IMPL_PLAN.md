# Picasso Python Bindings

## Overview

Implement full PyO3 bindings in picasso_py for all types re-exported in picasso's prelude.rs, plus full EncoderConfig/EncoderProperties bindings. Callbacks are Python callables wrapped via GIL-acquiring trait impls. OnRender passes SkiaCanvas to Python.

## Todos

- [x] Create src/error.rs: PicassoError -> PyErr conversion helper
- [x] Create src/encoder.rs: full Python bindings for EncoderConfig, EncoderProperties, and all 7 enums + 6 prop structs from deepstream_encoders
- [x] Create src/spec/ (mod.rs, general.rs, source.rs, codec.rs, conditional.rs, draw.rs): all spec wrappers
- [x] Create src/message.rs: PyEncodedOutput and PyBypassOutput wrappers
- [x] Change OnRender trait to pass &mut SkiaRenderer instead of &Canvas; update encode.rs call site; add SkiaCanvas.from_fbo() to deepstream_nvbufsurface
- [x] Create src/callbacks.rs: Python callable wrappers for all 6 callback traits + PyCallbacks builder. OnRender wrapper passes (source_id, fbo_id, width, height, frame) to Python; adapter in \_\_init\_\_.py wraps to SkiaCanvas.
- [x] Create src/engine.rs: PyPicassoEngine wrapping PicassoEngine with GIL-releasing methods
- [x] Update src/lib.rs: register all types in #[pymodule]
- [x] Create python/picasso/\_\_init\_\_.py with re-exports from _native and _OnRenderAdapter
- [x] Create python/picasso/_native.pyi type stubs
- [x] Update Cargo.toml if additional deps needed (savant_gstreamer, parking_lot)
- [x] Run cargo fmt, clippy, and tests; fix any issues

## Context

`picasso_py/` already has scaffolding (Cargo.toml, pyproject.toml, build.rs, empty `lib.rs` with `#[pymodule]`). We need to implement all bindings for the types in `prelude.rs`:

- **Callbacks**: `OnEncodedFrame`, `OnBypassFrame`, `OnRender`, `OnObjectDrawSpec`, `OnGpuMat`, `OnEviction`, `Callbacks`
- **Engine**: `PicassoEngine`
- **Errors**: `PicassoError`
- **Messages**: `BypassOutput`, `EncodedOutput`
- **Specs**: `CodecSpec`, `ConditionalSpec`, `EvictionDecision`, `GeneralSpec`, `ObjectDrawSpec`, `SourceSpec`

Additionally, `CodecSpec::Encode` requires `TransformConfig` (already has Python bindings in `deepstream_nvbufsurface`) and `EncoderConfig` + `EncoderProperties` (no Python bindings yet -- must be created).

## File Structure

```
picasso_py/src/
  lib.rs             -- #[pymodule] registration
  callbacks.rs       -- Python callable -> trait wrappers
  engine.rs          -- PyPicassoEngine
  error.rs           -- PicassoError -> PyErr
  message.rs         -- PyEncodedOutput, PyBypassOutput
  encoder.rs         -- PyEncoderConfig, PyEncoderProperties + all enum/struct variants
  spec/
    mod.rs           -- re-exports
    general.rs       -- PyGeneralSpec, PyEvictionDecision
    source.rs        -- PySourceSpec
    codec.rs         -- PyCodecSpec
    conditional.rs   -- PyConditionalSpec
    draw.rs          -- PyObjectDrawSpec
picasso_py/python/picasso/
  __init__.py        -- re-exports from _native + _OnRenderAdapter
  _native.pyi        -- type stubs
```

## Key Design Decisions

### Cross-crate type reuse

Types with existing Python wrappers in other crates:

- `VideoFrame` (`savant_core_py::primitives::frame::VideoFrame`) wrapping `VideoFrameProxy`
- `EndOfStream` (`savant_core_py::primitives::eos::EndOfStream`)
- `ObjectDraw` (`savant_core_py::draw_spec::ObjectDraw`)
- `BorrowedVideoObject` (`savant_core_py::primitives::object::BorrowedVideoObject`)
- `TransformConfig` (`deepstream_nvbufsurface::python::PyTransformConfig`)
- `Codec` (`savant_gstreamer::python::PyCodec`)
- `VideoFormat`, `NvBufSurfaceMemType` (`deepstream_nvbufsurface::python::*`)

These will be accepted directly in `#[pymethods]` function signatures. The `from_py_object` attribute on their `#[pyclass]` makes cross-module extraction work.

### GStreamer Buffer

No Python wrapper for `gstreamer::Buffer`. Follow the established pattern: accept `buf_ptr: usize` (from `hash(buffer)` in PyGObject) and reconstruct via `gst::BufferRef::from_mut_ptr()`.

### Callbacks as Python callables

Each callback trait gets a struct holding `Py<PyAny>` (a GIL-independent reference to a Python callable):

```rust
struct PyOnEncodedFrame(Py<PyAny>);

impl OnEncodedFrame for PyOnEncodedFrame {
    fn call(&self, output: EncodedOutput) {
        Python::with_gil(|py| {
            let py_output = PyEncodedOutput::from(output);
            if let Err(e) = self.0.call1(py, (py_output,)) {
                log::error!("on_encoded_frame callback error: {e}");
            }
        });
    }
}
```

The `Callbacks` struct is built in Python via `PyCallbacks` with optional setters for each slot.

### CodecSpec as tagged union

Since Python doesn't have Rust-style enums, expose via factory static methods:

- `CodecSpec.drop()` -- returns `Drop` variant
- `CodecSpec.bypass()` -- returns `Bypass` variant
- `CodecSpec.encode(transform, encoder)` -- returns `Encode` variant with `PyTransformConfig` and `PyEncoderConfig`

### EncoderConfig + EncoderProperties (full bindings)

New `encoder.rs` module wrapping all types from `deepstream_encoders/src/properties.rs`:

**Enums** (~7): `Platform`, `RateControl`, `H264Profile`, `HevcProfile`, `DgpuPreset`, `JetsonPresetLevel`, `TuningPreset` -- each as `#[pyclass(eq, eq_int)]`.

**Props structs** (~6): `H264DgpuProps`, `HevcDgpuProps`, `H264JetsonProps`, `HevcJetsonProps`, `JpegProps`, `Av1DgpuProps` -- each as `#[pyclass]` with `#[pyo3(get, set)]` on all `Option<T>` fields and `#[new]` with defaults.

**EncoderProperties** enum: factory static methods `h264_dgpu(props)`, `hevc_dgpu(props)`, etc.

**EncoderConfig**: `#[pyclass]` with typed constructor and getters/setters.

### OnRender callback (SkiaCanvas integration)

**Goal**: Python users receive a `SkiaCanvas` (from `deepstream_nvbufsurface`) in their `on_render` callback, giving them the full `skia-python` drawing API on the internal GPU canvas.

**Step 1 -- Rust trait change** in `picasso/src/callbacks.rs`:

Change `OnRender::call` to receive `&SkiaRenderer` instead of `&skia_safe::Canvas`. Rust users call `renderer.canvas()` to get the canvas. The call site in `picasso/src/pipeline/encode.rs:122-126` already has `skia: &mut SkiaRenderer`:

```rust
// callbacks.rs
pub trait OnRender: Send + Sync + 'static {
    fn call(&self, source_id: &str, renderer: &SkiaRenderer, frame: &VideoFrameProxy);
}

// encode.rs -- trivial call-site change:
// Before:  cb.call(source_id, skia.canvas(), &input.frame);
// After:   cb.call(source_id, skia, &input.frame);
```

**Step 2 -- Add `SkiaCanvas.from_fbo()`** in `deepstream_nvbufsurface/python/deepstream_nvbufsurface/skia_canvas.py`:

New classmethod that creates a `SkiaCanvas` from an existing FBO (no `SkiaContext` needed):

```python
@classmethod
def from_fbo(cls, fbo_id: int, width: int, height: int) -> SkiaCanvas:
    obj = cls.__new__(cls)
    obj._ctx = None
    interface = skia.GrGLInterface.MakeEGL()
    obj._gr_context = skia.GrDirectContext.MakeGL(interface)
    fb_info = skia.GrGLFramebufferInfo(fbo_id, GL_RGBA8)
    backend_rt = skia.GrBackendRenderTarget(width, height, 0, 8, fb_info)
    obj._surface = skia.Surface.MakeFromBackendRenderTarget(
        obj._gr_context, backend_rt,
        skia.kTopLeft_GrSurfaceOrigin, skia.kRGBA_8888_ColorType, None,
    )
    return obj
```

Also update `skia_canvas.pyi` and `__init__.py` accordingly.

**Step 3 -- Rust PyOnRender wrapper** (in `picasso_py/src/callbacks.rs`):

The Rust `PyOnRender` struct acquires the GIL and passes `(source_id, fbo_id, width, height, frame)` to the Python callable:

```rust
struct PyOnRender(Py<PyAny>);

impl OnRender for PyOnRender {
    fn call(&self, source_id: &str, renderer: &SkiaRenderer, frame: &VideoFrameProxy) {
        let fbo_id = renderer.fbo_id();
        let width = renderer.width();
        let height = renderer.height();
        Python::with_gil(|py| {
            let py_frame = savant_core_py::primitives::frame::VideoFrame(frame.clone());
            if let Err(e) = self.0.call1(py, (source_id, fbo_id, width, height, py_frame)) {
                log::error!("on_render callback error: {e}");
            }
        });
    }
}
```

**Step 4 -- Python-side adapter** (in `picasso_py/python/picasso/__init__.py`):

Wraps the user's `(source_id, skia_canvas, frame)` callback, caching the `SkiaCanvas`:

```python
class _OnRenderAdapter:
    def __init__(self, user_cb):
        self._user_cb = user_cb
        self._canvas = None
        self._fbo_id = None

    def __call__(self, source_id, fbo_id, width, height, frame):
        if self._canvas is None or self._fbo_id != fbo_id:
            from deepstream_nvbufsurface import SkiaCanvas
            self._canvas = SkiaCanvas.from_fbo(fbo_id, width, height)
            self._fbo_id = fbo_id
        self._user_cb(source_id, self._canvas, frame)
```

The `Callbacks` Python class wraps the user's `on_render` with `_OnRenderAdapter` before passing to the native `PyCallbacks`.

**User-facing API**:

```python
from picasso import Callbacks, PicassoEngine
from deepstream_nvbufsurface import SkiaCanvas

def my_on_render(source_id: str, skia_canvas: SkiaCanvas, frame: VideoFrame) -> None:
    canvas = skia_canvas.canvas()  # skia-python Canvas
    canvas.drawRect(skia.Rect(10, 10, 200, 100), paint)

callbacks = Callbacks(on_render=my_on_render)
engine = PicassoEngine(general_spec, callbacks)
```

### GIL release

`PicassoEngine` methods (`send_frame`, `send_eos`, `set_source_spec`, `shutdown`) use `py.detach(|| ...)` to release the GIL during Rust-side work, following the pattern in `deepstream_nvbufsurface`.

## Dependencies in Cargo.toml

Current `Cargo.toml` already includes: `picasso`, `savant_core`, `savant_core_py`, `deepstream_encoders`, `deepstream_nvbufsurface`, `gstreamer`, `glib`, `pyo3`, `log`, `skia-safe`. We may also need `savant_gstreamer` for `PyCodec`. Add `parking_lot` if needed.
