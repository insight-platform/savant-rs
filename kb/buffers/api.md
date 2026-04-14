# NvBufSurface Public API

Crate: `deepstream_buffers`

---

## Prelude

`use deepstream_buffers::prelude::*;` re-exports the crate's primary runtime and transform types (`BufferGenerator`, `UniformBatchGenerator`, `SurfaceBatch`, `NonUniformBatch`, `SharedBuffer`, `SurfaceView`, `TransformConfig`, `NvBufSurfaceError`, `VideoFormat`, etc).

---

## Top-Level Exports (lib.rs)

### Functions
| Function | Signature | Notes |
|---|---|---|
| `cuda_init` | `(gpu_id: u32) → Result<(), NvBufSurfaceError>` | Must call before creating generators outside DeepStream |
| `bridge_savant_id_meta` | `(element: &gst::Element) → Result<(), NvBufSurfaceError>` | PTS-keyed meta bridge for encoders. Returns `Err(MissingPad("sink"|"src"))` if the element lacks the required static pads |

### Enums
| Enum | Variants |
|---|---|
| `NvBufSurfaceMemType` | Default(0), CudaPinned(1), CudaDevice(2), CudaUnified(3), SurfaceArray(4), Handle(5), System(6) |

### Re-exports
```rust
pub use SavantIdMetaKind;                // from savant_gstreamer
pub use VideoFormat;                      // from savant_gstreamer
pub use SharedBuffer;           // from shared_buffer
pub use SurfaceView;                     // from surface_view
pub use CudaStream;                      // from cuda_stream
pub use extract_nvbufsurface, buffer_gpu_id, ComputeMode, DstPadding,
        Interpolation, Padding, Rect, TransformConfig, TransformError,
        MIN_EFFECTIVE_DIM;                                // from transform
#[cfg(feature = "skia")]
pub use SkiaRenderer;                                    // from skia_renderer
pub use BufferGenerator, BufferGeneratorBuilder;  // from buffers/single
pub use UniformBatchGenerator, UniformBatchGeneratorBuilder,
        SurfaceBatch;  // from buffers/batched/uniform
pub use NonUniformBatch;     // from buffers/batched/non_uniform (re-exported via buffers::*)
pub use BatchState;          // from buffers/batch_state
pub use pipeline::{BufferGeneratorExt, UniformBatchGeneratorExt};  // from pipeline.rs
```

---

## SharedBuffer

`SharedBuffer` is `Arc<parking_lot::Mutex<gst::Buffer>>` wrapper used as the ownership currency across buffers/decoders/infer/tracker stages.

Key methods:

- `lock(&self) -> MutexGuard<'_, gst::Buffer>`
- `into_buffer(self) -> Result<gst::Buffer, Self>` (requires sole ownership)
- `strong_count(&self) -> usize`
- `pts_ns`, `set_pts_ns`, `duration_ns`, `set_duration_ns`
- `savant_ids`, `set_savant_ids`
- `with_view(slot, f)` (scoped `SurfaceView`)
- `transform_into(...)`

---

## BufferGenerator (single-frame)

Thin wrapper around `UniformBatchGenerator` with `max_batch_size=1`.
No duplicate pool logic — all buffer allocation delegates to the inner generator.

```rust
pub struct BufferGenerator { inner: UniformBatchGenerator }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(format, w, h, fps_num, fps_den, gpu_id, mem_type) → Result<Self, E>` | Simple constructor |
| `builder` | `(format, w, h) → BufferGeneratorBuilder` | Advanced config |
| `acquire` | `(&self, id: Option<u128>) → Result<SharedBuffer, E>` | Acquire single buffer from pool. Attaches SavantIdMeta if id given |
| `try_acquire` | `(&self, id: Option<u128>) → Result<SharedBuffer, E>` | Non-blocking acquire (`PoolExhausted` when unavailable) |
| `transform` | `(&self, src: &SurfaceView, config: &TransformConfig, src_rect: Option<&Rect>) → Result<SharedBuffer, E>` | Acquire + GPU transform |
| `transform_to_buffer` | `(&self, src: &SurfaceView, config: &TransformConfig, src_rect: Option<&Rect>) -> Result<gst::Buffer, E>` | Acquire + transform + extract when sole owner |
| `width` / `height` / `format` / `gpu_id` | `(&self) → T` | Getters |
| `nvmm_caps` | `(&self) → String` | NVMM caps string (delegates to inner `UniformBatchGenerator`) |
| `raw_caps` | `(&self) → String` | Raw (non-NVMM) caps string |

Builder methods: `fps(num, den)`, `gpu_id(u32)`, `mem_type(NvBufSurfaceMemType)`,
`min_buffers(u32)`, `max_buffers(u32)`, `build()`.

### BufferGeneratorExt (pipeline.rs)

Extension trait providing GStreamer-level caps access:

| Method | Signature | Notes |
|---|---|---|
| `nvmm_caps_gst` | `(&self) → gst::Caps` | NVMM caps as `gst::Caps` object |
| `raw_caps_gst` | `(&self) → gst::Caps` | Raw caps as `gst::Caps` object |

### gst_app (free functions)

AppSrc helpers extracted from `BufferGenerator` into `deepstream_buffers::buffers::single::gst_app`.

| Function | Signature | Notes |
|---|---|---|
| `push_to_appsrc` | `(gen: &BufferGenerator, appsrc: &AppSrc, pts_ns: u64, duration_ns: u64, id: Option<u128>) → Result<(), E>` | Acquire + set PTS/duration + push |
| `send_eos` | `(appsrc: &AppSrc) → Result<(), E>` | Send EOS to AppSrc |

---

## UniformBatchGenerator (batched, homogeneous)

```rust
pub struct UniformBatchGenerator { /* pool, format, w, h, gpu_id, max_batch_size */ }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(format, w, h, max_batch_size, pool_size, gpu_id, mem_type) → Result<Self, E>` | Simple |
| `builder` | `(format, w, h, max_batch_size) → Builder` | Advanced, DEF pool_size=2 |
| `acquire_batch` | `(&self, config: TransformConfig, ids: Vec<SavantIdMetaKind>) → Result<SurfaceBatch, E>` | Main acquisition path |
| `try_acquire_batch` | `(&self, config: TransformConfig, ids: Vec<SavantIdMetaKind>) -> Result<SurfaceBatch, E>` | Non-blocking acquire (`PoolExhausted` when unavailable) |
| `max_batch_size` / `width` / `height` / `format` / `gpu_id` | `(&self) → T` | Getters |
| `nvmm_caps` | `(&self) → String` | NVMM caps string (includes `memory:NVMM` feature) |
| `nvmm_caps_gst` | `(&self) → gst::Caps` | NVMM caps as `gst::Caps` object |
| `raw_caps` | `(&self) → String` | Raw (non-NVMM) caps string |

**`acquire()` is `pub(crate)` only** — used internally by `BufferGenerator` (which
wraps `UniformBatchGenerator` with `max_batch_size=1`). External callers must use
`acquire_batch()` for batched buffers or `BufferGenerator::acquire()` for single frames.

Builder methods: `fps(num, den)`, `gpu_id(u32)`, `mem_type(NvBufSurfaceMemType)`,
`pool_size(u32)` (sets both min/max), `min_buffers(u32)`, `max_buffers(u32)`, `build()`.

### UniformBatchGeneratorExt (pipeline.rs)

Extension trait providing GStreamer-level caps access:

| Method | Signature | Notes |
|---|---|---|
| `nvmm_caps_gst` | `(&self) → gst::Caps` | NVMM caps as `gst::Caps` object |
| `raw_caps_gst` | `(&self) → gst::Caps` | Raw caps as `gst::Caps` object |

---

## SurfaceBatch (uniform batched)

```rust
pub struct SurfaceBatch {
    buffer: SharedBuffer,  // pool-allocated
    config: TransformConfig,   // stored at acquisition
    max_batch_size: u32,
    num_filled: u32,
    finalized: bool,
}
```

| Method | Signature | Notes |
|---|---|---|
| `transform_slot` | `(&mut self, slot: u32, src: &SurfaceView, src_rect: Option<&Rect>) → Result<(), E>` | GPU transform to slot. ⚠ Blocked after finalize |
| `num_filled` | `(&self) → u32` | One past highest slot written |
| `max_batch_size` | `(&self) → u32` | Max capacity |
| `finalize` | `(&mut self) → Result<(), E>` | Marks batch immutable and validates state |
| `is_finalized` | `(&self) → bool` | |
| `shared_buffer` | `(&self) → SharedBuffer` | Clone of Arc for shared access; use for downstream consumers |
| `view` | `(&self, index: u32) → Result<SurfaceView, E>` | Creates SurfaceView for a specific slot |
| `into_shared_buffer` | `(self) → SharedBuffer` | Consume batch and return inner SharedBuffer |

**REMOVED:** `as_gst_buffer`, `extract_slot_view`. Use `shared_buffer()` and `SurfaceView::from_buffer` instead.

---

## NonUniformBatch (batched, heterogeneous)

```rust
pub struct NonUniformBatch {
    params: Vec<NvBufSurfaceParams>,
    parents: Vec<SharedBuffer>,
    gpu_id: u32,
}
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(gpu_id: u32) → Self` | Constructor; no max_batch_size, no Result |
| `add` | `(&mut self, src: &SurfaceView) → Result<(), E>` | Zero-copy: copies params descriptor, stores SharedBuffer to keep source alive |
| `num_filled` | `(&self) → u32` | Number of slots added |
| `gpu_id` | `(&self) → u32` | GPU device ID |
| `finalize` | `(self, ids: Vec<SavantIdMetaKind>) → Result<SharedBuffer, E>` | Consuming. Allocates buffer, attaches GstParentBufferMeta, returns SharedBuffer |

**REMOVED:** `as_gst_buffer`, `extract_slot_view`, `slot_ptr`, `max_batch_size`, `is_finalized`.

---

## SurfaceView

```rust
pub struct SurfaceView {
    buffer: SharedBuffer,
    slot_index: u32,
    cuda_stream: CudaStream,
    _keepalive: Option<Box<dyn Any + Send + Sync>>,
    data_ptr: *mut c_void,
    pitch: u32, width: u32, height: u32,
    gpu_id: u32, channels: u32, color_format: u32,
}
```

Implements `Send + Sync`, `Debug`.

**The canonical single entry point for CUDA memory access** on both dGPU and Jetson.
Holds `SharedBuffer` + `slot_index` internally. On dGPU, `data_ptr` is read
directly from `NvBufSurfaceParams::dataPtr`. On Jetson (aarch64), construction
transparently attaches `EglCudaMeta` per slot for **permanent mapping**. No longer
uses `extract_slot_view` internally.

| Constructor | Signature | Notes |
|---|---|---|
| `from_gst_buffer` | `(buf: gst::Buffer, slot_index: u32) → Result<Self, E>` | Wraps buffer in `SharedBuffer`, resolves CUDA ptr for slot. |
| `from_buffer` | `(buf: &SharedBuffer, slot_index: u32) → Result<Self, E>` | **Primary constructor** for batched access. Borrows buf, clones Arc internally. Create one view per slot. |
| `from_cuda_ptr` | `(data_ptr, pitch, w, h, gpu_id, channels, color_format, keepalive) → Result<Self, E>` | Synthetic descriptor around raw CUDA ptr |
| `wrap` | `(buf: gst::Buffer) → Self` | Wrap plain buffer without NvBufSurface validation (zeroed params). For stubs/tests; real GPU paths need `from_buffer` / `from_gst_buffer`. |

| Accessor | Signature |
|---|---|
| `gst_buffer` | `(&self) → MutexGuard<'_, gst::Buffer>` | Lock for read/write access to the underlying buffer. |
| `shared_buffer` | `(&self) → SharedBuffer` | Clone of internal handle; for sibling views or encoder. |
| `slot_index` | `(&self) → u32` | Batch slot index this view refers to. |
| `into_gst_buffer` | `(self) → Result<gst::Buffer, Self>` | Extract buffer; **fails** if other refs exist (returns `Err(self)`). |

| Method | Signature | Notes |
|---|---|---|
| `memset` | `(&self, value: u8) → Result<(), NvBufSurfaceError>` | Fill surface with constant byte value. Uses CUDA driver API. |
| `fill` | `(&self, color: &[u8]) → Result<(), NvBufSurfaceError>` | Fill with repeating pixel colour. `color.len()` must match `channels()`. Supports 1 (GRAY8) and 4 (RGBA) channels. |
| `upload` | `(&self, data: &[u8], width: u32, height: u32, channels: u32) → Result<(), NvBufSurfaceError>` | Upload CPU pixel data. Row-by-row copy respecting GPU pitch. ⚠ 5 args: data, width, height, channels. |
| `cuda_stream` | `(&self) → &CudaStream` | CUDA stream for synchronization on release |
| `with_cuda_stream` | `(self, stream: CudaStream) → Self` | Override the stream; chainable |

| Method | Signature | Notes |
|---|---|---|
| `transform_into` | `(&self, dest: &SurfaceView, config: &TransformConfig, src_rect: Option<&Rect>) → Result<(), NvBufSurfaceError>` | GPU-to-GPU transform replacing old transform_slot/transform APIs |

| `data_ptr` | `(&self) → *mut c_void` |
| `pitch` / `width` / `height` / `gpu_id` / `channels` / `color_format` | `(&self) → u32` |

---

## EglCudaMeta (aarch64 only)

Custom `GstMeta` for EGL-CUDA interop on Jetson. **Multi-slot**: stores
`slots: [SlotRegistration; MAX_BATCH_SLOTS]` (64 slots) and `batch_size`.
Each slot is registered lazily on first access. Automatically deregisters
all slots when the buffer is freed (via `meta_free`).

**`MAX_BATCH_SLOTS` rationale:** This is a **Savant fixed-size cap** for the meta's
embedded slot array, not an NVIDIA hardware limit. `NvBufSurface.batchSize` is a `u32`
with no documented maximum in `nvbufsurface.h` (DeepStream SDK API Reference). We use a
fixed array so `gst_meta_register` has a stable size and we avoid heap allocation for
the per-slot EGL-CUDA registration state on every buffer. Larger values increase
per-buffer `GstMeta` memory; the default tracks typical DeepStream batching (including
larger nvinfer batch sizes).

```rust
// Module: egl_cuda_meta (conditionally compiled: #[cfg(target_arch = "aarch64")])

pub const MAX_BATCH_SLOTS: usize = 64;

pub struct EglCudaMapping {
    pub cuda_ptrs: [*mut c_void; 3],
    pub pitches: [u32; 3],
    pub plane_count: u32,
}

// EglCudaMetaInner (internal): surf_ptr, batch_size, slots[MAX_BATCH_SLOTS]

pub unsafe fn ensure_meta(buf: &mut BufferRef, slot_index: u32) → Result<EglCudaMapping, NvBufSurfaceError>
pub fn read_meta(buf: &BufferRef, slot_index: u32) → Option<EglCudaMapping>
```

| Function | Notes |
|---|---|
| `ensure_meta(buf, slot_index)` | Per-slot lazy registration. Attaches meta on first call; registers only the requested slot. Returns cached mapping if slot already registered. Performs `NvBufSurfaceMapEglImage(surf_ptr, slot_index)` → `cuGraphicsEGLRegisterImage` → `cuGraphicsResourceGetMappedEglFrame`. Sets `GST_META_FLAG_POOLED | GST_META_FLAG_LOCKED` so meta **survives GstBufferPool recycles**. |
| `read_meta(buf, slot_index)` | Per-slot read; returns `None` if meta absent or slot not registered. |
| `tracking_counts` | Returns `(registrations, deregistrations)`; counts are per individual slot (aarch64 / `egl_cuda_meta` only). |
| `meta_free` | Iterates all slots, deregisters each with non-null `resource` via `cuGraphicsUnregisterResource` and `NvBufSurfaceUnMapEglImage(surf_ptr, i)`. |

**Key implementation notes:**
- On Jetson, `cuGraphicsEGLRegisterImage` creates a **permanent implicit mapping** — `cuGraphicsUnmapResources` returns error 999. No RAII map/unmap cycle.
- **POOLED flag:** Meta survives pool recycles. Pool of 1 buffer, N acquisitions = 1 registration per slot (first time), 0 re-registrations. Deregistration only on pool destroy.
- Thread safety: `ensure_cuda_egl_context` ensures CUDA driver context and EGL display are initialized per-thread.

---

## CudaStream

```rust
pub struct CudaStream {
    raw: *mut c_void,
    owned: bool,
}
```

Safe RAII wrapper around a CUDA stream handle. Implements `Send`, `Sync`, `Debug`, `Default`, `Clone`.

| Constructor | Signature | Notes |
|---|---|---|
| `new_non_blocking` | `() → Result<Self, NvBufSurfaceError>` | Creates owned non-blocking stream |
| `from_raw` | `unsafe (ptr: *mut c_void) → Self` | Wrap existing handle, non-owning |
| `Default::default` | `() → Self` | Legacy default stream (null), non-owning |

| Accessor | Signature | Notes |
|---|---|---|
| `as_raw` | `(&self) → *mut c_void` | Raw CUDA stream pointer |
| `is_default` | `(&self) → bool` | True if null (legacy default stream) |
| `is_owned` | `(&self) → bool` | True if this handle will destroy the stream on drop |
| `synchronize` | `(&self) → Result<(), NvBufSurfaceError>` | Block until all enqueued work completes; returns error on CUDA failure |
| `synchronize_or_log` | `(&self)` | Same as `synchronize()` but logs a warning instead of returning an error |

| Trait | Behaviour |
|---|---|
| `Clone` | Always produces a **non-owning** copy |
| `Drop` | Destroys the stream only if **owned** and non-null |
| `Default` | Legacy default stream (null), non-owning |

---

## BatchState

Generic queue/deadline helper reused by batching operators:

- `BatchState::new()`
- `take(&mut self) -> Vec<T>`
- `is_empty(&self) -> bool`

## TransformConfig

```rust
#[derive(Debug, Clone)]
pub struct TransformConfig {
    pub padding: Padding,                  // DEF: Symmetric
    pub dst_padding: Option<DstPadding>,   // DEF: None
    pub interpolation: Interpolation,      // DEF: Bilinear
    pub compute_mode: ComputeMode,         // DEF: Default
    pub cuda_stream: CudaStream,           // DEF: CudaStream::default() (legacy default stream)
}
```

Implements `Default`, `Send`, `Sync`, `Clone` (not `Copy`).

| Constructor | Signature | Notes |
|---|---|---|
| `builder` | `() → TransformConfigBuilder` | Fluent builder starting from `Default` |

### TransformConfigBuilder

Builder methods (all return `Self`): `padding(Padding)`, `dst_padding(DstPadding)`,
`interpolation(Interpolation)`, `compute_mode(ComputeMode)`, `cuda_stream(CudaStream)`,
`build() → TransformConfig`.

---

## DstPadding

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DstPadding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}
```

Optional per-side destination padding. When set in `TransformConfig::dst_padding`,
reduces the effective destination area before the letterbox rect is computed.
Padding regions are filled with black. Validation: `pad_left + pad_right < dst_w`
and `pad_top + pad_bottom < dst_h`.

| Constructor | Signature | Notes |
|---|---|---|
| `new` | `(left: u32, top: u32, right: u32, bottom: u32) → Self` | Per-side values |
| `uniform` | `(value: u32) → Self` | Same value on all four sides |

---

## Padding / Interpolation / ComputeMode

```rust
pub enum Padding { None, RightBottom, #[default] Symmetric }
pub enum Interpolation {
    Nearest, #[default] Bilinear,
    GpuCubicVic5Tap, GpuSuperVic10Tap, GpuLanczosVicSmart, GpuIgnoredVicNicest,
    Default,
}
pub enum ComputeMode { #[default] Default, Gpu, Vic }
```

`Interpolation` and `Padding` implement `FromStr` for CLI parsing (e.g. `"gpu_cubic_vic_5tap".parse::<Interpolation>()`).
Legacy names `"cubic"`, `"super"`, `"lanczos"`, `"nicest"` and `"algo1"`–`"algo4"` are accepted for backward compatibility.
Underscores are stripped before matching, so `"GPU_CUBIC_VIC_5TAP"` and `"GpuCubicVic5Tap"` both work.

**Python API:** `Padding.from_name(name)` and `Interpolation.from_name(name)` parse a string name into the corresponding enum variant (raises `ValueError` on unknown input).

---

## Rect

```rust
pub struct Rect { pub top: u32, pub left: u32, pub width: u32, pub height: u32 }
```

| Constructor | Signature | Notes |
|---|---|---|
| `new` | `(left: u32, top: u32, width: u32, height: u32) → Self` | Convenience constructor |

---

## extract_nvbufsurface

```rust
pub unsafe fn extract_nvbufsurface(buf: &BufferRef) → Result<*mut NvBufSurface, TransformError>
```

Maps buffer, reads NvBufSurface pointer. ⚠ Pointer valid while GstMemory exists.

---

## buffer_gpu_id

```rust
pub fn buffer_gpu_id(buf: &BufferRef) → Result<u32, TransformError>
```

Extract `gpuId` from NvBufSurface inside a GStreamer buffer.

---

## EglHeadlessContext (feature = "skia")

Headless EGL display/context for OpenGL without a window system.

| Method | Signature | Notes |
|---|---|---|
| `new` | `() → Result<Self, EglError>` | Device platform + surfaceless context. |
| `get_proc_address` | `(name: &str) → *const c_void` | Wraps `eglGetProcAddress`. Returns null if `name` contains an interior NUL (invalid C string) or if the function is not present — same as EGL for unknown symbols. |

---

## SkiaRenderer (feature = "skia")

```rust
pub struct SkiaRenderer {
    surface: skia_safe::Surface,
    gr_context: skia_safe::gpu::DirectContext,
    _egl: EglHeadlessContext,
    gl_texture: u32,
    gl_fbo: u32,
    cuda_resource: cudaGraphicsResource_t,
    width: u32,
    height: u32,
    gpu_id: u32,
    temp_gen: Option<BufferGenerator>,
    cuda_stream: CudaStream,
}
```

GPU-accelerated Skia renderer with CUDA-GL interop for NvBufSurface.
`!Send` and `!Sync` — all operations must occur on the creating thread.

Drop order: Skia surface → `DirectContext` → EGL context. Explicit GL cleanup
(FBO, texture) and CUDA resource unregister in `Drop`.

| Constructor | Signature | Notes |
|---|---|---|
| `new` | `(width: u32, height: u32, gpu_id: u32) → Result<Self, SkiaRendererError>` | Headless EGL + GL texture/FBO + CUDA-GL interop + Skia DirectContext + Surface |
| `unsafe from_nvbuf` | `(width: u32, height: u32, gpu_id: u32, data_ptr: *const c_void, pitch: usize) → Result<Self, SkiaRendererError>` | `new` + `load_from_nvbuf` in one call. Caller supplies CUDA pointer (e.g. from `SurfaceView`). |

| Method | Signature | Notes |
|---|---|---|
| `unsafe load_from_nvbuf` | `(&mut self, data_ptr: *const c_void, pitch: usize) → Result<(), SkiaRendererError>` | GPU-to-GPU copy of existing pixels INTO the GL texture so Skia can draw on top. Uses `cudaMemcpy2DToArray` device-to-device. |
| `canvas` | `(&mut self) → &skia_safe::Canvas` | Skia canvas for drawing. |
| `width` | `(&self) → u32` | Render target width. |
| `height` | `(&self) → u32` | Render target height. |
| `fbo_id` | `(&self) → u32` | OpenGL FBO ID (needed by Python skia-python to create its own Surface). |
| `with_cuda_stream` | `(self, stream: CudaStream) → Self` | Builder-style CUDA stream override. |
| `set_cuda_stream` | `(&mut self, stream: CudaStream)` | Replace CUDA stream on existing renderer. |
| `render_to_nvbuf` | `(&mut self, dst_buf: &mut BufferRef, config: Option<&TransformConfig>) → Result<(), SkiaRendererError>` | Convenience method: creates a `SurfaceView` internally to resolve the CUDA pointer, **keeps the view alive** during rendering (critical on Jetson where COW buffer copy holds the EGL-CUDA meta). Delegates to `render_to_nvbuf_with_ptr`. |
| `unsafe render_to_nvbuf_with_ptr` | `(&mut self, dst_buf: &mut BufferRef, dst_ptr: *mut c_void, dst_pitch: usize, config: Option<&TransformConfig>) → Result<(), SkiaRendererError>` | **Primary API.** Fast path (no scaling): direct CUDA-GL copy when dimensions match and config is `None`. Scaled path: GL → temp RGBA NvBufSurface → `NvBufSurfTransform` → dst. Caller supplies `(dst_ptr, dst_pitch)`. |
| `unsafe render_to_nvbuf_raw` | `(&mut self, data_ptr: *mut c_void, pitch: u32) → Result<(), SkiaRendererError>` | Direct 1:1 GPU-to-GPU copy, no scaling. Available on **all platforms**. |

### SkiaRendererError

```rust
pub enum SkiaRendererError {
    Egl(#[from] EglError),
    Gl(String),
    Cuda(i32, String),
    Skia(String),
    NvBuf(String),
}
```

| Variant | Trigger |
|---|---|
| `Egl` | EGL context creation failure (from `EglError`) |
| `Gl` | OpenGL error during texture/FBO creation |
| `Cuda` | `cudaGraphicsGLRegisterImage`, `cudaGraphicsMapResources`, `cudaMemcpy2D*` failures |
| `Skia` | Skia `DirectContext` or `Surface` creation failure |
| `NvBuf` | `SurfaceView` creation failure, `extract_nvbufsurface` failure, temp generator creation/acquire failure, `NvBufSurfTransform` failure |
