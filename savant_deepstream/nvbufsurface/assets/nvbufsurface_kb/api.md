# NvBufSurface Public API

Crate: `deepstream_nvbufsurface`

---

## Top-Level Exports (lib.rs)

### Functions
| Function | Signature | Notes |
|---|---|---|
| `cuda_init` | `(gpu_id: u32) → Result<(), NvBufSurfaceError>` | Must call before creating generators outside DeepStream |
| `bridge_savant_id_meta` | `(element: &gst::Element)` | PTS-keyed meta bridge for encoders |
| `memset_surface` | `(view: &SurfaceView, value: u8) → Result<(), NvBufSurfaceError>` | Fill the first surface with a constant byte value. Platform-aware: uses CUDA driver API on dGPU, NvBufSurfaceMap on Jetson. Unsafety contained in `SurfaceView` construction. |
| `fill_surface` | `(view: &SurfaceView, color: &[u8]) → Result<(), NvBufSurfaceError>` | Fill surface with a per-pixel color pattern. `color.len()` must match `view.channels()`. Each pixel row is filled with the repeating color pattern. Platform-aware. |
| `upload_to_surface` | `(view: &SurfaceView, data: &[u8], width: u32, height: u32, channels: u32) → Result<(), NvBufSurfaceError>` | Upload CPU pixel data to the first surface. `channels`: 4 for RGBA, 3 for RGB. Row-by-row copy respecting GPU pitch. Platform-aware (CUDA on dGPU, NvBufSurfaceMap on Jetson). Unsafety contained in `SurfaceView` construction. ⚠ 5 args, not 4. |

### Enums
| Enum | Variants |
|---|---|
| `NvBufSurfaceMemType` | Default(0), CudaPinned(1), CudaDevice(2), CudaUnified(3), SurfaceArray(4), Handle(5), System(6) |

### Re-exports
```rust
pub use SavantIdMeta, SavantIdMetaKind;  // from savant_gstreamer
pub use VideoFormat;                      // from savant_gstreamer
pub use SharedMutableGstBuffer;           // from shared_buffer
pub use SurfaceView;                     // from surface_view
pub use CudaStream;                      // from cuda_stream
pub use extract_nvbufsurface, buffer_gpu_id, ComputeMode, DstPadding,
        Interpolation, Padding, Rect, TransformConfig, TransformError,
        MIN_EFFECTIVE_DIM;                                // from transform
#[cfg(feature = "skia")]
pub use SkiaRenderer;                                    // from skia_renderer
pub use DsNvSurfaceBufferGenerator, DsNvSurfaceBufferGeneratorBuilder;  // from buffers/single
pub use DsNvUniformSurfaceBufferGenerator, DsNvUniformSurfaceBufferGeneratorBuilder,
        DsNvUniformSurfaceBuffer;  // from buffers/batched/uniform
pub use DsNvNonUniformSurfaceBuffer;     // from buffers/batched/non_uniform
pub use fill_surface, memset_surface, upload_to_surface;  // from surface_ops
```

---

## SharedMutableGstBuffer

```rust
pub struct SharedMutableGstBuffer(Arc<parking_lot::Mutex<gst::Buffer>>);
```

Shared currency for passing NvBufSurface-backed buffers between `SurfaceView`,
Picasso, and the encoder without ownership transfer. Implements `Send + Sync`.

**Note:** Uses `parking_lot::Mutex`, not `std::sync::Mutex`.

| Trait / Method | Signature | Notes |
|---|---|---|
| `From<gst::Buffer>` | `impl From<gst::Buffer> for SharedMutableGstBuffer` | Wrap a buffer in shared handle |
| `Clone` | `fn clone(&self) -> Self` | Cheap Arc increment; no data copy |
| `lock` | `(&self) → MutexGuard<'_, gst::Buffer>` | Lock for read/write; guard auto-derefs to `&gst::Buffer` / `&mut gst::Buffer` |
| `into_buffer` | `(self) → Result<gst::Buffer, Self>` | Extract inner buffer; **fails** if other refs exist (returns `Err(self)`) |
| `strong_count` | `(&self) → usize` | Number of strong references; 1 = sole owner |
| `pts_ns` | `(&self) → Option<u64>` | Buffer PTS in nanoseconds, or `None` if unset |
| `set_pts_ns` | `(&self, pts_ns: u64)` | Set buffer PTS in nanoseconds |
| `duration_ns` | `(&self) → Option<u64>` | Buffer duration in nanoseconds, or `None` if unset |
| `set_duration_ns` | `(&self, duration_ns: u64)` | Set buffer duration in nanoseconds |
| `savant_ids` | `(&self) → Vec<(String, i64)>` | Read `SavantIdMeta` as `("frame"/"batch", id)` pairs |
| `Debug` | `impl Debug` | Prints `SharedMutableGstBuffer { strong_count: N }` |

---

## DsNvSurfaceBufferGenerator (single-frame)

```rust
pub struct DsNvSurfaceBufferGenerator { /* pool, format, w, h, fps, gpu_id */ }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(format, w, h, fps_num, fps_den, gpu_id, mem_type) → Result<Self, E>` | Simple constructor |
| `builder` | `(format, w, h) → DsNvSurfaceBufferGeneratorBuilder` | Advanced config |
| `acquire_surface` | `(&self, frame_id: Option<i64>) → Result<gst::Buffer, E>` | Acquire single buffer from pool. Attaches SavantIdMeta if frame_id given |
| `push_to_appsrc` | `(&self, appsrc: &AppSrc, ...) → Result<(), E>` | Acquire + push |
| `transform` | `(&self, src: &gst::Buffer, config: &TransformConfig, src_rect: Option<&Rect>) → Result<gst::Buffer, E>` | Acquire + GPU transform |
| `transform_with_ptr` | `(&self, src: &gst::Buffer, config: &TransformConfig, src_rect: Option<&Rect>) → Result<(gst::Buffer, *mut c_void, u32), E>` | Transform + return GPU ptr and pitch |
| `width` / `height` / `format` / `gpu_id` | `(&self) → T` | Getters |

Builder methods: `fps(num, den)`, `gpu_id(u32)`, `mem_type(NvBufSurfaceMemType)`,
`min_buffers(u32)`, `max_buffers(u32)`, `build()`.

---

## DsNvUniformSurfaceBufferGenerator (batched, homogeneous)

```rust
pub struct DsNvUniformSurfaceBufferGenerator { /* pool, format, w, h, gpu_id, max_batch_size */ }
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(format, w, h, max_batch_size, pool_size, gpu_id, mem_type) → Result<Self, E>` | Simple |
| `builder` | `(format, w, h, max_batch_size) → Builder` | Advanced, DEF pool_size=2 |
| `acquire_batched_surface` | `(&self, config: TransformConfig) → Result<DsNvUniformSurfaceBuffer, E>` | ⚠ `config` is moved |
| `acquire_buffer` | `(&self, frame_id: Option<i64>) → Result<SharedMutableGstBuffer, E>` | Acquires single-frame buffer, wraps in SharedMutableGstBuffer |
| `max_batch_size` / `width` / `height` / `format` / `gpu_id` | `(&self) → T` | Getters |

Builder methods: `fps(num, den)`, `gpu_id(u32)`, `mem_type(NvBufSurfaceMemType)`,
`pool_size(u32)`, `build()`.

---

## DsNvUniformSurfaceBuffer

```rust
pub struct DsNvUniformSurfaceBuffer {
    shared: SharedMutableGstBuffer,  // pool-allocated
    config: TransformConfig,   // stored at acquisition
    ids: Vec<Option<SavantIdMetaKind>>,
    max_batch_size: u32,
    num_filled: u32,
    finalized: bool,
}
```

| Method | Signature | Notes |
|---|---|---|
| `fill_slot` | `(&mut self, src: &gst::Buffer, src_rect: Option<&Rect>, id: Option<i64>) → Result<(), E>` | GPU transform to next slot. ⚠ Blocked after finalize |
| `slot_ptr` | `(&self, index: u32) → Result<(*mut c_void, u32), E>` | `(dataPtr, pitch)` for direct GPU writes |
| `num_filled` | `(&self) → u32` | Current count |
| `max_batch_size` | `(&self) → u32` | Max capacity |
| `finalize` | `(&mut self, num_filled: u32, ids: Vec<SavantIdMetaKind>) → Result<(), E>` | Sets numFilled + SavantIdMeta. ⚠ Double-call → AlreadyFinalized |
| `is_finalized` | `(&self) → bool` | |
| `shared_buffer` | `(&self) → SharedMutableGstBuffer` | Clone of Arc for shared access; use for downstream consumers |
| `slot_view` | `(&self, index: u32) → Result<SurfaceView, E>` | Creates SurfaceView for a specific slot |

**REMOVED:** `as_gst_buffer`, `extract_slot_view`. Use `shared_buffer()` and `SurfaceView::from_shared` instead.

---

## DsNvNonUniformSurfaceBuffer (batched, heterogeneous)

```rust
pub struct DsNvNonUniformSurfaceBuffer {
    params: Vec<NvBufSurfaceParams>,
    parents: Vec<SharedMutableGstBuffer>,
    gpu_id: u32,
}
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(gpu_id: u32) → Self` | Constructor; no max_batch_size, no Result |
| `add` | `(&mut self, src: &SurfaceView, id: Option<i64>) → Result<(), E>` | Zero-copy: copies params descriptor, stores SharedMutableGstBuffer to keep source alive |
| `num_filled` | `(&self) → u32` | Number of slots added |
| `gpu_id` | `(&self) → u32` | GPU device ID |
| `finalize` | `(self, ids: Vec<SavantIdMetaKind>) → Result<SharedMutableGstBuffer, E>` | Consuming. Allocates buffer, attaches GstParentBufferMeta, returns SharedMutableGstBuffer |

**REMOVED:** `as_gst_buffer`, `extract_slot_view`, `slot_ptr`, `max_batch_size`, `is_finalized`.

---

## SurfaceView

```rust
pub struct SurfaceView {
    buffer: SharedMutableGstBuffer,
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
Holds `SharedMutableGstBuffer` + `slot_index` internally. On dGPU, `data_ptr` is read
directly from `NvBufSurfaceParams::dataPtr`. On Jetson (aarch64), construction
transparently attaches `EglCudaMeta` per slot for **permanent mapping**. No longer
uses `extract_slot_view` internally.

| Constructor | Signature | Notes |
|---|---|---|
| `from_buffer` | `(buf: gst::Buffer, slot_index: u32) → Result<Self, E>` | Wraps buffer in `SharedMutableGstBuffer`, resolves CUDA ptr for slot. |
| `from_shared` | `(buf: &SharedMutableGstBuffer, slot_index: u32) → Result<Self, E>` | **Primary constructor** for batched access. Borrows buf, clones Arc internally. Create one view per slot. |
| `from_cuda_ptr` | `(data_ptr, pitch, w, h, gpu_id, channels, color_format, keepalive) → Result<Self, E>` | Synthetic descriptor around raw CUDA ptr |
| `wrap` | `(buf: gst::Buffer) → Self` | `#[cfg(any(test, feature = "testing"))]` only. Wrap plain buffer without NvBufSurface validation (zeroed params). |

| Accessor | Signature |
|---|---|
| `buffer` | `(&self) → MutexGuard<'_, gst::Buffer>` | Lock for read/write; replaces old `buffer()` and `buffer_mut()`. |
| `shared_buffer` | `(&self) → SharedMutableGstBuffer` | Clone of internal handle; for sibling views or encoder. |
| `slot_index` | `(&self) → u32` | Batch slot index this view refers to. |
| `into_buffer` | `(self) → Result<gst::Buffer, Self>` | Extract buffer; **fails** if other refs exist (returns `Err(self)`). |
| `cuda_stream` | `(&self) → &CudaStream` | CUDA stream for synchronization on release |
| `with_cuda_stream` | `(self, stream: CudaStream) → Self` | Override the stream; chainable |

| Method | Signature | Notes |
|---|---|---|
| `transform_into` | `(&self, dest: &SurfaceView, config: &TransformConfig, src_rect: Option<&Rect>) → Result<(), NvBufSurfaceError>` | GPU-to-GPU transform replacing old fill_slot/transform APIs |

| `data_ptr` | `(&self) → *mut c_void` |
| `pitch` / `width` / `height` / `gpu_id` / `channels` / `color_format` | `(&self) → u32` |

---

## EglCudaMeta (aarch64 only)

Custom `GstMeta` for EGL-CUDA interop on Jetson. **Multi-slot**: stores
`slots: [SlotRegistration; MAX_BATCH_SLOTS]` (32 slots) and `batch_size`.
Each slot is registered lazily on first access. Automatically deregisters
all slots when the buffer is freed (via `meta_free`).

```rust
// Module: egl_cuda_meta (conditionally compiled: #[cfg(target_arch = "aarch64")])

pub const MAX_BATCH_SLOTS: usize = 32;

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
| `tracking_counts` | `#[cfg(test)]` only: returns `(registrations, deregistrations)`; counts are per individual slot. |
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
| `synchronize` | `(&self)` | Block until all enqueued work completes |

| Trait | Behaviour |
|---|---|
| `Clone` | Always produces a **non-owning** copy |
| `Drop` | Destroys the stream only if **owned** and non-null |
| `Default` | Legacy default stream (null), non-owning |

---

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

---

## Padding / Interpolation / ComputeMode

```rust
pub enum Padding { None, RightBottom, #[default] Symmetric }
pub enum Interpolation { Nearest, #[default] Bilinear, Algo1, Algo2, Algo3, Algo4, Default }
pub enum ComputeMode { #[default] Default, Gpu, Vic }
```

All have `from_str_name(&str) → Option<Self>` for CLI parsing.

---

## Rect

```rust
pub struct Rect { pub top: u32, pub left: u32, pub width: u32, pub height: u32 }
```

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
    temp_gen: Option<DsNvSurfaceBufferGenerator>,
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
