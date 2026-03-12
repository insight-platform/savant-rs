# NvBufSurface Public API

Crate: `deepstream_nvbufsurface`

---

## Top-Level Exports (lib.rs)

### Functions
| Function | Signature | Notes |
|---|---|---|
| `cuda_init` | `(gpu_id: u32) → Result<(), NvBufSurfaceError>` | Must call before creating generators outside DeepStream |
| `create_cuda_stream` | `() → Result<*mut c_void, NvBufSurfaceError>` | Non-blocking CUDA stream |
| `destroy_cuda_stream` | `unsafe (stream: *mut c_void) → Result<(), NvBufSurfaceError>` | Null is no-op |
| `bridge_savant_id_meta` | `(element: &gst::Element)` | PTS-keyed meta bridge for encoders |
| `memset_surface` | `unsafe (buf: &gst::Buffer, value: u8) → Result<(), NvBufSurfaceError>` | Fill the first surface in buf with a constant byte value. Platform-aware: uses CUDA driver API on dGPU, NvBufSurfaceMap on Jetson. |
| `upload_to_surface` | `unsafe (buf: &gst::Buffer, data: &[u8], width: u32, height: u32, channels: u32) → Result<(), NvBufSurfaceError>` | Upload CPU pixel data to the first surface in buf. Validates that `channels` matches the surface color format. Row-by-row copy respecting GPU pitch. Platform-aware. |

### Enums
| Enum | Variants |
|---|---|
| `NvBufSurfaceMemType` | Default(0), CudaPinned(1), CudaDevice(2), CudaUnified(3), SurfaceArray(4), Handle(5), System(6) |

### Re-exports
```rust
pub use SavantIdMeta, SavantIdMetaKind;  // from savant_gstreamer
pub use VideoFormat;                      // from savant_gstreamer
pub use SurfaceView;                     // from surface_view
pub use extract_nvbufsurface, buffer_gpu_id, ComputeMode, Interpolation,
        Padding, Rect, TransformConfig, TransformError;  // from transform
pub use DsNvSurfaceBufferGenerator, DsNvSurfaceBufferGeneratorBuilder;  // from buffers/single
pub use DsNvUniformSurfaceBufferGenerator, DsNvUniformSurfaceBufferGeneratorBuilder,
        DsNvUniformSurfaceBuffer, set_num_filled;  // from buffers/batched/uniform
pub use DsNvNonUniformSurfaceBuffer;     // from buffers/batched/non_uniform
pub use extract_slot_view;               // from buffers/batched/slot_view
pub use memset_surface, upload_to_surface;  // from surface_ops
```

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
| `max_batch_size` / `width` / `height` / `format` / `gpu_id` | `(&self) → T` | Getters |

Builder methods: `fps(num, den)`, `gpu_id(u32)`, `mem_type(NvBufSurfaceMemType)`,
`pool_size(u32)`, `build()`.

---

## DsNvUniformSurfaceBuffer

```rust
pub struct DsNvUniformSurfaceBuffer {
    buffer: gst::Buffer,       // pool-allocated
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
| `finalize` | `(&mut self) → Result<(), E>` | Non-consuming. Sets numFilled + SavantIdMeta. ⚠ Double-call → AlreadyFinalized |
| `is_finalized` | `(&self) → bool` | |
| `as_gst_buffer` | `(&self) → Result<gst::Buffer, E>` | Self-contained wrapper with inlined surfaceList + GstParentBufferMeta. ⚠ Requires finalize() first |
| `extract_slot_view` | `(&self, slot_index: u32) → Result<gst::Buffer, E>` | Zero-copy single-frame view. ⚠ Requires finalize() first |

---

## DsNvNonUniformSurfaceBuffer (batched, heterogeneous)

```rust
pub struct DsNvNonUniformSurfaceBuffer {
    buffer: gst::Buffer,       // system memory, self-contained
    ids: Vec<Option<SavantIdMetaKind>>,
    max_batch_size: u32,
    num_filled: u32,
    gpu_id: u32,
    finalized: bool,
}
```

| Method | Signature | Notes |
|---|---|---|
| `new` | `(max_batch_size: u32, gpu_id: u32) → Result<Self, E>` | Allocates system memory |
| `add` | `(&mut self, src: &gst::Buffer, id: Option<i64>) → Result<(), E>` | Zero-copy: copies params descriptor, adds GstParentBufferMeta to keep source alive. ⚠ Blocked after finalize |
| `slot_ptr` | `(&self, index: u32) → Result<(*mut c_void, u32, u32, u32), E>` | `(dataPtr, pitch, width, height)`. ⚠ Requires finalize() first |
| `num_filled` / `max_batch_size` / `gpu_id` | `(&self) → T` | Getters |
| `finalize` | `(&mut self) → Result<(), E>` | Non-consuming. ⚠ Double-call → AlreadyFinalized |
| `is_finalized` | `(&self) → bool` | |
| `as_gst_buffer` | `(&self) → Result<gst::Buffer, E>` | Returns `self.buffer.clone()` (refcount+1, zero-copy). ⚠ Requires finalize() first |
| `extract_slot_view` | `(&self, slot_index: u32) → Result<gst::Buffer, E>` | Zero-copy single-frame view. ⚠ Requires finalize() first |

---

## extract_slot_view (free function)

```rust
pub fn extract_slot_view(
    batch: &gst::Buffer,
    slot_index: u32,
) → Result<gst::Buffer, NvBufSurfaceError>
```

Creates a system-memory buffer with `batchSize=1, numFilled=1`, inlined
`surfaceList[0]` from the batch slot, `GstParentBufferMeta` to keep batch
alive. Propagates PTS, DTS, duration, offset, and per-slot SavantIdMeta.

---

## SurfaceView

```rust
pub struct SurfaceView {
    buffer: gst::Buffer,
    _keepalive: Option<Box<dyn Any + Send + Sync>>,
    data_ptr: *mut c_void,
    pitch: u32, width: u32, height: u32,
    gpu_id: u32, channels: u32, color_format: u32,
}
```

Implements `Send + Sync` (unsafe, documented).

| Constructor | Signature | Notes |
|---|---|---|
| `wrap` | `(buf: gst::Buffer) → Self` | Plain wrapper, params zeroed. NOGPU |
| `from_buffer` | `(buf: &gst::Buffer, slot_index: u32) → Result<Self, E>` | Extract from NvBufSurface buffer. Uses `extract_slot_view` for batched |
| `from_cuda_ptr` | `(data_ptr, pitch, w, h, gpu_id, channels, color_format, keepalive) → Result<Self, E>` | Synthetic descriptor around raw CUDA ptr |

| Accessor | Signature |
|---|---|
| `buffer` | `(&self) → &gst::Buffer` |
| `buffer_mut` | `(&mut self) → &mut gst::Buffer` |
| `data_ptr` | `(&self) → *mut c_void` |
| `pitch` / `width` / `height` / `gpu_id` / `channels` / `color_format` | `(&self) → u32` |

---

## set_num_filled (free function)

```rust
pub fn set_num_filled(buffer: &mut gst::BufferRef, count: u32) → Result<(), NvBufSurfaceError>
```

Low-level helper for manual slot filling via `slot_ptr`.

---

## TransformConfig

```rust
#[derive(Debug, Clone)]
pub struct TransformConfig {
    pub padding: Padding,                  // DEF: Symmetric
    pub dst_padding: Option<DstPadding>,   // DEF: None
    pub interpolation: Interpolation,      // DEF: Bilinear
    pub compute_mode: ComputeMode,         // DEF: Default
    pub cuda_stream: *mut c_void,          // DEF: null (legacy default stream)
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
