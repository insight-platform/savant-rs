# NvBufSurface Architecture

## Module Tree
```
savant_deepstream/buffers/
├── Cargo.toml
├── build.rs                    # bindgen for NvBufSurface + NvBufSurfTransform
├── buffers_rs.h           # C header for NvBufSurface bindgen
├── nvbufsurftransform_rs.h     # C header for NvBufSurfTransform bindgen
├── src/
│   ├── lib.rs                  # Root: NvBufSurfaceError, NvBufSurfaceMemType,
│   │                           #   cuda_init, bridge_savant_id_meta
│   ├── cuda_stream.rs          # CudaStream: safe RAII CUDA stream wrapper
│   ├── ffi.rs                  # bindgen output: NvBufSurface, NvBufSurfaceParams,
│   │                           #   NvBufSurfTransform, CUDA, gst_nvds_buffer_pool_new
│   ├── transform.rs            # Padding, Interpolation, ComputeMode, Rect,
│   │                           #   TransformConfig, TransformError,
│   │                           #   extract_nvbufsurface, buffer_gpu_id,
│   │                           #   clear_surface_black (platform-aware),
│   │                           #   do_transform, do_transform_to_slot
│   ├── shared_buffer.rs         # SharedBuffer: Arc<parking_lot::Mutex<gst::Buffer>>,
│   │                           #   shared currency for SurfaceView, Picasso, encoder
│   ├── surface_view.rs         # SurfaceView: zero-copy single-surface view
│   │                           #   memset, fill, upload (methods on SurfaceView)
│   ├── pipeline.rs             # BufferGeneratorExt, UniformBatchGeneratorExt traits,
│   │                           #   re-exports extract_nvbufsurface, buffer_gpu_id
│   ├── buffers.rs              # re-exports single + batched
│   ├── buffers/
│   │   ├── single.rs           # BufferGenerator + Builder (batchSize=1)
│   │   ├── batched.rs          # re-exports uniform, non_uniform
│   │   ├── batch_state.rs      # generic BatchState<T> used by higher-level operators
│   │   └── batched/
│   │       ├── uniform.rs      # UniformBatchGenerator + Builder,
│   │       │                   #   SurfaceBatch
│   │       └── non_uniform.rs  # NonUniformBatch
│   ├── egl_cuda_meta.rs        # [aarch64] EglCudaMeta GstMeta (multi-slot),
│   │                           #   EglCudaMapping, ensure_meta/read_meta per slot,
│   │                           #   EGL-CUDA zero-copy interop, meta_free iterates slots
│   ├── egl_context.rs          # [skia] EglHeadlessContext, EglError
│   └── skia_renderer.rs        # [skia] SkiaRenderer, SkiaRendererError
├── tests/
│   ├── common/mod.rs           # init() — GStreamer + CUDA one-time setup
│   ├── batched.rs              # Uniform batched generator tests
│   ├── heterogeneous.rs        # NonUniformBatch tests
│   ├── slot_view.rs            # SurfaceView + SharedBuffer integration tests
│   ├── bridge_meta.rs          # PTS-keyed SavantIdMeta bridge tests
│   └── surface_view_gpu.rs     # SurfaceView GPU tests: CUDA addressability,
│                               #   write/read roundtrip, recycled buffer mapping,
│                               #   uniform batch slot tests, EglCudaMeta tracking,
│                               #   map_unmap_cycle verification
└── benches/
    ├── surface_view_mapping.rs # Criterion benchmarks: bench_registration_plus_first_view
    │                           #   (fresh buffer), bench_recycled_buffer_view (recycled pool;
    │                           #   POOLED meta survives recycle, no re-registration)
    ├── non_uniform_batch.rs    # Criterion benchmarks for non-uniform batch operations
    └── transform_into.rs       # Criterion benchmarks for transform operations
```

## Memory Model

## Shared batching state

`deepstream_buffers::BatchState<T>` is the single shared implementation for
size/time batching queues. `nvinfer::NvInferBatchingOperator` and
`nvtracker::NvTrackerBatchingOperator` both use this type instead of local
duplicate `state.rs` structs.

### NvBufSurface C Struct Layout
```
NvBufSurface {
    gpuId:        u32,
    batchSize:    u32,       // max capacity
    numFilled:    u32,       // actual count
    isContiguous: bool,
    memType:      u32,       // NvBufSurfaceMemType
    surfaceList:  *mut NvBufSurfaceParams,  // ⚠ raw pointer
    ...
}

NvBufSurfaceParams {
    width:       u32,
    height:      u32,
    pitch:       u32,        // row stride in bytes
    colorFormat: u32,
    dataPtr:     *mut c_void, // GPU memory pointer
    dataSize:    u32,
    ...
}
```

⚠ `surfaceList` is an **absolute pointer**. For pool-allocated buffers it
points to memory managed by the NvDs allocator. For self-contained buffers
(our wrappers), it points within the buffer's own GstMemory.

### Three Memory Patterns

#### 1. Pool-Allocated (Uniform Generator)
```
GstBufferPool (NvDS) → gst::Buffer
  └── GstMemory: NvBufSurface header
  └── surfaceList → external NvDs allocator memory
  └── surfaceList[i].dataPtr → GPU memory (CUDA device)
```
The pool owns the external allocator memory. When the buffer's refcount
drops to 0, it returns to the pool and the allocator memory is reused.

#### 2. System-Memory Self-Contained (Non-Uniform, Synthetic Views)
```
gst::Buffer
  └── GstMemory (system malloc):
        [NvBufSurface header | NvBufSurfaceParams[0] | ... | NvBufSurfaceParams[N-1]]
  └── surfaceList → points within same GstMemory (header + sizeof(NvBufSurface))
  └── GstParentBufferMeta → keeps pool/source buffers alive
  └── SavantIdMeta → frame IDs
```

This is the layout used by:
- **Non-uniform** batches: `finalize(ids)` allocates a self-contained buffer and returns
  `SharedBuffer`; no `as_gst_buffer()` (removed from `NonUniformBatch`).
- **Uniform** batches: `as_gst_buffer()` has been removed (use `shared_buffer()` +
  `SurfaceView::from_buffer` for per-slot access; `shared.into_buffer()` to extract
  `gst::Buffer` for NvInfer/encoder).
- `extract_slot_view()` has been removed; use `SurfaceView::from_buffer(&shared, i)` instead.
- `SurfaceView::from_cuda_ptr()` (synthetic descriptor around raw CUDA pointer)

#### 3. Non-Uniform Batch Internal
```
gst::Buffer (system memory, owned by SharedBuffer from finalize)
  └── GstMemory: [NvBufSurface | params[0] | ... | params[N-1]]
  └── surfaceList → within GstMemory
  └── GstParentBufferMeta[0] → source buffer 0 (keeps its GPU memory alive)
  └── GstParentBufferMeta[1] → source buffer 1
  └── ...
```

Non-uniform batches no longer use `as_gst_buffer()`; `finalize(ids)` returns
`SharedBuffer` directly.

### GstParentBufferMeta Chain
`GstParentBufferMeta` is a standard GStreamer meta that increments the
parent buffer's refcount. It propagates through `gst_buffer_copy`
(COW), so cloned buffers also keep parents alive.

Reference chain examples:
```
SurfaceView::from_buffer(&shared, i) → shared (SharedBuffer) → pool buffer → pool

SurfaceView::from_buffer(&shared, i) → shared (from non-uniform finalize) → source buffers[0..N]
```

### Buffer Pool Lifecycle
```
UniformBatchGenerator
  ├── pool: gst::BufferPool (NvDS, active)
  ├── Drop → pool.set_active(false)
  ├── acquire(id) → SharedBuffer (direct pool acquisition)
  └── acquire_batch(config, ids) → SurfaceBatch { buffer: SharedBuffer, ids }

SurfaceBatch
  ├── buffer: SharedBuffer (Arc<parking_lot::Mutex<gst::Buffer>>)
  ├── transform_slot() → NvBufSurfTransform writes to GPU
  ├── finalize(num_filled, ids) → sets numFilled, attaches SavantIdMeta
  ├── shared_buffer() → SharedBuffer for per-slot access
  ├── SurfaceView::from_buffer(&shared, i) → per-slot GPU view
  ├── shared.into_buffer() → extract gst::Buffer for NvInfer/encoder (when sole ref)
  └── Drop → buffer refcount decremented
       └── if refcount=0 → buffer returns to pool
```

## FFI Layer

### Key FFI Functions (ffi.rs)
| Function | Purpose |
|---|---|
| `gst_nvds_buffer_pool_new()` | Create DeepStream buffer pool |
| `NvBufSurfaceMap` | Map surface for CPU access (auto-generated via bindgen) |
| `NvBufSurfaceUnMap` | Unmap surface after CPU access (auto-generated via bindgen) |
| `NvBufSurfaceMapEglImage` | Map surface to EGLImage for EGL-CUDA interop (bindgen) |
| `NvBufSurfaceUnMapEglImage` | Unmap EGLImage (bindgen) |
| `NvBufSurfaceSyncForDevice` | Sync CPU writes to device (auto-generated via bindgen) |
| `NvBufSurfaceSyncForCpu` | Sync device writes to CPU (auto-generated via bindgen) |
| `cuMemsetD8_v2` | CUDA memset (manually declared) |
| `cuMemcpyHtoD_v2` | CUDA host-to-device copy (manually declared) |
| `cudaSetDevice(device)` | Set active CUDA device |
| `cudaFree(ptr)` | Free CUDA memory / trigger lazy context init |
| `cudaMemset2DAsync(...)` | Clear GPU surface to black for letterboxing |
| `cudaMemcpy2D(...)` | 2D memory copy with pitch (manually declared) |
| `cudaStreamCreateWithFlags(...)` | Create non-blocking CUDA stream |
| `cudaStreamDestroy(stream)` | Destroy CUDA stream |
| `cudaStreamSynchronize(stream)` | Wait for stream operations to complete |
| `cuInit(flags)` | Initialize CUDA driver API (manually declared in lib.rs) |
| `cuDevicePrimaryCtxRetain(...)` | Retain primary CUDA context for device |
| `cuCtxSetCurrent(ctx)` | Set CUDA context on current thread |

### Key FFI Functions (EGL-CUDA interop, ffi.rs — aarch64 only)
| Function | Purpose |
|---|---|
| `cuGraphicsEGLRegisterImage(...)` | Register EGLImage with CUDA (creates permanent implicit mapping on Jetson) |
| `cuGraphicsResourceGetMappedEglFrame(...)` | Get CUDA pointer from registered EGLImage |
| `cuGraphicsUnregisterResource(...)` | Unregister CUDA graphics resource |
| `NvBufSurfaceUnMapEglImage(surf_ptr, 0)` | Called from `meta_free` alongside unregister |

**Note:** `cuGraphicsMapResources` / `cuGraphicsUnmapResources` are **not used** on Jetson — the mapping is implicit and permanent from registration until unregister.

### Key FFI Types (EGL-CUDA interop, ffi.rs)
| Type | Purpose |
|---|---|
| `CUgraphicsResource` | Opaque handle for CUDA graphics resource |
| `CUeglFrame` | EGL frame descriptor with CUDA pointers and pitches |
| `CUeglFrameData` | Union: `pArray[3]` or `pPitch[3]` CUDA pointers |

### Key FFI Functions (transform_ffi)
| Function | Purpose |
|---|---|
| `NvBufSurfTransformSetSessionParams(...)` | Set compute mode, GPU, stream |
| `NvBufSurfTransform(src, dst, params)` | Perform scaling/cropping |

### Key FFI Functions (used via extern "C" in non_uniform.rs)
| Function | Purpose |
|---|---|
| `gst_buffer_add_parent_buffer_meta(buf, parent)` | Add parent buffer meta (keeps parent alive) |

## SavantIdMeta Propagation

`SavantIdMeta` is a custom GstMeta that carries frame IDs through the
pipeline. Propagation rules:

| Operation | ID Behavior |
|---|---|
| `transform_slot(src, None, Some(42))` | Explicit ID `Frame(42)` stored |
| `transform_slot(src, None, None)` | Auto-propagate first ID from source buffer |
| `add(src, Some(42))` | Explicit ID `Frame(42)` stored |
| `add(src, None)` | Auto-propagate first ID from source buffer |
| `finalize(num_filled, ids)` (uniform) | Explicit `ids` attached as `SavantIdMeta` on batch buffer |
| `finalize(ids)` (non-uniform) | Explicit `ids` attached; returns `SharedBuffer` |
| `SurfaceView::from_buffer(&shared, i)` | Does not propagate IDs — they live on the batch buffer accessed via `shared_buffer()` |
| `bridge_savant_id_meta(element)` | PTS-keyed bridging across encoders |

## CUDA Stream Model

By default, `TransformConfig.cuda_stream` is null, which uses CUDA's
legacy default stream (stream 0). This has implicit sync semantics that
serialize all GPU operations. For concurrent transforms, use
`CudaStream::new_non_blocking()` to get a non-blocking stream and set it
in the config.  The stream is destroyed automatically on drop.

⚠ After each transform, `cudaStreamSynchronize` is called to prevent
stale-data artifacts when source buffers are returned to pools.

## Platform-Aware Memory Access

### SharedBuffer — Shared Currency

`SharedBuffer` is a newtype around `Arc<parking_lot::Mutex<gst::Buffer>>`
(not `std::sync::Mutex`) that serves as the shared currency for passing
NvBufSurface-backed buffers between `SurfaceView`, Picasso, and the encoder without
ownership transfer. Multiple `SurfaceView`s (for different batch slots) can reference
the same underlying buffer via `from_buffer`. `Clone` is cheap (Arc increment).
`into_buffer()` extracts the inner buffer only when this is the sole strong reference.

### Unified via SurfaceView + EglCudaMeta

Most GPU memory access is unified through `SurfaceView::from_buffer` or
`SurfaceView::from_buffer`, which wrap the buffer in `SharedBuffer` and
provide a CUDA device pointer on both platforms:

- **On dGPU:** `data_ptr` is read directly from `NvBufSurfaceParams::dataPtr`.
- **On Jetson (aarch64):** `from_buffer` / `from_buffer` call `EglCudaMeta::ensure_meta`
  with the slot index, which performs per-slot EGL-CUDA interop to obtain a
  zero-copy CUDA device pointer. The mapping is attached as `GstMeta` on the buffer
  and automatically deregistered when the buffer is freed.

`SurfaceView` holds `SharedBuffer` + `slot_index` internally. It no longer
uses `extract_slot_view` — batched buffers are accessed directly via `from_buffer`
with different slot indices.

### EGL-CUDA Interop Details (Jetson, aarch64 only)

The `egl_cuda_meta` module implements **multi-slot** EGL-CUDA interop:

- `EglCudaMetaInner` has `slots: [SlotRegistration; MAX_BATCH_SLOTS]` (64 slots)
  and `batch_size: u32`. Each slot is registered lazily on first access. The slot
  count cap is a Savant design choice (fixed-size `GstMeta`, no heap slot table);
  it is not an NvBufSurface or hardware maximum (`batchSize` is an unconstrained `u32`
  in NVIDIA headers).
- `ensure_meta(buf, slot_index)` — per-slot lazy registration; returns cached
  pointers if the slot is already registered.
- `read_meta(buf, slot_index)` — per-slot read; returns `None` if slot not registered.
- `meta_free` — iterates all slots, deregisters each with non-null `resource`
  via `cuGraphicsUnregisterResource` and `NvBufSurfaceUnMapEglImage`.
- Tracking counters (`tracking_counts`) are per individual slot registration/deregistration.

Interop chain per slot:
1. `NvBufSurfaceMapEglImage(surf_ptr, slot_index)` — maps VIC surface to EGLImage
2. `cuGraphicsEGLRegisterImage` — registers EGLImage with CUDA; on Jetson this
   creates a **permanent implicit mapping** (`cuGraphicsUnmapResources` returns
   error 999). No RAII map/unmap cycle — pointer valid from registration until
   unregister.
3. `cuGraphicsResourceGetMappedEglFrame` — gets CUDA pointer directly (no
   `cuGraphicsMapResources` call)
4. On buffer free: `meta_free` iterates slots and calls `cuGraphicsUnregisterResource`
   and `NvBufSurfaceUnMapEglImage(surf_ptr, i)` for each registered slot.

Thread safety: `ensure_cuda_egl_context` initializes CUDA driver context
(`cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`) and EGL display per thread.

**Key behaviour:** `EglCudaMeta` is created with `GST_META_FLAG_POOLED | GST_META_FLAG_LOCKED`,
so the meta **survives GstBufferPool recycles**. Pool of 1 buffer, N acquisitions =
1 registration per slot (first time), 0 re-registrations. Deregistration only on pool destroy.

### Modules with platform-aware paths

| Module | Functions | Jetson path | dGPU path |
|---|---|---|---|
| `surface_view` | `from_buffer`, `from_gst_buffer` | `EglCudaMeta::ensure_meta(buf, slot_index)` per-slot (zero-copy EGL-CUDA interop, permanent mapping, POOLED meta) | Direct `NvBufSurfaceParams::dataPtr` |
| `surface_view` | `memset`, `fill`, `upload` (methods) | `view.memset()`, `view.fill()`, `view.upload()` — CUDA driver API | Same: `cuMemsetD8_v2`, `cuMemsetD32_v2`, `cudaMemcpy2D` |
| `skia_renderer` | `load_from_nvbuf`, `from_nvbuf`, `render_to_nvbuf`, `render_to_nvbuf_with_ptr`, `render_to_nvbuf_raw` | `render_to_nvbuf` creates a `SurfaceView` internally to resolve the CUDA pointer and **keeps the view alive** until after the pointer is used (prevents use-after-free on Jetson where COW buffer copy would invalidate the EGL-CUDA pointer if dropped early). `render_to_nvbuf_with_ptr`/`render_to_nvbuf_raw`: caller supplies `(data_ptr, pitch)`. Scaled path creates `SurfaceView` internally for temp buffer. | Same: caller supplies pointer for `*_with_ptr`/`*_raw`; `render_to_nvbuf` creates `SurfaceView` internally |
| `transform` | `clear_surface_black` (letterbox padding) | `clear_surface_black_mapped`: Map → zero all planes → Sync → Unmap | `cudaMemset2DAsync` + `cudaStreamSynchronize` |

**Note:** `clear_surface_black` in `transform.rs` still uses CPU-staging on Jetson
because it operates on raw `*mut NvBufSurface` pointers (not `gst::Buffer`). The
overhead is acceptable as it only runs during letterbox padding operations.

---

## Features

| Feature | Effect |
|---|---|
| `default` | empty |
| `skia` | Enables `skia-safe`, `gl`; adds EglHeadlessContext, SkiaRenderer |
| `testing` | Enables test-only helpers (e.g. `SurfaceView::wrap`, `EglCudaMeta::reset_tracking`) |

## Python Bindings

PyO3 bindings live in `savant_core/savant_core_py/src/deepstream.rs`, not
in this crate. They wrap:
- `SurfaceBatch` → `PySurfaceBatch`
- `NonUniformBatch` → `PyNonUniformBatch`
- `SharedBuffer` → `PySharedBuffer` (uses `Option<SharedBuffer>` internally for Python move semantics; no constructors/clone/deconstruct exposed)
- `SurfaceView` → `PySurfaceView`
- Free functions: `set_num_filled`, `render_to_nvbuf`, `set_buffer_pts`, etc.

Type stubs: `savant_python/python/savant_rs/deepstream/deepstream.pyi`
