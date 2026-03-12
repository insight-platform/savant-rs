# NvBufSurface Architecture

## Module Tree
```
savant_deepstream/nvbufsurface/
├── Cargo.toml
├── build.rs                    # bindgen for NvBufSurface + NvBufSurfTransform
├── nvbufsurface_rs.h           # C header for NvBufSurface bindgen
├── nvbufsurftransform_rs.h     # C header for NvBufSurfTransform bindgen
├── src/
│   ├── lib.rs                  # Root: NvBufSurfaceError, NvBufSurfaceMemType,
│   │                           #   cuda_init, create/destroy_cuda_stream,
│   │                           #   bridge_savant_id_meta
│   ├── ffi.rs                  # bindgen output: NvBufSurface, NvBufSurfaceParams,
│   │                           #   NvBufSurfTransform, CUDA, gst_nvds_buffer_pool_new
│   ├── transform.rs            # Padding, Interpolation, ComputeMode, Rect,
│   │                           #   TransformConfig, TransformError,
│   │                           #   extract_nvbufsurface, buffer_gpu_id,
│   │                           #   clear_surface_black (platform-aware),
│   │                           #   do_transform, do_transform_to_slot
│   ├── surface_view.rs         # SurfaceView: zero-copy single-surface view
│   ├── surface_ops.rs          # memset_surface, upload_to_surface (platform-aware)
│   ├── buffers.rs              # re-exports single + batched
│   ├── buffers/
│   │   ├── single.rs           # DsNvSurfaceBufferGenerator + Builder (batchSize=1)
│   │   ├── batched.rs          # re-exports uniform, non_uniform, slot_view
│   │   └── batched/
│   │       ├── uniform.rs      # DsNvUniformSurfaceBufferGenerator + Builder,
│   │       │                   #   DsNvUniformSurfaceBuffer, set_num_filled
│   │       ├── non_uniform.rs  # DsNvNonUniformSurfaceBuffer
│   │       └── slot_view.rs    # extract_slot_view()
│   ├── egl_context.rs          # [skia] EglHeadlessContext, EglError
│   └── skia_renderer.rs        # [skia] SkiaRenderer, SkiaRendererError
└── tests/
    ├── common/mod.rs           # init() — GStreamer + CUDA one-time setup
    ├── batched.rs              # Uniform batched generator tests
    ├── heterogeneous.rs        # DsNvNonUniformSurfaceBuffer tests
    ├── slot_view.rs            # extract_slot_view + as_gst_buffer safety/leak tests
    ├── generator.rs            # Single-frame DsNvSurfaceBufferGenerator tests
    ├── transform.rs            # NvBufSurfTransform (scale/letterbox) tests
    └── bridge_meta.rs          # PTS-keyed SavantIdMeta bridge tests
```

## Memory Model

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

#### 2. System-Memory Self-Contained (Wrappers, Non-Uniform, Slot Views)
```
gst::Buffer
  └── GstMemory (system malloc):
        [NvBufSurface header | NvBufSurfaceParams[0] | ... | NvBufSurfaceParams[N-1]]
  └── surfaceList → points within same GstMemory (header + sizeof(NvBufSurface))
  └── GstParentBufferMeta → keeps pool/source buffers alive
  └── SavantIdMeta → frame IDs
```

This is the layout used by:
- `as_gst_buffer()` on **uniform** batches (copies header + params, parent ref to pool buffer)
- `as_gst_buffer()` on **non-uniform** batches (buffer.clone(), already self-contained)
- `extract_slot_view()` (batchSize=1, single params entry, parent ref to batch)
- `SurfaceView::from_cuda_ptr()` (synthetic descriptor around raw CUDA pointer)

#### 3. Non-Uniform Batch Internal
```
gst::Buffer (system memory, owned by DsNvNonUniformSurfaceBuffer)
  └── GstMemory: [NvBufSurface | params[0] | ... | params[N-1]]
  └── surfaceList → within GstMemory
  └── GstParentBufferMeta[0] → source buffer 0 (keeps its GPU memory alive)
  └── GstParentBufferMeta[1] → source buffer 1
  └── ...
```

### GstParentBufferMeta Chain
`GstParentBufferMeta` is a standard GStreamer meta that increments the
parent buffer's refcount. It propagates through `gst_buffer_copy`
(COW), so cloned buffers also keep parents alive.

Reference chain examples:
```
slot_view → batch wrapper → pool buffer → pool

slot_view from non-uniform → non-uniform buffer → source buffers[0..N]
```

### Buffer Pool Lifecycle
```
DsNvUniformSurfaceBufferGenerator
  ├── pool: gst::BufferPool (NvDS, active)
  ├── Drop → pool.set_active(false)
  └── acquire_batched_surface() → DsNvUniformSurfaceBuffer { buffer: gst::Buffer }

DsNvUniformSurfaceBuffer
  ├── buffer: gst::Buffer (refcount managed by GStreamer)
  ├── fill_slot() → NvBufSurfTransform writes to GPU
  ├── finalize() → sets numFilled, attaches SavantIdMeta
  ├── as_gst_buffer() → new self-contained buffer + GstParentBufferMeta
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
| `NvBufSurfaceSyncForDevice` | Sync CPU writes to device (auto-generated via bindgen) |
| `NvBufSurfaceSyncForCpu` | Sync device writes to CPU (auto-generated via bindgen) |
| `cuMemsetD8_v2` | CUDA memset (manually declared) |
| `cuMemcpyHtoD_v2` | CUDA host-to-device copy (manually declared) |
| `cudaSetDevice(device)` | Set active CUDA device |
| `cudaFree(ptr)` | Free CUDA memory / trigger lazy context init |
| `cudaMemset2DAsync(...)` | Clear GPU surface to black for letterboxing |
| `cudaStreamCreateWithFlags(...)` | Create non-blocking CUDA stream |
| `cudaStreamDestroy(stream)` | Destroy CUDA stream |
| `cudaStreamSynchronize(stream)` | Wait for stream operations to complete |

### Key FFI Functions (transform_ffi)
| Function | Purpose |
|---|---|
| `NvBufSurfTransformSetSessionParams(...)` | Set compute mode, GPU, stream |
| `NvBufSurfTransform(src, dst, params)` | Perform scaling/cropping |

### Key FFI Functions (used via extern "C" in uniform.rs, non_uniform.rs, slot_view.rs)
| Function | Purpose |
|---|---|
| `gst_buffer_add_parent_buffer_meta(buf, parent)` | Add parent buffer meta (keeps parent alive) |

## SavantIdMeta Propagation

`SavantIdMeta` is a custom GstMeta that carries frame IDs through the
pipeline. Propagation rules:

| Operation | ID Behavior |
|---|---|
| `fill_slot(src, None, Some(42))` | Explicit ID `Frame(42)` stored |
| `fill_slot(src, None, None)` | Auto-propagate first ID from source buffer |
| `add(src, Some(42))` | Explicit ID `Frame(42)` stored |
| `add(src, None)` | Auto-propagate first ID from source buffer |
| `finalize()` | All collected IDs attached as `SavantIdMeta` on batch buffer |
| `as_gst_buffer()` | IDs copied from internal buffer to wrapper |
| `extract_slot_view(batch, i)` | ID at index `i` propagated to slot view |
| `bridge_savant_id_meta(element)` | PTS-keyed bridging across encoders |

## CUDA Stream Model

By default, `TransformConfig.cuda_stream` is null, which uses CUDA's
legacy default stream (stream 0). This has implicit sync semantics that
serialize all GPU operations. For concurrent transforms, use
`create_cuda_stream()` to get a non-blocking stream and set it in the
config. Always call `destroy_cuda_stream()` when done.

⚠ After each transform, `cudaStreamSynchronize` is called to prevent
stale-data artifacts when source buffers are returned to pools.

## Platform-Aware Memory Access

Multiple modules handle the Jetson vs dGPU memory difference via
`cfg(target_arch = "aarch64")`:

- **On Jetson (aarch64):** `NvBufSurfaceMemType::Default` maps to `SurfaceArray`
  (VIC-managed), which is **NOT** CUDA-addressable. Must use
  `NvBufSurfaceMap` → CPU write → `NvBufSurfaceSyncForDevice` →
  `NvBufSurfaceUnMap`.

- **On dGPU:** `Default` maps to `CudaDevice`, and `cuMemsetD8_v2` /
  `cudaMemset2DAsync` / `cuMemcpyHtoD_v2` work directly on the GPU memory.

### Modules with platform-aware paths

| Module | Functions | Jetson path | dGPU path |
|---|---|---|---|
| `surface_ops` | `memset_surface`, `upload_to_surface` | `NvBufSurfaceMap` + CPU write | `cuMemsetD8_v2`, `cudaMemcpy2D` |
| `transform` | `clear_surface_black` (letterbox padding) | `clear_surface_black_mapped`: Map → zero all planes → Sync → Unmap | `cudaMemset2DAsync` + `cudaStreamSynchronize` |
| `skia_renderer` | `load_from_nvbuf`, `copy_gl_to_nvbuf` | Map → SyncForCpu/SyncForDevice → `cudaMemcpy2D{To,From}Array` (H2D/D2H) → UnMap | `cudaMemcpy2D{To,From}Array` (D2D) directly on `dataPtr` |

The `clear_surface_black` function in `transform.rs` is called by `do_transform()`
when the letterboxed image doesn't fill the entire destination surface. On Jetson,
the mapped path (`clear_surface_black_mapped`) iterates all planes via
`planeParams.num_planes` and uses `planeParams.pitch[plane]` /
`planeParams.height[plane]` for correct multi-plane format support (NV12, I420).

---

## Features

| Feature | Effect |
|---|---|
| `default` | empty |
| `skia` | Enables `skia-safe`, `gl`; adds EglHeadlessContext, SkiaRenderer |

## Python Bindings

PyO3 bindings live in `savant_core/savant_core_py/src/deepstream.rs`, not
in this crate. They wrap:
- `DsNvUniformSurfaceBuffer` → `PyDsNvUniformSurfaceBuffer`
- `DsNvNonUniformSurfaceBuffer` → `PyDsNvNonUniformSurfaceBuffer`
- `gst::Buffer` wrapper → `PyDsNvBufSurfaceGstBuffer`
- `SurfaceView` → `PySurfaceView`
- Free functions: `set_num_filled`, `render_to_nvbuf`, `set_buffer_pts`, etc.

Type stubs: `savant_python/python/savant_rs/deepstream/deepstream.pyi`
