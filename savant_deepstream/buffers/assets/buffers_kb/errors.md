# Error Reference

## NvBufSurfaceError (lib.rs)

```rust
#[derive(Debug, thiserror::Error)]
pub enum NvBufSurfaceError {
    #[error("Failed to create NvDS buffer pool")]
    PoolCreationFailed,

    #[error("Failed to get buffer pool configuration")]
    PoolConfigFailed,

    #[error("Failed to set buffer pool configuration: {0}")]
    PoolSetConfigFailed(String),

    #[error("Failed to activate buffer pool: {0}")]
    PoolActivationFailed(String),

    #[error("Failed to acquire buffer from pool: {0}")]
    BufferAcquisitionFailed(String),

    #[error("Failed to copy buffer contents: {0}")]
    BufferCopyFailed(String),

    #[error("Null pointer: {0}")]
    NullPointer(String),

    #[error("CUDA initialization failed with error code {0}")]
    CudaInitFailed(i32),

    #[error("Batch overflow: tried to fill more than {max} slots")]
    BatchOverflow { max: u32 },

    #[error("Slot index {index} out of bounds (max batch size {max})")]
    SlotOutOfBounds { index: u32, max: u32 },

    #[error("Operation requires finalize() to be called first")]
    NotFinalized,

    #[error("Batch has already been finalized; mutation is not allowed")]
    AlreadyFinalized,

    #[error("NvBufSurfaceMap failed (code {0})")]
    SurfaceMapFailed(i32),

    #[error("NvBufSurfaceUnMap failed (code {0})")]
    SurfaceUnmapFailed(i32),

    #[error("NvBufSurfaceSyncForDevice failed (code {0})")]
    SurfaceSyncFailed(i32),

    #[error("CUDA driver API {function} failed (code {code})")]
    CudaDriverError { function: &'static str, code: u32 },

    #[error("{0}")]
    InvalidInput(String),
}
```

### When Each Variant Triggers

| Variant | Trigger |
|---|---|
| `PoolCreationFailed` | `gst_nvds_buffer_pool_new()` returns null |
| `PoolConfigFailed` | `gst_buffer_pool_get_config()` returns null |
| `PoolSetConfigFailed` | `gst_buffer_pool_set_config()` returns FALSE |
| `PoolActivationFailed` | `pool.set_active(true)` fails |
| `BufferAcquisitionFailed` | `pool.acquire()` or `Buffer::with_size()` or `map_writable()` fails |
| `BufferCopyFailed` | `extract_nvbufsurface()` or transform fails |
| `NullPointer` | `SurfaceView::from_cuda_ptr(null, ...)` |
| `CudaInitFailed` | `cudaSetDevice` or `cudaFree(NULL)` or stream ops fail |
| `BatchOverflow` | `transform_slot`/`add` when `num_filled >= max_batch_size` |
| `SlotOutOfBounds` | `slot_ptr(index >= max)` or `SurfaceView::from_buffer` with invalid slot index |
| `NotFinalized` | Operations that require `finalize()` to have been called first |
| `AlreadyFinalized` | `transform_slot()`/`add()` after `finalize()`, or double `finalize()` |
| `SurfaceMapFailed` | `NvBufSurfaceMap()` returns non-zero (Jetson CPU-staging path in `surface_ops`, `clear_surface_black`) |
| `SurfaceUnmapFailed` | `NvBufSurfaceUnMap()` returns non-zero (Jetson CPU-staging path) |
| `SurfaceSyncFailed` | `NvBufSurfaceSyncForDevice()` returns non-zero (Jetson CPU-staging path) |
| `CudaDriverError` | `cuMemsetD8_v2` or `cuMemcpyHtoD_v2` fails in `surface_ops` (dGPU path) |
| `InvalidInput` | `SurfaceView::upload` dimension/data mismatch, `EglCudaMeta` registration failure (e.g., `NvBufSurfaceMapEglImage` or `cuGraphicsEGLRegisterImage` error), `extract_nvbufsurface` failure |

---

## TransformError (transform.rs)

```rust
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("NvBufSurfTransform failed with error code {0}")]
    TransformFailed(i32),

    #[error("NvBufSurfTransformSetSessionParams failed with error code {0}")]
    SetSessionFailed(i32),

    #[error("Invalid buffer: {0}")]
    InvalidBuffer(&'static str),

    #[error("Invalid dst_padding: {0}")]
    InvalidDstPadding(&'static str),

    #[error("CUDA error: {0}")]
    CudaError(i32),
}
```

### When Each Variant Triggers

| Variant | Trigger |
|---|---|
| `TransformFailed` | `NvBufSurfTransform()` FFI call returns non-zero |
| `SetSessionFailed` | `NvBufSurfTransformSetSessionParams()` returns non-zero |
| `InvalidBuffer` | Buffer too small for NvBufSurface, null pointer, `dst_slot >= batchSize`, mapped address null after `NvBufSurfaceMap` |
| `InvalidDstPadding` | `dst_padding` leaves effective width or height below 16 px |
| `CudaError` | `cudaMemset2DAsync` or `cudaStreamSynchronize` fails (dGPU); `NvBufSurfaceMap`, `NvBufSurfaceSyncForDevice`, or `NvBufSurfaceUnMap` fails (Jetson) |

⚠ On Jetson (aarch64), `CudaError(1)` from the clearing step means
`cudaMemset2DAsync` was called on NVMM memory (not a CUDA device pointer).
The `clear_surface_black` function in `transform.rs` handles this via
`cfg(target_arch)` — Jetson uses the map/CPU-memset/sync/unmap path instead.

---

## EglError (egl_context.rs, skia feature)

```rust
#[derive(Debug, thiserror::Error)]
pub enum EglError {
    #[error("EGL extension not available: {0}")]
    MissingExtension(String),
    #[error("No EGL devices found")]
    NoDevices,
    #[error("EGL error 0x{0:x}: {1}")]
    Egl(u32, String),
}
```

---

## SkiaRendererError (skia_renderer.rs, skia feature)

```rust
pub enum SkiaRendererError {
    Egl(#[from] EglError),
    Gl(String),
    Cuda(i32, String),
    Skia(String),
    NvBuf(String),
}
```

---

## Testing Error Paths

### AlreadyFinalized
```rust
let mut batch = gen.acquire_batch(TransformConfig::default(), vec![SavantIdMetaKind::Frame(1)]).unwrap();
let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
batch.transform_slot(0, &src_view, None).unwrap();
batch.finalize().unwrap();
// After finalize:
assert!(matches!(batch.finalize(), Err(NvBufSurfaceError::AlreadyFinalized)));
assert!(matches!(batch.transform_slot(1, &src_view, None), Err(NvBufSurfaceError::AlreadyFinalized)));
```

### BatchOverflow
```rust
let gen = UniformBatchGenerator::new(RGBA, 640, 640, 2, 2, 0, Default).unwrap();
let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
let mut batch = gen.acquire_batch(TransformConfig::default(), ids).unwrap();
let v1 = SurfaceView::from_buffer(&src1, 0).unwrap();
let v2 = SurfaceView::from_buffer(&src2, 0).unwrap();
let v3 = SurfaceView::from_buffer(&src3, 0).unwrap();
batch.transform_slot(0, &v1, None).unwrap();
batch.transform_slot(1, &v2, None).unwrap();
assert!(matches!(batch.transform_slot(2, &v3, None), Err(NvBufSurfaceError::SlotOutOfBounds { .. })));
```

### SlotOutOfBounds
```rust
let result = SurfaceView::from_buffer(&shared, 999);
assert!(matches!(result, Err(NvBufSurfaceError::SlotOutOfBounds { .. })));
```
