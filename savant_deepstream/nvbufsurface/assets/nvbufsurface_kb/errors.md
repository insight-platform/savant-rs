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
}
```

### When Each Variant Triggers

| Variant | Trigger |
|---|---|
| `PoolCreationFailed` | `gst_nvds_buffer_pool_new()` returns null |
| `PoolConfigFailed` | `gst_buffer_pool_get_config()` returns null |
| `PoolSetConfigFailed` | `gst_buffer_pool_set_config()` returns FALSE |
| `PoolActivationFailed` | `pool.set_active(true)` fails |
| `BufferAcquisitionFailed` | `pool.acquire_buffer()` or `Buffer::with_size()` or `map_writable()` fails |
| `BufferCopyFailed` | `extract_nvbufsurface()` or transform fails |
| `NullPointer` | `SurfaceView::from_cuda_ptr(null, ...)` |
| `CudaInitFailed` | `cudaSetDevice` or `cudaFree(NULL)` or stream ops fail |
| `BatchOverflow` | `fill_slot`/`add` when `num_filled >= max_batch_size` |
| `SlotOutOfBounds` | `slot_ptr(index >= max)` or `extract_slot_view(index >= numFilled)` |
| `NotFinalized` | `as_gst_buffer()`, `extract_slot_view()`, `slot_ptr()` (non-uniform) before `finalize()` |
| `AlreadyFinalized` | `fill_slot()`/`add()` after `finalize()`, or double `finalize()` |

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

    #[error("CUDA error: {0}")]
    CudaError(i32),
}
```

### When Each Variant Triggers

| Variant | Trigger |
|---|---|
| `TransformFailed` | `NvBufSurfTransform()` FFI call returns non-zero |
| `SetSessionFailed` | `NvBufSurfTransformSetSessionParams()` returns non-zero |
| `InvalidBuffer` | Buffer too small for NvBufSurface, null pointer, `dst_slot >= batchSize` |
| `CudaError` | `cudaMemset2DAsync` or `cudaStreamSynchronize` fails |

---

## EglError (egl_context.rs, skia feature)

```rust
pub enum EglError {
    MissingExtension(String),
    NoDevices,
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

### NotFinalized / AlreadyFinalized
```rust
let mut batch = gen.acquire_batched_surface(TransformConfig::default()).unwrap();
// Before finalize:
assert!(matches!(batch.as_gst_buffer(), Err(NvBufSurfaceError::NotFinalized)));
assert!(matches!(batch.extract_slot_view(0), Err(NvBufSurfaceError::NotFinalized)));

batch.finalize().unwrap();
// After finalize:
assert!(matches!(batch.finalize(), Err(NvBufSurfaceError::AlreadyFinalized)));
assert!(matches!(batch.fill_slot(&src, None, None), Err(NvBufSurfaceError::AlreadyFinalized)));
```

### BatchOverflow
```rust
let gen = DsNvUniformSurfaceBufferGenerator::new(RGBA, 640, 640, 2, 2, 0, Default).unwrap();
let mut batch = gen.acquire_batched_surface(TransformConfig::default()).unwrap();
batch.fill_slot(&src1, None, Some(1)).unwrap();
batch.fill_slot(&src2, None, Some(2)).unwrap();
assert!(matches!(batch.fill_slot(&src3, None, Some(3)), Err(NvBufSurfaceError::BatchOverflow { max: 2 })));
```

### SlotOutOfBounds
```rust
let view = extract_slot_view(&batch_buf, 999);
assert!(matches!(view, Err(NvBufSurfaceError::SlotOutOfBounds { .. })));
```
