# PicassoError Reference

```rust
#[derive(Debug, thiserror::Error)]
pub enum PicassoError {
    #[error("Source '{0}' not found")]
    SourceNotFound(String),

    #[error("Worker channel for source '{0}' is disconnected")]
    ChannelDisconnected(String),

    #[error("Encoder error for source '{0}': {1}")]
    Encoder(String, String),

    #[error("Transform error for source '{0}': {1}")]
    Transform(String, String),

    #[error("Skia renderer error for source '{0}': {1}")]
    Renderer(String, String),

    #[error("Invalid transformation chain: {0}")]
    InvalidTransformationChain(String),

    #[error("GPU mismatch for source '{source_id}': buffer on GPU {buffer_gpu}, encoder on GPU {encoder_gpu}")]
    GpuMismatch { source_id: String, buffer_gpu: u32, encoder_gpu: u32 },

    #[error("TransformConfig.cuda_stream must be null; Picasso manages its own CUDA streams")]
    ExternalCudaStream,

    #[error("Failed to create worker CUDA stream: {0}")]
    CudaStreamCreationFailed(String),

    #[error("Source worker send failed: {0}")]
    SourceWorkerSendFailed(String),

    #[error("Invalid letterbox parameters: {0}")]
    InvalidLetterboxParams(String),

    #[error("Buffer error: {0}")]
    Buffer(#[from] NvBufSurfaceError),

    #[error("Engine is shut down")]
    Shutdown,
}
```

## When Each Variant is Returned

| Variant | Trigger |
|---|---|
| `Shutdown` | `send_frame`, `send_eos`, `set_source_spec` after `engine.shutdown()` |
| `ChannelDisconnected` | Worker thread died or channel dropped; `send_frame`, `send_eos`, `set_source_spec` (UpdateSpec path) |
| `Encoder` | `NvEncoder::submit_frame` or `NvEncoder::finish` failure (includes `PtsReordered` on non-monotonic PTS); also returned when `SharedBuffer::into_buffer()` fails (strong_count > 1) |
| `Transform` | `SurfaceView::transform_into` failure |
| `Renderer` | `SkiaRenderer::from_nvbuf` / `load_from_nvbuf` / `render_to_nvbuf` failure |
| `InvalidTransformationChain` | Frame doesn't have exactly `[InitialSize(w,h)]` matching its dimensions; also ScaleSpec errors |
| `GpuMismatch` | Incoming buffer's `NvBufSurface.gpuId` differs from `EncoderConfig.gpu_id`; checked at top of `process_encode`. Fail-open: skipped if GPU ID cannot be extracted (e.g., non-NVMM buffers). |
| `ExternalCudaStream` | Returned by `set_source_spec` when `TransformConfig.cuda_stream` is not default (Picasso manages its own CUDA streams) |
| `CudaStreamCreationFailed` | Reserved for future use |
| `SourceWorkerSendFailed` | Source worker channel closed or disconnected. Triggered by `SourceWorker::send_frame()`, `send_eos()`, `send_update_spec()` when crossbeam channel send fails. Low-level worker API only; engine layer returns `ChannelDisconnected` instead. |
| `InvalidLetterboxParams` | Invalid letterbox parameters. Triggered by `compute_letterbox_params()` when `dst_padding` reduces effective dimensions below 16 px. |
| `Buffer` | Auto-converted from `deepstream_buffers::NvBufSurfaceError` via `#[from]`. Any `SurfaceView` or buffer operation failures become this variant. |
| `SourceNotFound` | Currently unused in engine (kept for future use) |

## Testing Error Paths

### Post-shutdown
```rust
let mut engine = PicassoEngine::new(GeneralSpec::default(), Callbacks::default());
engine.shutdown();
assert!(matches!(engine.send_frame("x", frame, view, None), Err(PicassoError::Shutdown)));
assert!(matches!(engine.send_eos("x"), Err(PicassoError::Shutdown)));
assert!(matches!(engine.set_source_spec("x", spec), Err(PicassoError::Shutdown)));
```

### Invalid transformation chain
```rust
let frame = make_frame_sized("t", 800, 600);
let mut fm = frame.clone();
fm.add_transformation(VideoFrameTransformation::Padding(10, 10, 10, 10)); // breaks invariant
let result = rewrite_frame_transformations(&frame, 640, 480, &TransformConfig::default(), None);
assert!(result.is_err());
```
