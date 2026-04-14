# NvInferError Reference

```rust
#[derive(Debug, thiserror::Error)]
pub enum NvInferError {
    PipelineError(String),
    ElementCreationFailed(String),
    LinkFailed(String),
    InvalidProperty(String),
    InvalidConfig(String),
    BatchMetaFailed(String),
    NullPointer(String),
    GstInit(String),
    BatchFormationFailed(String),
    OperatorShutdown,
    TensorTypeMismatch { expected: &'static str, actual: &'static str },
    HostDataUnavailable,
    Buffer(#[from] deepstream_buffers::NvBufSurfaceError),
    PipelineFailed,
    OperatorFailed,
    FrameworkError(#[from] savant_gstreamer::pipeline::PipelineError),
    ChannelDisconnected,
    ShuttingDown,
}
```

## When Each Variant is Returned

| Variant | Trigger |
|---|---|
| `PipelineError` | Submission/runtime pipeline errors, including shared-buffer ownership failures |
| `PipelineFailed` | Pipeline entered terminal failed state (operation timeout exceeded) |
| `OperatorFailed` | Batching operator entered terminal failed state: pending batch exceeded `pending_batch_timeout`. Operator must be recreated — all subsequent calls return this error. |
| `ElementCreationFailed` | `gst::ElementFactory::make` fails for appsrc, appsink, nvinfer, queue |
| `LinkFailed` | `gst::Element::link_many` fails |
| `InvalidProperty` | GStreamer element property set fails |
| `InvalidConfig` | `process-mode != 2`, `output-tensor-meta != 1`, `network-type != 100`, or `gie-unique-id != 1`; forbidden key in nvinfer_properties (`operate-on-gie-id`, `operate-on-class-ids`, `secondary-reinfer-interval`, `num-detected-classes`, `disable-output-host-copy`); temp file creation/write fails |
| `BatchMetaFailed` | `attach_batch_meta_with_rois` or batch meta read fails |
| `NullPointer` | Null pointer in batch meta / surface extraction |
| `GstInit` | `gst::init()` fails |
| `BatchFormationFailed` | `SurfaceView::from_buffer`, `NonUniformBatch::add`, or `NonUniformBatch::finalize` fails during batch formation |
| `OperatorShutdown` | `add_frame`, `flush`, or `submit_batch` called after operator shutdown |
| `TensorTypeMismatch` | Typed tensor accessor called with wrong expected type |
| `HostDataUnavailable` | Host tensor data is unavailable (disabled host copy / null host pointer) |
| `Buffer` | Wrapped `deepstream_buffers::NvBufSurfaceError` |
| `FrameworkError` | Error bubbled from `savant_gstreamer::pipeline` |
| `ChannelDisconnected` | Channel disconnected between producer/consumer sides |
| `ShuttingDown` | Submission attempted while graceful shutdown is active |

## Testing Error Paths

### SharedBuffer outstanding refs
```rust
let shared = batch.shared_buffer();
let result = nvinfer.submit(shared, None);
assert!(result.is_err()); // PipelineError
```

### Invalid process-mode
```rust
let mut props = HashMap::new();
props.insert("process-mode".into(), "1".into());
let config = NvInferConfig::new(
    props,
    VideoFormat::RGBA,
    640,
    480,
    ModelColorFormat::RGB,
);
let result = config.validate_and_materialize();
assert!(matches!(result, Err(NvInferError::InvalidConfig(_))));
```
