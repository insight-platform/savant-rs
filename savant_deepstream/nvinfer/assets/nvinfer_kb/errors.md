# NvInferError Reference

```rust
#[derive(Debug, thiserror::Error)]
pub enum NvInferError {
    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Element creation failed: {0}")]
    ElementCreationFailed(String),

    #[error("Link failed: {0}")]
    LinkFailed(String),

    #[error("Invalid property: {0}")]
    InvalidProperty(String),

    #[error("Invalid nvinfer config: {0}")]
    InvalidConfig(String),

    #[error("Batch meta attachment failed: {0}")]
    BatchMetaFailed(String),

    #[error("Null pointer: {0}")]
    NullPointer(String),

    #[error("GStreamer initialization failed: {0}")]
    GstInit(String),
}
```

## When Each Variant is Returned

| Variant | Trigger |
|---|---|
| `PipelineError` | `submit`/`infer_sync` when `SharedBuffer::into_buffer()` fails (outstanding refs); appsrc push failure; infer_sync timeout (30s); channel disconnect |
| `ElementCreationFailed` | `gst::ElementFactory::make` fails for appsrc, appsink, nvinfer, queue |
| `LinkFailed` | `gst::Element::link_many` fails |
| `InvalidProperty` | GStreamer element property set fails |
| `InvalidConfig` | `process-mode != 2`, `output-tensor-meta != 1`, `network-type != 100`, or `gie-unique-id != 1`; forbidden key in nvinfer_properties (`operate-on-gie-id`, `operate-on-class-ids`, `secondary-reinfer-interval`, `num-detected-classes`, `disable-output-host-copy`); temp file creation/write fails |
| `BatchMetaFailed` | `attach_batch_meta_with_rois` or batch meta read fails |
| `NullPointer` | Null pointer in batch meta / surface extraction |
| `GstInit` | `gst::init()` fails |

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
let config = NvInferConfig::new(props, "RGBA", 640, 480);
let result = config.validate_and_materialize();
assert!(matches!(result, Err(NvInferError::InvalidConfig(_))));
```
