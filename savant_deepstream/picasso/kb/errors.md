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

    #[error("Engine is shut down")]
    Shutdown,
}
```

## When Each Variant is Returned

| Variant | Trigger |
|---|---|
| `Shutdown` | `send_frame`, `send_eos`, `set_source_spec` after `engine.shutdown()` |
| `ChannelDisconnected` | Worker thread died or channel dropped; `send_frame`, `send_eos`, `set_source_spec` (UpdateSpec path) |
| `Encoder` | `NvEncoder::submit_frame` or `NvEncoder::finish` failure |
| `Transform` | `NvBufSurfaceGenerator::transform` failure |
| `Renderer` | `SkiaRenderer::from_nvbuf` / `load_from_nvbuf` / `render_to_nvbuf` failure |
| `InvalidTransformationChain` | Frame doesn't have exactly `[InitialSize(w,h)]` matching its dimensions; also ScaleSpec errors |
| `SourceNotFound` | Currently unused in engine (kept for future use) |

## Testing Error Paths

### Post-shutdown
```rust
let mut engine = PicassoEngine::new(GeneralSpec::default(), Callbacks::default());
engine.shutdown();
assert!(matches!(engine.send_frame("x", frame, buf), Err(PicassoError::Shutdown)));
assert!(matches!(engine.send_eos("x"), Err(PicassoError::Shutdown)));
assert!(matches!(engine.set_source_spec("x", spec), Err(PicassoError::Shutdown)));
```

### Invalid transformation chain
```rust
let frame = make_frame(800, 600);
let mut fm = frame.clone();
fm.add_transformation(VideoFrameTransformation::Padding(10, 10, 10, 10)); // breaks invariant
let result = rewrite_frame_transformations(&frame, 640, 480, &TransformConfig::default());
assert!(result.is_err());
```
