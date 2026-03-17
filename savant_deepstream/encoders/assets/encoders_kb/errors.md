# EncoderError Reference

```rust
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    #[error("NVENC hardware not available on GPU {gpu_id} (required for {codec})")]
    NvencNotAvailable { codec: String, gpu_id: u32 },

    #[error("Invalid encoder property '{name}': {reason}")]
    InvalidProperty { name: String, reason: String },

    #[error("Input PTS reordering detected: frame {frame_id} has PTS {pts_ns} which is <= previous PTS {prev_pts_ns}")]
    PtsReordered { frame_id: u128, pts_ns: u64, prev_pts_ns: u64 },

    #[error("Output PTS reordering detected (B-frames?): ...")]
    OutputPtsReordered { frame_id: u128, pts_ns: u64, prev_pts_ns: u64 },

    #[error("Output DTS > PTS detected (B-frames?): ...")]
    OutputDtsExceedsPts { frame_id: u128, dts_ns: u64, pts_ns: u64 },

    #[error("GStreamer pipeline error: {0}")]
    PipelineError(String),

    #[error("Failed to create GStreamer element '{0}'")]
    ElementCreationFailed(String),

    #[error("Failed to link GStreamer elements: {0}")]
    LinkFailed(String),

    #[error("Failed to acquire buffer: {0}")]
    BufferAcquisitionFailed(String),

    #[error("Encoder has been finalized (EOS sent), no more frames can be submitted")]
    AlreadyFinalized,

    #[error("NvBufSurface error: {0}")]
    NvBufSurfaceError(#[from] deepstream_buffers::NvBufSurfaceError),
}
```

## When Each Variant is Returned

| Variant | Trigger |
|---|---|
| `NvencNotAvailable` | `NvEncoder::new()` when codec is H264/HEVC/AV1 and `nvidia_gpu_utils::has_nvenc(gpu_id)` returns false. Orin Nano, some datacenter GPUs. |
| `UnsupportedCodec` | `EncoderProperties::from_pairs(Codec::Av1, Platform::Jetson, _)` — AV1 not on Jetson. |
| `InvalidProperty` | Wrong property for codec/platform, or `encoder_params` codec mismatch, or PNG with non-RGBA format, or RawProps given properties. |
| `PtsReordered` | `submit_frame` when pts_ns <= last submitted PTS. |
| `OutputPtsReordered` | `pull_encoded*` when output PTS < last output PTS (B-frame reordering). |
| `OutputDtsExceedsPts` | `pull_encoded*` when output DTS > PTS (B-frame reordering). |
| `AlreadyFinalized` | `submit_frame` after `finish()`. |
| `PipelineError` | GStreamer bus error, appsrc push failure, EOS failure, NvBufSurfTransform failure. |
| `ElementCreationFailed` | GStreamer element factory cannot create element (missing plugin). |
| `LinkFailed` | GStreamer elements cannot be linked (caps negotiation failure). |
| `BufferAcquisitionFailed` | Buffer not writable. |
| `NvBufSurfaceError` | Upstream pool/transform error (via `From` impl). |

## Testing Error Paths

### NVENC not available
```rust
// Only testable on hardware without NVENC (Orin Nano):
if !has_nvenc() {
    let config = EncoderConfig::new(Codec::H264, 320, 240);
    match NvEncoder::new(&config) {
        Err(EncoderError::NvencNotAvailable { .. }) => {}
        other => panic!("Expected NvencNotAvailable, got {:?}", other),
    }
}
```

### PTS reordering
```rust
encoder.submit_frame(buf1, 0, 100, None).unwrap();
let result = encoder.submit_frame(buf2, 1, 50, None);
assert!(matches!(result, Err(EncoderError::PtsReordered { .. })));
```

### After finalize
```rust
let _ = encoder.finish(Some(1000));
let result = encoder.submit_frame(buf, 0, 0, None);
assert!(matches!(result, Err(EncoderError::AlreadyFinalized)));
```

### Codec mismatch in properties
```rust
let config = EncoderConfig::new(Codec::H264, 640, 480)
    .properties(EncoderProperties::HevcDgpu(HevcDgpuProps::default()));
assert!(NvEncoder::new(&config).is_err());
```
