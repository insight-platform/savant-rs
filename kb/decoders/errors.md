# DecoderError Reference

```rust
pub enum DecoderError {
    NvdecNotAvailable { codec: String, gpu_id: u32 },
    InvalidProperty { name: String, reason: String },
    PtsReordered { frame_id: u128, pts_ns: u64, prev_pts_ns: u64 },
    PipelineError(String),
    ElementCreationFailed(String),
    LinkFailed(String),
    BufferError(String),
    AlreadyFinalized,
    NvBufSurfaceError(deepstream_buffers::NvBufSurfaceError),
}
```

## Typical Triggers

- `NvdecNotAvailable`: V4L2 decoder path requested but hardware/plugin missing
- `InvalidProperty`: bad config (for example zero raw dimensions or missing codec_data)
- `PtsReordered`: non-monotonic input PTS on `submit_packet`
- `PipelineError`: GStreamer setup/runtime failures and drain thread/restart failures
- `BufferError`: pool acquire/map/upload/transform failures and image decode failures
- `AlreadyFinalized`: `submit_packet` called after `send_eos`

## Related Event Errors

`DecoderEvent` may deliver:

- `Error(DecoderError)`
- `PipelineRestarted { reason, lost_frame_count }`

`PipelineRestarted` is an event, not a `DecoderError` variant.
