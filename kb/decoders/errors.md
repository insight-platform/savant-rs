# DecoderError Reference

```rust
pub enum DecoderError {
    GstInit(String),
    NvdecNotAvailable { codec: String, gpu_id: u32 },
    InvalidProperty { name: String, reason: String },
    PtsReordered { frame_id: u128, pts_ns: u64, prev_pts_ns: u64 },
    PipelineError(String),
    ElementCreationFailed(String),
    LinkFailed(String),
    BufferError(String),
    ShuttingDown,
    ChannelDisconnected,
    PipelineFailed,
    AlreadyFinalized,
    FrameworkError(savant_gstreamer::pipeline::PipelineError),
    NvBufSurfaceError(deepstream_buffers::NvBufSurfaceError),
}
```

## Typical Triggers

- `GstInit`: `gst::init()` failed
- `NvdecNotAvailable`: V4L2 decoder path requested but hardware/plugin missing
- `InvalidProperty`: bad config (for example zero raw dimensions or missing codec_data)
- `PtsReordered`: non-monotonic input PTS on `submit_packet`
- `PipelineError`: GStreamer setup/runtime failures
- `BufferError`: pool acquire/map/upload/transform failures and image decode failures
- `ShuttingDown`: input submitted while graceful shutdown is in progress
- `ChannelDisconnected`: framework channel was disconnected
- `PipelineFailed`: framework pipeline entered terminal failed state (timeout watchdog)
- `AlreadyFinalized`: `submit_packet` called after `send_eos`
- `FrameworkError`: error propagated from `savant_gstreamer::pipeline`

## Related Event Errors

`NvDecoderOutput` may deliver:

- `Error(DecoderError)` as regular output item on the receiver API.

The decoder no longer exposes `PipelineRestarted`; failure signaling is via
`NvDecoderOutput::Error(...)` and terminal `PipelineFailed`.
