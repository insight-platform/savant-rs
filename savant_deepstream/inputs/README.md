# savant-deepstream-inputs

`savant-deepstream-inputs` adapts compressed Savant video packets into DeepStream decode sessions, especially when frames arrive from ZeroMQ, RTSP, file source, MP4 demuxer, or similar ingest layers as `VideoFrameProxy` metadata plus payload bytes. Import it as `deepstream_inputs` (the Rust library module name is `deepstream_inputs`) when you need a flexible single-stream decoder, multi-source ingest pool, `SharedBuffer` delivery, codec detection, or JPEG/JFIF-aware packet handling.

## What's inside

- `flexible_decoder::FlexibleDecoder` is the single-stream entry point. It watches incoming `VideoFrameProxy` metadata, resolves codecs, creates an internal `NvDecoder` on demand, and recreates it when codec or resolution changes.
- `FlexibleDecoderConfig` controls GPU selection, buffer-pool size, idle drain timeout, keyframe detection buffering, and an optional `DecoderConfigCallback` that can rewrite the resolved `DecoderConfig` right before activation.
- `FlexibleDecoderOutput` is the callback payload for decoded frames, parameter changes, skipped packets, orphan frames, per-source EOS, GStreamer events, and decode errors. `DecoderParameters`, `SkipReason`, and `SealedDelivery` make it practical to reason about decoder state and buffer ownership.
- `decoder_pool::FlexibleDecoderPool` and `FlexibleDecoderPoolConfig` scale the same model across many sources. The pool creates one `FlexibleDecoder` per `source_id`, routes packets automatically, and evicts idle decoders with `EvictionDecision::{Keep, Evict}`.
- `codec_resolve` contains `resolve_video_codec`, `CodecResolve`, and `DetectionStrategy`. That is where H.264 / HEVC keyframe probing, JPEG backend selection, raw uploader mapping, and unsupported codec rejection are centralized.
- The crate builds on `savant-deepstream-decoders`, so it does not open RTSP sockets or ZeroMQ connections by itself; instead, it is the handoff point between upstream transport / demux code and downstream DeepStream decode.

## Usage

```rust
use deepstream_inputs::flexible_decoder::{
    FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderOutput,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use std::time::Duration;

let decoder = FlexibleDecoder::new(
    FlexibleDecoderConfig::new("cam-1", 0, 4)
        .idle_timeout(Duration::from_secs(2))
        .detect_buffer_limit(30),
    |out| {
        if let FlexibleDecoderOutput::Frame { frame, decoded, .. } = out {
            println!("{} -> {:?}", frame.get_source_id(), decoded.format);
        }
    },
);

let frame = VideoFrameProxy::new(
    "cam-1",
    (30, 1),
    1280,
    720,
    VideoFrameContent::Internal(vec![/* compressed packet bytes */]),
    VideoFrameTranscodingMethod::Copy,
    Some(VideoCodec::Jpeg),
    None,
    (1, 1_000_000_000),
    0,
    None,
    None,
)?;

decoder.submit(&frame, None)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-deepstream-inputs = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ plus the decoder plugins required by `savant-deepstream-decoders`; practical deployments usually pair this crate with DeepStream parsers, demuxers, and upstream packet sources.
- CUDA driver and toolkit support when using GPU decode or raw uploader paths.
- Jetson and dGPU are both supported through the underlying decoder crate; platform-specific decode tuning is applied in the resolved `DecoderConfig`.
- JPEG-heavy ingest may rely on `jfifdump` for packet inspection alongside DeepStream decode.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-inputs)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
