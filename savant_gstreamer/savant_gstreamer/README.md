# savant-gstreamer

`savant-gstreamer` is a GStreamer and gstreamer-rs utility crate for Savant, the real-time video-analytics framework. It packages the small but frequently reused building blocks for pipeline assembly: `appsrc`/`appsink` runners, `VideoFormat` and codec-to-caps mappings, MP4 muxer and demuxer helpers, custom `GstMeta`, and convenient buffer wrappers for RTSP, MP4, H.264, HEVC, NV12, and RGBA workflows.

## What's inside

- `VideoFormat` maps common raw video formats such as `RGBA`, `BGRx`, `NV12`, `NV21`, `I420`, `UYVY`, and `GRAY8` to their GStreamer caps names.
- `Codec` maps encoded formats such as H.264, HEVC, JPEG, AV1, VP8, VP9, and raw RGB/RGBA/NV12 payloads to parser, encoder, decoder, and caps strings. This is the crate's lightweight lookup layer for element selection and caps negotiation.
- `GstBuffer` is a shared `gst::Buffer` wrapper with helpers for PTS, DTS, duration, flags, offsets, deep copies, memory inspection, and `SavantIdMeta` attachment or removal.
- `id_meta::SavantIdMeta` and `SavantIdMetaKind` define a custom `GstMeta` used to carry Savant frame or batch identifiers through a pipeline.
- `mp4_muxer::Mp4Muxer` builds a minimal `appsrc -> parser -> qtmux -> filesink` pipeline for writing encoded packets into MP4.
- `mp4_demuxer::{Mp4Demuxer, DemuxedPacket, Mp4DemuxerOutput, VideoInfo}` builds a `filesrc -> qtdemux -> queue -> appsink` pipeline, emits elementary-stream packets with PTS, DTS, duration, and keyframe information, and exposes container-level `VideoInfo` (codec, encoded width/height, framerate) via the new `Mp4DemuxerOutput::StreamInfo` variant and the `Mp4Demuxer::video_info()` / `wait_for_video_info()` accessors.
- `pipeline::{GstPipeline, PipelineConfig, PipelineInput, PipelineOutput, PtsPolicy}` provides a reusable bounded-channel pipeline runner around `appsrc` and `appsink`, with timeout handling, event forwarding, metadata bridging, and orderly EOS or shutdown behavior.
- `pipeline::bridge_meta::bridge_savant_id_meta_across` preserves `SavantIdMeta` across elements that allocate fresh output buffers, such as hardware encoders.
- `pipeline::source_eos::{build_source_eos_event, parse_source_eos_event}` helps encode and decode per-source EOS custom events.
- `pipeline::set_element_property` is a small convenience for dynamic element property assignment from strings.

## Usage

```rust
use gstreamer as gst;
use savant_gstreamer::pipeline::{GstPipeline, PipelineConfig, PipelineInput};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    gst::init()?;

    let identity = gst::ElementFactory::make("identity").build()?;
    let config = PipelineConfig {
        name: "passthrough".to_string(),
        appsrc_caps: gst::Caps::builder("application/octet-stream").build(),
        elements: vec![identity],
        input_channel_capacity: 16,
        output_channel_capacity: 16,
        operation_timeout: Some(Duration::from_secs(5)),
        drain_poll_interval: Duration::from_millis(10),
        idle_flush_interval: None,
        appsrc_probe: None,
        pts_policy: None,
        leak_on_finalize: false,
    };

    let (tx, _rx, mut pipeline) = GstPipeline::start(config)?;
    tx.send(PipelineInput::Buffer(gst::Buffer::from_mut_slice(vec![1, 2, 3])))?;
    tx.send(PipelineInput::Eos)?;
    pipeline.shutdown()?;
    Ok(())
}
```

## Install

```toml
[dependencies]
savant-gstreamer = "2"
```

This crate does not expose Cargo features today.

## System requirements

Requires GStreamer 1.24+ development headers at build time.

If you use the codec helpers with NVIDIA hardware elements such as `nvv4l2h264enc`, `nvv4l2h265enc`, `nvv4l2decoder`, or `nvjpegenc`, those plugins must also be present in the runtime GStreamer installation.

## Documentation

- [docs.rs](https://docs.rs/savant-gstreamer)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
