# savant-deepstream-encoders

`savant-deepstream-encoders` is a Rust wrapper over NVIDIA DeepStream hardware encoder pipelines for H.264, HEVC/H.265, AV1, JPEG, PNG, and raw frame download paths. Import it as `deepstream_encoders` (the Rust library module name is `deepstream_encoders`) when you need typed NVENC configuration, `nvv4l2h264enc` / `nvv4l2h265enc` / `nvjpegenc` setup, bitrate control, IDR and keyframe handling, or RGBA / RGB downloader flows on Jetson and dGPU.

## What's inside

- `NvEncoder`, `NvEncoderOutput`, and `EncodedFrame` form the runtime API. You submit GPU-backed buffers, then pull `Frame`, `SourceEos`, `Event`, `Eos`, or `Error` outputs from the encoder.
- `EncoderConfig` is the top-level codec enum, with typed per-codec builders: `H264EncoderConfig`, `HevcEncoderConfig`, `Av1EncoderConfig`, `JpegEncoderConfig`, `PngEncoderConfig`, and `RawEncoderConfig`.
- `NvEncoderConfig` adds pipeline-level knobs such as `gpu_id`, channel capacities, memory type, and timeouts around a codec-specific `EncoderConfig`.
- `properties` exposes platform-aware tuning structs for real hardware encoder work: `H264DgpuProps`, `HevcDgpuProps`, `Av1DgpuProps`, Jetson counterparts, plus enums such as `RateControl`, `H264Profile`, `HevcProfile`, `DgpuPreset`, `TuningPreset`, and `JetsonPresetLevel`. These are the knobs you use for CBR / VBR, bitrate control, IDR cadence, preset selection, and similar NVENC tuning.
- The crate re-exports `BufferGenerator`, `SharedBuffer`, `SurfaceView`, `VideoFormat`, and `cuda_init` from `deepstream_buffers`, so the common workflow is: acquire an NVMM buffer, render or transform into it, then hand it to `NvEncoder`.
- The encoder always disables B-frames in the underlying GStreamer elements, so callers get monotonic output ordering and simpler frame correlation. `EncodedFrame::keyframe` tells you whether the packet is intra-coded.
- Besides compressed codecs, the raw variants (`Codec::RawRgba`, `Codec::RawRgb`, `Codec::RawNv12`) act as a downloader that turns GPU frames into tightly packed bytes for diagnostics, handoff, or CPU-side post-processing.

## Usage

```rust
use deepstream_encoders::{
    cuda_init, EncoderConfig, H264EncoderConfig, NvEncoder, NvEncoderConfig, NvEncoderOutput,
    VideoFormat,
};
use std::time::Duration;

gstreamer::init()?;
cuda_init(0)?;

let encoder_cfg = EncoderConfig::H264(
    H264EncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .fps(30, 1),
);

let encoder = NvEncoder::new(
    NvEncoderConfig::new(0, encoder_cfg)
        .name("camera-a")
        .operation_timeout(Duration::from_secs(5)),
)?;

let buffer = encoder
    .generator()
    .lock()
    .acquire(Some(1))?
    .into_buffer()
    .expect("sole owner");

encoder.submit_frame(buffer, 1, 0, Some(33_333_333))?;
encoder.graceful_shutdown(Some(Duration::from_secs(2)), |out| {
    if let NvEncoderOutput::Frame(frame) = out {
        println!("{} bytes, keyframe={}", frame.data.len(), frame.keyframe);
    }
})?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-deepstream-encoders = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ with the DeepStream / NVIDIA video plugins that provide `nvv4l2h264enc`, `nvv4l2h265enc`, `nvv4l2av1enc`, parser elements, and `nvjpegenc`; PNG encoding additionally needs `pngenc` from `gst-plugins-good`.
- CUDA driver and toolkit support for NVENC on dGPU systems.
- Jetson and dGPU are both supported, but they use different property structs and hardware capabilities; AV1, JPEG, preset, and memory behavior differ by platform.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-encoders)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
