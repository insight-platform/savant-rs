# savant-deepstream-decoders

`savant-deepstream-decoders` is a Rust DeepStream decode layer for NVDEC-backed video ingest, CPU image decode fallbacks, and raw uploader paths. Import it as `deepstream_decoders` (the Rust library module name is `deepstream_decoders`) when you need typed `nvv4l2decoder` configuration, H.264 / HEVC / AV1 / MJPEG decode, codec detection, or a reusable building block for multi-stream pipelines that decode each source into GPU RGBA buffers.

## What's inside

- `NvDecoder`, `NvDecoderOutput`, and `DecodedFrame` are the main runtime types. They expose a pull-based API for decoded frames, downstream events, logical per-source EOS markers, and pipeline errors.
- `DecoderConfig` groups all supported codecs: `H264DecoderConfig`, `HevcDecoderConfig`, `Vp8DecoderConfig`, `Vp9DecoderConfig`, `Av1DecoderConfig`, `JpegDecoderConfig`, `PngDecoderConfig`, `RawRgbaDecoderConfig`, and `RawRgbDecoderConfig`.
- H.264 and HEVC configuration is explicit about packet format through `H264StreamFormat` and `HevcStreamFormat`, including Annex-B vs length-prefixed AVCC / HVCC bitstreams.
- `JpegBackend` lets you choose the GPU JPEG path (`nvjpegdec`) or the CPU fallback path. The raw decoder variants are a convenient uploader from tightly packed RGBA / RGB bytes into `SharedBuffer`.
- `detect_stream_config` and `is_random_access_point` inspect one access unit to decide how to configure H.264 / HEVC decode. They are especially useful when upstream code reads MP4 demuxer output or receives packets from a network source and needs codec detection before building the decoder.
- The crate re-exports `BufferGenerator`, `SharedBuffer`, `SurfaceView`, `TransformConfig`, `VideoFormat`, and `cuda_init`, which makes it easy to wire decode output into later DeepStream, TensorRT, or rendering stages.
- `CudadecMemtype` exposes the desktop `cudadec-memtype` setting, while Jetson-specific low-latency options live on the Jetson-only fields in the per-codec config structs.

## Usage

```rust
use deepstream_decoders::{
    detect_stream_config, BufferGenerator, NvDecoder, NvDecoderConfig, NvDecoderOutput,
    TransformConfig, VideoFormat, Codec,
};
use deepstream_buffers::NvBufSurfaceMemType;
use std::time::Duration;

gstreamer::init()?;

let first_access_unit: Vec<u8> = /* H.264 or HEVC packet bytes */;
let decoder_cfg = detect_stream_config(Codec::H264, &first_access_unit).expect("stream config");

let pool = BufferGenerator::new(
    VideoFormat::RGBA,
    1920,
    1080,
    30,
    1,
    0,
    NvBufSurfaceMemType::Default,
)?;

let decoder = NvDecoder::new(
    NvDecoderConfig::new(0, decoder_cfg).operation_timeout(Duration::from_secs(5)),
    pool,
    TransformConfig::default(),
)?;

decoder.submit_packet(&first_access_unit, 1, 0, Some(0), Some(33_333_333))?;
decoder.graceful_shutdown(Some(Duration::from_secs(2)), |out| {
    if let NvDecoderOutput::Frame(frame) = out {
        println!("decoded {:?} {}", frame.format, frame.pts_ns);
    }
})?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-deepstream-decoders = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ with `nvv4l2decoder`, parser elements, and JPEG decode support; CPU image decode paths rely on standard image codecs in addition to DeepStream.
- CUDA driver and toolkit support for NVDEC and raw uploader paths on dGPU.
- Jetson and dGPU are both supported, but low-latency settings, memory types, and available hardware decoder behavior differ by platform.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-decoders)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
