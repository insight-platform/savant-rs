# savant-picasso

`savant-picasso` is a Skia-based compositor and encode-on-demand pipeline for DeepStream frames, built for bounding box rendering, label rendering, hot-swappable per-source specs, and worker lifecycle control. Import it as `picasso` (the Rust library module name is `picasso`) when you need `skia-safe` overlays, OpenGL / EGL / Vulkan-backed rendering, custom draw specs, font control, EOS flush, or idle eviction in a Rust video-processing service.

## What's inside

- `PicassoEngine` is the main entry point. It owns per-source workers, routes frames to them, stores default behavior in `GeneralSpec`, and manages shutdown plus idle-source eviction.
- `GeneralSpec`, `GeneralSpecBuilder`, `PtsResetPolicy`, and `EvictionDecision` define global engine behavior such as queue depth, what happens when PTS decreases, and whether an idle worker is kept alive or terminated.
- `SourceSpec`, `SourceSpecBuilder`, `CodecSpec`, `ConditionalSpec`, `ObjectDrawSpec`, and `CallbackInvocationOrder` define per-source behavior. This is where you choose drop, bypass, or encode paths, turn on Skia rendering, set fonts, and hot-swap draw specs per stream.
- `CodecSpec::Encode` combines a `TransformConfig` with a boxed `NvEncoderConfig`, so `savant-picasso` can transform, render, then encode with the same DeepStream encoder stack used elsewhere in the workspace.
- `Callbacks` and `CallbacksBuilder` collect the runtime hooks: `OnEncodedFrame`, `OnBypassFrame`, `OnRender`, `OnObjectDrawSpec`, `OnGpuMat`, `OnEviction`, and `OnStreamReset`.
- `OutputMessage` is the output payload delivered to encoded or bypass callbacks. `SourceWorker` is the per-source execution unit that manages encoder startup, EOS flush, hot swaps, and shutdown.
- The `skia` and `transform` modules provide the rendering primitives behind bounding box drawing, labels, blur/dot helpers, and letterbox-aware coordinate handling via `LetterboxParams`.

## Usage

```rust
use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, SurfaceView, TransformConfig, VideoFormat};
use deepstream_encoders::{cuda_init, EncoderConfig, JpegEncoderConfig, NvEncoderConfig};
use picasso::{Callbacks, CodecSpec, GeneralSpec, PicassoEngine, SourceSpec};
use savant_core::primitives::frame::{VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod};

gstreamer::init()?;
cuda_init(0)?;

let engine = PicassoEngine::new(GeneralSpec::builder().name("overlay").build(), Callbacks::default());
let spec = SourceSpec::builder()
    .codec(CodecSpec::Encode {
        transform: TransformConfig::default(),
        encoder: Box::new(NvEncoderConfig::new(
            0,
            EncoderConfig::Jpeg(JpegEncoderConfig::new(1280, 720)),
        )),
    })
    .build();
engine.set_source_spec("cam-1", spec)?;

let frame = VideoFrameProxy::new("cam-1", (30, 1), 1280, 720, VideoFrameContent::None, VideoFrameTranscodingMethod::Copy, None, None, (1, 1_000_000_000), 0, None, None)?;
let shared = BufferGenerator::new(VideoFormat::RGBA, 1280, 720, 30, 1, 0, NvBufSurfaceMemType::Default)?.acquire(Some(1))?;
let view = SurfaceView::from_buffer(&shared, 0)?;

engine.send_frame("cam-1", frame, view, None)?;
engine.send_eos("cam-1")?;
engine.shutdown();
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-picasso = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ plus the DeepStream buffer and encoder stack used by the selected `CodecSpec::Encode` pipeline.
- CUDA support for DeepStream surfaces and GPU-side rendering / transforms.
- Jetson and dGPU are both supported, but rendering throughput and encoder choices depend on the platform and graphics stack.
- Skia prebuilt binaries must match `skia-safe` exactly; this workspace currently uses `SKIA_BINARIES_VERSION=0.93.1` from `.envrc`.
- The Skia build in this workspace enables EGL / OpenGL / Vulkan-related features, so deploy hosts need the graphics libraries required by the backend you use.

## Documentation

- [docs.rs](https://docs.rs/savant-picasso)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
