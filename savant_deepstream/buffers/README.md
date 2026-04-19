# savant-deepstream-buffers

`savant-deepstream-buffers` is Savant's safe Rust layer for `NvBufSurface` allocation, zero-copy GPU buffer access, and batched surface handling on NVIDIA DeepStream. It is the crate you use to allocate NV12 or RGBA surfaces, inspect batch slots through `SurfaceView`, build homogeneous or heterogeneous batches, run CUDA-backed transforms, and pass GPU buffers through GStreamer without giving up ownership safety.

## What's inside

- `BufferGenerator` and `BufferGeneratorBuilder` allocate single-frame `NvBufSurface` buffers from a DeepStream buffer pool. They are the simplest entry point when you need one GPU buffer at a time plus matching `video/x-raw` or `memory:NVMM` caps.
- `UniformBatchGenerator` and `UniformBatchGeneratorBuilder` allocate homogeneous batched surfaces where every slot shares the same format and resolution. They feed `SurfaceBatch`, which supports `transform_slot`, `finalize`, `view`, `shared_buffer`, `num_filled`, and `max_batch_size`.
- `NonUniformBatch` assembles a zero-copy heterogeneous batch from existing `SurfaceView` inputs. It copies `NvBufSurfaceParams`, keeps parents alive with `GstParentBufferMeta`, and finalizes into a synthetic batched descriptor without allocating new GPU memory.
- `SharedBuffer` is the shared ownership handle for a `gst::Buffer`. It offers `lock`, `into_buffer`, `with_view`, timestamp helpers, `SavantIdMeta` helpers, and direct batch-to-batch `transform_into` support.
- `SurfaceView` is the per-slot zero-copy accessor for CUDA-addressable surface memory. It exposes cached width, height, pitch, slot index, GPU ID, color format, and GPU pointer, plus data-path methods such as `memset`, `fill`, `upload`, and `transform_into`.
- `CudaStream` gives transform and surface operations an owned CUDA stream handle when you want to avoid implicit synchronization on the default stream.
- `transform` exports the core transform vocabulary: `TransformConfig`, `TransformConfigBuilder`, `Padding`, `DstPadding`, `Interpolation`, `ComputeMode`, `Rect`, `TransformError`, `buffer_gpu_id`, `extract_nvbufsurface`, and `MIN_EFFECTIVE_DIM`.
- `surface_readers::{read_surface_header, read_slot_dimensions}` are small utilities for inspecting `NvBufSurface` layout from GStreamer buffers.
- `MetaClearPolicy` and `bridge_savant_id_meta` help integrate batch buffers with larger pipelines, especially when metadata must survive hardware encoders or be cleared between stages.
- The crate re-exports `VideoFormat`, `SavantIdMeta`, and `SavantIdMetaKind` so callers can work with formats and Savant IDs from one place.
- With the optional `skia` feature enabled, `SkiaRenderer`, `egl_context::EglHeadlessContext`, and the Skia or EGL integration modules add GPU rendering on top of `SurfaceView`.

## Usage

```rust
use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, SurfaceView, VideoFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    gstreamer::init()?;
    deepstream_buffers::cuda_init(0)?;

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        480,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )?;

    let shared = gen.acquire(Some(42))?;
    let view = SurfaceView::from_buffer(&shared, 0)?;
    view.memset(0)?;
    println!("{}x{} {:?}", view.width(), view.height(), view.data_ptr());
    Ok(())
}
```

## Install

```toml
[dependencies]
savant-deepstream-buffers = "2"
```

`skia`: enables `SkiaRenderer`, EGL context management, and Skia-based rendering on surface views.

## System requirements

Requires Linux with NVIDIA DeepStream SDK 7.x, CUDA, and GStreamer 1.24+ development headers. The crate is designed for NVIDIA Jetson and dGPU systems and relies on DeepStream's `NvBufSurface` allocator plus CUDA-backed transform APIs.

On Jetson, `SurfaceView` uses EGL-CUDA interop to resolve CUDA-addressable pointers for VIC-managed surfaces. With the `skia` feature enabled, you also need the Skia and OpenGL dependencies required by `skia-safe`.

The `docs.rs` build is limited to `x86_64-unknown-linux-gnu`.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-buffers)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
