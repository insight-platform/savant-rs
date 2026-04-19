# savant-deepstream-sys

`savant-deepstream-sys` is the raw FFI layer for NVIDIA DeepStream 7 in the Savant real-time video-analytics stack. It exposes bindgen-generated Rust declarations for DeepStream metadata, inference, tracker, and `gstnvdsmeta` entry points, so low-level code can call the C API directly when safe wrappers are not enough.

## What's inside

- The crate exports a single generated module, `gstnvdsmeta`, and re-exports its contents at the crate root for direct access.
- The bindings cover DeepStream C headers included by `gstnvdsmeta_rs.h`: `gstnvdsmeta.h`, `gstnvdsinfer.h`, `nvdsmeta.h`, `nvdsinfer_context.h`, `nvds_tracker_meta.h`, and `gst-nvevent.h`.
- Raw metadata structs such as `NvDsBatchMeta`, `NvDsFrameMeta`, `NvDsObjectMeta`, and `NvDsUserMeta` are available for manual traversal of batch, frame, object, and user metadata lists.
- Raw tensor and inference structs such as `NvDsInferDims` and `NvDsInferTensorMeta` are exposed for direct parsing of `nvinfer` outputs.
- Raw buffer-side types such as `NvBufSurfaceParams` are available for DeepStream surface and color-format interop.
- Low-level functions such as `nvds_meta_api_get_type`, `gst_buffer_get_nvds_batch_meta`, `nvds_create_batch_meta`, `nvds_acquire_frame_meta_from_pool`, `nvds_add_obj_meta_to_frame`, `nvds_acquire_meta_lock`, and `nvds_release_meta_lock` are exposed exactly as the C API defines them.
- Because this is a `-sys` crate, helper types like `__BindgenBitfieldUnit` and other generated layout shims are also part of the public surface.

## Usage

`savant-deepstream-sys` intentionally exposes unsafe raw bindings only; for most applications, prefer the safe wrapper crates [`savant-deepstream`](https://crates.io/crates/savant-deepstream) and [`savant-deepstream-buffers`](https://crates.io/crates/savant-deepstream-buffers).

## Install

```toml
[dependencies]
savant-deepstream-sys = "2"
```

This crate does not expose Cargo features today.

## System requirements

Requires Linux with the NVIDIA DeepStream SDK 7.x installed under `/opt/nvidia/deepstream/deepstream`, plus the CUDA toolkit, GStreamer headers, and GLib headers needed by bindgen and linking.

Both Jetson (`aarch64`) and dGPU (`x86_64`) targets are supported in the build script. The `docs.rs` build is limited to `x86_64-unknown-linux-gnu`.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-sys)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
