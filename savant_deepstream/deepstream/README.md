# savant-deepstream

`savant-deepstream` is the safe metadata wrapper crate for NVIDIA DeepStream inside Savant, the real-time video-analytics framework. It turns the raw DeepStream C metadata graph into idiomatic Rust types for frames, objects, tensors, tracker output, and user metadata on both Jetson and dGPU systems.

## What's inside

- `BatchMeta` is the entry point for batch-level DeepStream metadata. It wraps `NvDsBatchMeta`, keeps the DeepStream metadata lock alive through RAII, exposes `frames()` and `batch_user_meta()`, and pairs with `ensure_nvds_meta_api_registered()` when reading metadata from a `GstBuffer`.
- `clear_all_frame_objects` is a focused helper for clearing stale `NvDsObjectMeta` entries from every frame in a batch buffer before or after a stage such as inference or tracking.
- `FrameMeta` wraps `NvDsFrameMeta` and exposes frame number, PTS, source ID, dimensions, batch slot, geometry helpers, attached objects, and frame-level user metadata.
- `ObjectMeta` wraps `NvDsObjectMeta` and is used for detections, ROIs, and tracker outputs. It exposes class and object IDs, confidence fields, labels, bounding-box helpers based on `rect_params`, parent or child relationships, and object-level user metadata.
- `UserMeta` wraps `NvDsUserMeta` and lets you inspect `meta_type`, raw user pointers, and typed tensor output via `as_infer_tensor_meta()`.
- `InferTensorMeta` and `InferDims` provide safe read access to `nvinfer` tensor output: output layer names, per-layer dimensions, data types, host or device buffer pointers, `network_info`, `gpu_id`, and `maintain_aspect_ratio`.
- `tracker_meta` parses DeepStream tracker side-channel data into owned Rust values: `TrackState`, `TargetMiscFrame`, `TargetMiscObject`, `TargetMiscStream`, `TargetMiscBatch`, and `target_misc_batch_from_user_meta()` for shadow, terminated, or past-frame tracker lists.
- `DeepStreamError` and the crate-local `Result<T>` unify error handling across the wrapper API.
- The raw FFI crate is re-exported as `sys` for advanced cases where a safe wrapper is not yet available.

## Usage

```rust
use deepstream::BatchMeta;

unsafe fn inspect_batch(buf_ptr: *mut deepstream::sys::GstBuffer) -> deepstream::Result<()> {
    let batch = BatchMeta::from_gst_buffer(buf_ptr)?;

    for frame in batch.frames() {
        println!("source={} objects={}", frame.source_id(), frame.num_objects());

        for user_meta in frame.user_meta() {
            if let Some(tensor) = user_meta.as_infer_tensor_meta() {
                println!("{:?}", tensor.layer_dimensions());
            }
        }
    }

    Ok(())
}
```

## Install

```toml
[dependencies]
savant-deepstream = "2"
```

This crate does not expose Cargo features today.

## System requirements

Requires Linux with NVIDIA DeepStream SDK 7.x installed at build and run time. It links against the DeepStream metadata libraries provided by the SDK and is typically used inside GStreamer and DeepStream applications on Jetson or NVIDIA dGPU hosts.

The `docs.rs` build is limited to `x86_64-unknown-linux-gnu`.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
