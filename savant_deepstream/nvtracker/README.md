# savant-deepstream-nvtracker

`savant-deepstream-nvtracker` is a Rust wrapper for DeepStream object tracking with multi-source batching, track ID extraction, and per-stream reset handling. Import it as `deepstream_nvtracker` (the Rust library module name is `deepstream_nvtracker`; the underlying GStreamer element string stays `"nvtracker"`) when you need `nvtracker` with IoU tracker, NvSORT, or NvDCF configs, graceful shutdown, source EOS, and automatic reset on resolution change or PTS discontinuity.

## What's inside

- `NvTracker`, `NvTrackerOutput`, and `TrackedFrame` are the main runtime types. You submit one or more per-source GPU frames together with ROI detections, then receive tracking output, events, EOS markers, or errors.
- `NvTrackerConfig` captures tracker dimensions, `max_batch_size`, low-level library/config paths, GPU id, input format, timeout behavior, and `TrackingIdResetMode`.
- `TrackingIdResetMode` maps directly to DeepStream tracker reset policy, so callers can choose whether track IDs reset on stream reset, EOS, or both.
- `Roi` is the input detection primitive, while `attach_detection_meta` writes those detections into DeepStream metadata before tracking.
- `TrackerOutput`, `TrackedObject`, `MiscTrackData`, `MiscTrackFrame`, and `TrackState` expose current tracks plus shadow, terminated, and past-frame tracker metadata. This is the part you use when you need object tracker history rather than just current boxes.
- `default_ll_lib_path` helps locate the DeepStream multi-object-tracker shared library, while low-level tracker behavior comes from YAML configs such as IoU tracker, NvSORT, or NvDCF profiles.
- `NvTrackerBatchingOperator`, `NvTrackerBatchingOperatorConfig`, `TrackerOperatorOutput`, `TrackerOperatorTrackingOutput`, and `SealedDeliveries` provide a higher-level operator for code that batches frames before tracking.
- Internally, the runtime validates per-source dimensions and continuity, and will reset an individual stream when the same source changes resolution or regresses in input PTS.

## Usage

```rust
use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, VideoFormat};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, NvTrackerOutput, Roi, TrackedFrame,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;

let mut cfg = NvTrackerConfig::new(
    default_ll_lib_path(),
    "assets/config_tracker_IOU.yml",
);
cfg.tracker_width = 640;
cfg.tracker_height = 384;

let tracker = NvTracker::new(cfg)?;
let buffer = BufferGenerator::new(VideoFormat::RGBA, 640, 384, 30, 1, 0, NvBufSurfaceMemType::Default)?
    .acquire(None)?;

let frame = TrackedFrame {
    source: "cam-1".to_string(),
    buffer,
    rois: HashMap::from([(
        0,
        vec![Roi::new(1, RBBox::ltwh(40.0, 40.0, 80.0, 60.0).unwrap())],
    )]),
};

tracker.submit(&[frame], vec![SavantIdMetaKind::Frame(1)])?;
if let NvTrackerOutput::Tracking(output) = tracker.recv()? {
    println!("tracks={}", output.current_tracks.len());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-deepstream-nvtracker = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ with the DeepStream `nvtracker` plugin and the low-level multi-object tracker library.
- CUDA and the DeepStream tracker runtime appropriate for your platform.
- Jetson and dGPU are both supported; choose low-level tracker YAML and performance settings for the hardware you deploy on.
- Tracking quality and behavior depend on the selected tracker backend such as NvDCF, IoU tracker, or NvSORT.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-nvtracker)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
