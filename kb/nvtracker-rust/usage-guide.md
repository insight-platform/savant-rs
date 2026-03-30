# NvTracker — Rust Usage Guide

Comprehensive code samples for all batch patterns. Every example assumes DeepStream 7.1, GPU 0, and the IOU low-level tracker.

## Prerequisites

```rust
use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, VideoFormat,
};
use nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, Roi, TrackedFrame,
    TrackingIdResetMode, TrackerOutput,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;
```

## Configuration

```rust
let mut config = NvTrackerConfig::new(
    default_ll_lib_path(),
    "/path/to/config_tracker_IOU.yml",
);
config.tracker_width = 320;
config.tracker_height = 240;
config.max_batch_size = 4;               // must be >= max frames in any call
config.tracking_id_reset_mode = TrackingIdResetMode::OnStreamReset;
config.queue_depth = 0;                  // 0 = synchronous; >0 = insert GStreamer queue
```

`max_batch_size` controls the pad probe response to `gst_nvquery_batch_size` / `gst_nvquery_numStreams_size`. Set it to the maximum number of frames you will ever push in a single `track`/`track_sync` call.

`queue_depth` inserts a GStreamer `queue` element between `appsrc` and `nvtracker` when set to a value greater than zero. This decouples the push thread from the tracker processing thread, which can help absorb latency spikes. Set to `0` (default) for synchronous operation.

> **Note:** The `sub-batches` element property is **rejected** by `NvTracker`. Each instance is its own isolated tracker — create separate `NvTracker` instances for different tracking workloads instead of sub-batching within one element.

## Creating the tracker

```rust
let callback: Box<dyn FnMut(TrackerOutput) + Send> = Box::new(|output| {
    for t in &output.current_tracks {
        println!("{}: obj {} class {} at ({},{} {}x{})",
            t.source_id, t.object_id, t.class_id,
            t.bbox_left, t.bbox_top, t.bbox_width, t.bbox_height);
    }
});
let mut tracker = NvTracker::new(config, callback)?;
```

## Buffer helper

Each `TrackedFrame` requires a single-surface `SharedBuffer`. Use `BufferGenerator`:

```rust
fn make_buffer(w: u32, h: u32) -> SharedBuffer {
    BufferGenerator::new(
        VideoFormat::RGBA, w, h, 30, 1, 0, NvBufSurfaceMemType::Default,
    ).unwrap()
    .acquire(None)
    .unwrap()
}
```

The tracker builds a `NonUniformBatch` internally from the per-frame buffers — callers never need to construct batches manually.

## Pattern A: Single-source tracking

One camera, one frame per call. The simplest case.

```rust
let frame = TrackedFrame {
    source: "cam-1".to_string(),
    buffer: make_buffer(320, 240),
    rois: HashMap::from([(0, vec![
        Roi { id: 1, bbox: RBBox::ltwh(40.0, 40.0, 80.0, 60.0).unwrap() },
        Roi { id: 2, bbox: RBBox::ltwh(180.0, 100.0, 70.0, 70.0).unwrap() },
    ])]),
};
let ids = vec![SavantIdMetaKind::Frame(1)];
let output = tracker.track_sync(&[frame], ids)?;
assert_eq!(output.current_tracks.len(), 2);
```

## Pattern B: Multi-source (stream isolation)

Two cameras in one call. Each gets independent track IDs.

```rust
let roi = Roi { id: 1, bbox: RBBox::ltwh(50.0, 50.0, 60.0, 60.0).unwrap() };

let frames = vec![
    TrackedFrame {
        source: "cam-a".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![roi.clone()])]),
    },
    TrackedFrame {
        source: "cam-b".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![roi.clone()])]),
    },
];
let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
let out = tracker.track_sync(&frames, ids)?;
let by_src: HashMap<_, _> = out.current_tracks.iter()
    .map(|t| (t.source_id.clone(), t.object_id))
    .collect();
assert_ne!(by_src["cam-a"], by_src["cam-b"]);
```

## Pattern C: Multi-frame per source (temporal batch)

Two frames from the **same** camera in one call. The tracker processes them sequentially.

```rust
let frames = vec![
    TrackedFrame {
        source: "cam-a".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![
            Roi { id: 1, bbox: RBBox::ltwh(50.0, 50.0, 60.0, 60.0).unwrap() },
        ])]),
    },
    TrackedFrame {
        source: "cam-a".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![
            Roi { id: 1, bbox: RBBox::ltwh(55.0, 52.0, 60.0, 60.0).unwrap() },
        ])]),
    },
];
let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
let out = tracker.track_sync(&frames, ids)?;
// Both frames tracked under the same source — IDs are stable across the temporal batch
```

Both frames are tracked under the same source — IDs are stable across the temporal batch.

## Pattern D: Mixed batch (multi-source + multi-frame, heterogeneous resolutions)

Four frames: cam-a×2 (320×240) + cam-b×2 (640×480). This is the most complex pattern and the primary production use case.

```rust
let frames = vec![
    TrackedFrame {
        source: "cam-a".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![Roi { id: 1, bbox: RBBox::ltwh(40.0, 40.0, 50.0, 50.0).unwrap() }])]),
    },
    TrackedFrame {
        source: "cam-a".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![Roi { id: 1, bbox: RBBox::ltwh(45.0, 45.0, 50.0, 50.0).unwrap() }])]),
    },
    TrackedFrame {
        source: "cam-b".to_string(),
        buffer: make_buffer(640, 480),
        rois: HashMap::from([(0, vec![Roi { id: 1, bbox: RBBox::ltwh(200.0, 200.0, 100.0, 100.0).unwrap() }])]),
    },
    TrackedFrame {
        source: "cam-b".to_string(),
        buffer: make_buffer(640, 480),
        rois: HashMap::from([(0, vec![Roi { id: 1, bbox: RBBox::ltwh(210.0, 210.0, 100.0, 100.0).unwrap() }])]),
    },
];
let ids = (1..=4).map(|i| SavantIdMetaKind::Frame(i)).collect();
let out = tracker.track_sync(&frames, ids)?;
assert_eq!(out.current_tracks.len(), 4);
```

Frame order matters: it determines the surface slot order in the `NonUniformBatch` built internally. Frames for the same source must have the same resolution.

## Pattern E: Class-ID tracking

Detections grouped by `class_id` are propagated to the tracker. Each `NvDsObjectMeta` carries the specified `class_id`.

```rust
let frame = TrackedFrame {
    source: "cam-1".to_string(),
    buffer: make_buffer(320, 240),
    rois: HashMap::from([
        (0, vec![Roi { id: 1, bbox: RBBox::ltwh(40.0, 40.0, 80.0, 60.0).unwrap() }]),
        (1, vec![Roi { id: 2, bbox: RBBox::ltwh(180.0, 100.0, 70.0, 70.0).unwrap() }]),
    ]),
};
let out = tracker.track_sync(&[frame], vec![SavantIdMetaKind::Frame(1)])?;
let class_ids: HashSet<i32> = out.current_tracks.iter().map(|t| t.class_id).collect();
assert_eq!(class_ids, HashSet::from([0, 1]));
```

## Stream reset

Reset clears the tracker's internal state for one stream. After reset (with `TrackingIdResetMode::OnStreamReset`), the same object receives a new track ID.

```rust
// Assumes tracker created with TrackingIdResetMode::OnStreamReset

let roi = Roi { id: 1, bbox: RBBox::ltwh(60.0, 60.0, 90.0, 90.0).unwrap() };
let frame = || TrackedFrame {
    source: "cam-1".to_string(),
    buffer: make_buffer(320, 240),
    rois: HashMap::from([(0, vec![roi.clone()])]),
};
let ids = || vec![SavantIdMetaKind::Frame(1)];

let out0 = tracker.track_sync(&[frame()], ids())?;
let id_before = out0.current_tracks[0].object_id;

// Confirm stability
let out1 = tracker.track_sync(&[frame()], ids())?;
assert_eq!(out1.current_tracks[0].object_id, id_before);

// Reset
tracker.reset_stream("cam-1")?;

// New ID assigned
let out2 = tracker.track_sync(&[frame()], ids())?;
assert_ne!(out2.current_tracks[0].object_id, id_before);
```

In a multi-source scenario, resetting one stream does not affect other streams.

## Output consumption

```rust
let output: TrackerOutput = tracker.track_sync(&frames, ids)?;

// Current tracked objects
for obj in &output.current_tracks {
    println!("source={} id={} class={} bbox=({},{} {}x{}) conf={:.2} tracker_conf={:.2}",
        obj.source_id, obj.object_id, obj.class_id,
        obj.bbox_left, obj.bbox_top, obj.bbox_width, obj.bbox_height,
        obj.confidence, obj.tracker_confidence);
}

// Shadow / terminated / past-frame (NvDCF/DeepSORT only; empty with IOU tracker)
for s in &output.shadow_tracks {
    println!("shadow: obj {} from {}", s.object_id, s.source_id);
}
for t in &output.terminated_tracks {
    println!("terminated: obj {} from {}", t.object_id, t.source_id);
}

// Output buffer (still owns the NVMM data)
let _buf: SharedBuffer = output.buffer;
```

## Error handling

All errors are `NvTrackerError` with structured, self-explaining messages. No panics in production code.

```rust
match tracker.track_sync(&frames, ids) {
    Ok(output) => { /* process */ }
    Err(NvTrackerError::TrackSyncTimeout { timeout_secs, pts_key }) => {
        log::error!("Timeout after {}s (pts={})", timeout_secs, pts_key);
    }
    Err(NvTrackerError::ResolutionMismatch { source_id, slot_a, w_a, h_a, slot_b, w_b, h_b }) => {
        log::error!("{}: slot {} is {}x{} but slot {} is {}x{}",
            source_id, slot_a, w_a, h_a, slot_b, w_b, h_b);
    }
    Err(NvTrackerError::BufferOwnership { operation }) => {
        log::error!("Buffer still borrowed during {}", operation);
    }
    Err(e) => log::error!("Tracker error: {}", e),
}
```

## Shutdown

Always shut down the tracker when done. `shutdown()` is idempotent and also called automatically in `Drop`.

```rust
tracker.shutdown()?;
```

## IOU tracker limitations (DS 7.1)

- Object IDs start from **0** (not 1). Use `Option<u64>` if you need a sentinel.
- Shadow, terminated, and past-frame misc lists are **not** populated by the IOU tracker. Use NvDCF or DeepSORT for these features.
- `probationAge: 0` in the IOU config means objects are immediately confirmed (no tentative state).
