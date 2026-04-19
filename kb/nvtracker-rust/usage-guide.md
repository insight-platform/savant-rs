# NvTracker — Rust Usage Guide

This guide reflects the current channel-based API (`submit` + `recv*`), not the removed `track_sync` style.

## Imports

```rust
use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, VideoFormat,
};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, NvTrackerOutput, Roi, TrackedFrame,
    TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::time::Duration;
```

## Configure and create tracker

```rust
let mut config = NvTrackerConfig::new(
    default_ll_lib_path(),
    "/path/to/config_tracker_IOU.yml",
);
config.tracker_width = 320;
config.tracker_height = 240;
config.max_batch_size = 4;
config.tracking_id_reset_mode = TrackingIdResetMode::OnStreamReset;
config.operation_timeout = Duration::from_secs(30);

let tracker = NvTracker::new(config)?;
```

## Build input frames

```rust
fn make_buffer(w: u32, h: u32) -> SharedBuffer {
    BufferGenerator::new(
        VideoFormat::RGBA,
        w,
        h,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )?
    .acquire(None)
}
```

```rust
let frame = TrackedFrame {
    source: "cam-1".to_string(),
    buffer: make_buffer(320, 240)?,
    rois: HashMap::from([(0, vec![
        Roi { id: 1, bbox: RBBox::ltwh(40.0, 40.0, 80.0, 60.0)? },
        Roi { id: 2, bbox: RBBox::ltwh(180.0, 100.0, 70.0, 70.0)? },
    ])]),
};
```

## Submit and receive

```rust
tracker.submit(&[frame], vec![SavantIdMetaKind::Frame(1)])?;

match tracker.recv_timeout(Duration::from_secs(5))? {
    Some(NvTrackerOutput::Tracking(output)) => {
        for t in &output.current_tracks {
            println!("{} -> object {}", t.source_id, t.object_id);
        }
    }
    Some(NvTrackerOutput::Eos { source_id }) => {
        println!("source EOS: {source_id}");
    }
    Some(NvTrackerOutput::Error(e)) => {
        eprintln!("tracker error: {e}");
    }
    Some(NvTrackerOutput::Event(_)) => {}
    None => {
        eprintln!("timed out waiting for tracker output");
    }
}
```

## Stream reset and EOS

```rust
tracker.reset_stream("cam-1")?;
tracker.send_eos("cam-1")?;
```

## Shutdown

```rust
tracker.graceful_shutdown(Duration::from_secs(2))?;
// or tracker.shutdown()?;
```

## Notes

- `submit` consumes per-frame buffers internally; ensure no outstanding `SurfaceView` borrows exist.
- Frames in one `submit` call become one internal batch; order defines slot order.
- Frames from the same `source` must have consistent resolution or `ResolutionMismatch` is returned.
