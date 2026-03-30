# savant_core — Patterns & Idioms

## Arc-backed Primitives

`RBBox` and `VideoFrameProxy` are reference-counted (`Arc`). This means:

```rust
let a = RBBox::new(10.0, 20.0, 30.0, 40.0, None);
let b = a.clone(); // thin clone — shares data
a.set_xc(99.0);
assert_eq!(b.get_xc(), 99.0); // mutation visible through b

let c = a.copy(); // deep copy — independent
a.set_xc(0.0);
assert_eq!(c.get_xc(), 99.0); // c is independent
```

Same principle applies to `VideoFrameProxy`:
```rust
let frame = VideoFrameProxy::new(...)?;
let alias = frame.clone(); // shares internal RwLock
let independent = frame.smart_copy(); // deep copy
```

## Creating Frames

```rust
use savant_core::primitives::frame::{VideoFrameProxy, VideoFrameContent, VideoFrameTranscodingMethod};

let frame = VideoFrameProxy::new(
    "cam-1",                           // source_id
    "30/1",                            // framerate
    1920, 1080,                        // width, height
    VideoFrameContent::None,           // no pixel data
    VideoFrameTranscodingMethod::Copy, // transcoding method
    None,                              // codec
    Some(true),                        // keyframe
    (1, 30),                            // time_base (num, den)
    Some(0),                           // pts
    None, None,                        // dts, duration
)?;
```

## Adding Objects to Frames

```rust
let obj = frame.create_object(
    "detector",                         // namespace
    "person",                           // label
    None,                               // parent_id
    RBBox::new(100.0, 200.0, 50.0, 80.0, None), // detection_box
    Some(0.95),                         // confidence
    None,                               // track_id
    None,                               // track_box
    vec![],                             // attributes
)?;
```

## Querying Objects

```rust
use savant_core::match_query::*;

// Find all "person" objects with confidence > 0.5
let query = MatchQuery::And(vec![
    MatchQuery::Label(StringExpression::EQ("person".into())),
    MatchQuery::Confidence(FloatExpression::GT(0.5)),
]);

let matches = frame.access_objects(&query);
```

Macro shorthand:
```rust
let query = and!(
    MatchQuery::Namespace(eq("detector")),
    MatchQuery::Confidence(gt(0.8)),
);
```

## RBBox Coordinate Systems

```rust
// Center-based (always works, even rotated)
let bbox = RBBox::new(100.0, 200.0, 50.0, 80.0, Some(45.0));
let (xc, yc, w, h) = bbox.as_xcycwh(); // always succeeds

// Left-top-width-height (fails if rotated)
let bbox = RBBox::ltwh(10.0, 20.0, 50.0, 80.0)?;
let (l, t, w, h) = bbox.as_ltwh()?; // Ok for angle=None/0

// Convert rotated → axis-aligned envelope
let wrapping = bbox.get_wrapping_bbox(); // always axis-aligned
let (l, t, w, h) = wrapping.as_ltwh()?; // now safe

// Clamp to frame boundaries
let visual = bbox.get_visual_box(
    &PaddingDraw::default_padding(), // zero padding
    0,                                // border_width
    1920.0,                          // max_x
    1080.0,                          // max_y
)?;
```

## Message Serialization Round-trip

```rust
use savant_core::message::{Message, save_message, load_message};

let msg = Message::video_frame(&frame);
let bytes = save_message(&msg)?;
let restored = load_message(&bytes);
assert!(restored.is_video_frame());
```

## Pipeline Usage

```rust
use savant_core::pipeline::*;

let config = PipelineConfigurationBuilder::default().build()?;
let pipeline = Pipeline::new(
    "my-pipeline",
    vec![
        ("input".into(), PipelineStagePayloadType::Frame, None, None),
        ("process".into(), PipelineStagePayloadType::Batch, None, None),
        ("output".into(), PipelineStagePayloadType::Frame, None, None),
    ],
    config,
)?;

// Add frame
let id = pipeline.add_frame("input", frame)?;

// Move to next stage (move_as_is takes Vec<i64>, not &[i64])
pipeline.move_as_is("process", vec![id])?;

// Pack into batch (move_and_pack_frames takes Vec<i64>)
let batch_id = pipeline.move_and_pack_frames("process", vec![id])?;
```

## Webserver Status (async)

`get_status()` is an async function — must be called from an async context:
```rust
use savant_core::webserver::{get_status, PipelineStatus};

async fn check_status() {
    let status: PipelineStatus = get_status().await;
    // ...
}
```

## Testing Conventions

1. Unit tests next to the code:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() { ... }
}
```

2. Integration tests in `tests/` at crate root.

3. Tests needing the async runtime:
```rust
#[tokio::test]
async fn test_async_thing() { ... }
```

4. Tests needing serial execution (shared global state):
```rust
#[test]
#[serial_test::serial]
fn test_with_global_state() { ... }
```

## Benchmarking

```rust
// benches/my_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_something(c: &mut Criterion) {
    c.bench_function("name", |b| {
        b.iter(|| { /* ... */ });
    });
}

criterion_group!(benches, bench_something);
criterion_main!(benches);
```

Run: `cargo bench -p savant_core`
