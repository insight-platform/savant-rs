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

Frame metadata uses a typed codec and a rational **time base** `(numerator, denominator)` in **`i64`** (seconds per tick = `num/den`). GStreamer clock scale is `(1, 1_000_000_000)`.

```rust
use savant_core::primitives::frame::{VideoFrameProxy, VideoFrameContent, VideoFrameTranscodingMethod};
use savant_core::primitives::rust::VideoCodec;

let frame = VideoFrameProxy::new(
    "cam-1",                           // source_id
    (30, 1),                           // fps (numerator, denominator) — i64
    1920, 1080,                        // width, height
    VideoFrameContent::None,           // no pixel data
    VideoFrameTranscodingMethod::Copy, // transcoding method
    Some(VideoCodec::H264),            // codec: Option<VideoCodec>
    Some(true),                        // keyframe
    (1, 30),                           // time_base (num, den) — i64 tuple
    0,                                 // pts (in time_base units)
    None,                              // dts
    None,                              // duration
)?;
```

Protobuf `VideoFrame` carries **`NanosecondsU128`** for **`creation_timestamp_ns`** (field 4), **`Rational32`** for **`fps`** (field 6) and **`time_base`** (field 12), and **`VideoCodec`** enum (field 10). JSON on the Rust side uses **`fps`** / **`time_base`** as `[num, den]` arrays and codec as the canonical **string** name (`"h264"`, `"swjpeg"`, …).

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

## Wire-to-domain narrowing in protobuf deserialization

When Rust domain fields are narrower than protobuf wire fields, validate explicitly.
For `u16` Rust fields stored as `u32` on wire, use `u16::try_from(...)` and return a typed
`protobuf::serialize::Error` on failure, never `as` casts.
For protobuf enums (`i32` wire), use `try_from` plus an exhaustive `match` and map invalid
wire values to `Error::Unknown*` variants.
Do not use wildcard enum fallbacks/defaults in from-pb helpers; unknown wire values must hard-fail
at the message boundary to prevent silent data corruption.
Reference implementation: `savant_core/src/protobuf/serialize/video_frame.rs` misc-track helpers.

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

Run: `cargo bench -p savant-core`

---

## Numeric widths on public structs (i64 convention)

All logical ids, counters, and ages on savant-owned public structs use
`i64`. `u64` is reserved for values whose semantics are inherently
non-negative 64-bit (e.g. `object_id`, `unique_id`,
`surface_stream_id`). Narrow widths (`u16` / `u32` / `i32`) only exist
inside FFI-mirror structs (`deepstream-sys`, `deepstream::tracker_meta`,
DS C headers) and are widened **losslessly** at the first
savant-owned struct past the FFI seam.

Examples in the tracker/misc-track area:

| Struct | Field | Width |
|---|---|---|
| `MiscTrackData` | `class_id` | `i64` |
| `MiscTrackFrame` | `frame_num`, `age` | `i64` |
| `TrackedObject` (nvtracker) | `class_id`, `slot_number` | `i64` |
| `ElementOutput` (nvinfer) | `slot_number` | `i64` |

Protobuf messages mirror the domain: `MiscTrackData.class_id`,
`MiscTrackFrame.frame_num`, `MiscTrackFrame.age` are all `int64` on the
wire (see `savant_protobuf/src/savant_rs.proto`). This removed the
previous wire-vs-domain narrowing and its associated
`Error::MiscTrackClassIdOverflow` variant.

Why: arbitrary-precision Python ints already cross the boundary
without surprises; there is no narrowing to check on the way back; and
tracker/detector producers that emit class ids > u16 do not silently
truncate.
