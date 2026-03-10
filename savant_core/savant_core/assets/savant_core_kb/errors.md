# savant_core — Error Handling

## General Convention

savant_core uses **`anyhow::Result<T>`** for most public functions. This allows
callers to propagate errors with context via `.context("description")`.

Typed error enums (via `thiserror`) are used in specific modules where callers
need to match on error variants.

## Common Error Sources

### RBBox

| Operation | Error Condition |
|-----------|----------------|
| `as_ltwh()` / `as_ltrb()` | Box is rotated (angle is Some and != 0) |
| `get_top/left/right/bottom()` | Box is rotated |
| `set_top/left()` | Box is rotated |
| `iou/ios/ioo()` | Polygon construction fails (degenerate box) |
| `inside()` | Polygon construction fails |
| `get_visual_box()` | Width or height ≤ 0 after padding/clamping |
| `ltwh()` / `ltrb()` | Width or height ≤ 0 |

### VideoFrameProxy

| Operation | Error Condition |
|-----------|----------------|
| `new()` | Invalid framerate format (must be "num/den") |
| `add_object(obj, Error)` | ID collision with existing object |
| `set_parent_by_id()` | Object or parent not found; cycle detected |
| `export_complete_object_trees()` | Inconsistent parent chain |
| `update()` | Update references nonexistent objects |

### Pipeline

| Operation | Error Condition |
|-----------|----------------|
| `new()` | Duplicate stage names; empty stages |
| `add_frame(stage)` | Stage not found; stage type mismatch |
| `move_as_is(dest, ids)` | Dest stage not found; payload type mismatch |
| `move_and_pack_frames()` | Dest stage not Batch type |
| `move_and_unpack_batch()` | Dest stage not Frame type |
| `delete(id)` | ID not found |
| `access_objects(id)` | Frame ID not found |

### Message Serialization

| Operation | Error Condition |
|-----------|----------------|
| `save_message()` | Protobuf encoding failure |
| `load_message()` | Protobuf decode failure; version mismatch |

### ZeroMQ Transport

| Operation | Error Condition |
|-----------|----------------|
| `Reader::new() / Writer::new()` | Socket bind/connect failure |
| `Reader::receive()` | Timeout, topic prefix mismatch, blacklisted source |
| `Writer::send()` | Send timeout, ack timeout |
| `parse_zmq_socket_uri()` | Malformed URI |

### Symbol Mapper

| Operation | Error Condition |
|-----------|----------------|
| `register_model_objects(_, _, ErrorIfNonUnique)` | Duplicate model or object name |

## Error Propagation Pattern

```rust
use anyhow::{Context, Result};

fn process_frame(frame: &VideoFrameProxy) -> Result<()> {
    let bbox = frame
        .get_object(42)
        .context("object 42 not found")?
        .get_detection_box();

    let (l, t, w, h) = bbox
        .as_ltwh()
        .context("cannot convert rotated bbox to LTWH")?;

    Ok(())
}
```

## Panic Policy

savant_core avoids panics. Known exceptions:
- `Affine2D::inverse()` — panics on degenerate (zero scale). Callers must check.
- `BBOX_UNDEFINED` — lazy_static initialization (infallible).
- Assertions in test code.
