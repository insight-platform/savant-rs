# savant-protobuf

`savant-protobuf` provides the proto3 schema and `prost`-generated Rust types that Savant uses for IPC between real-time video-analytics pipeline stages. The crate models video frames, detected objects, attributes, frame updates, batching, and control messages in a serialization format shared across Savant messaging components.

## What's inside

- The `generated` module contains the complete protobuf schema as Rust types, including geometry and attribute primitives such as `BoundingBox`, `Point`, `PolygonalArea`, `Intersection`, `AttributeValue`, `Attribute`, and `AttributeSet`.
- Frame and object payloads are represented by `VideoObject`, `VideoObjectWithForeignParent`, `VideoFrame`, `VideoFrameBatch`, `VideoFrameUpdate`, and `UserData`, which are the core schema types passed through an analytics pipeline.
- The top-level `Message` envelope wraps wire-level payloads with protocol metadata such as `protocol_version`, `routing_labels`, `propagated_context`, `seq_id`, and `system_id`. Its `message::Content` oneof carries `VideoFrame`, `VideoFrameBatch`, `VideoFrameUpdate`, `UserData`, `EndOfStream`, `Unknown`, and `Shutdown`.
- Supporting enums such as `VideoCodec`, `VideoFrameTranscodingMethod`, `AttributeUpdatePolicy`, `ObjectUpdatePolicy`, `TrackState`, and `MiscTrackCategory` make the schema explicit and stable across Rust services and bindings.
- `version()` returns the current crate version and is typically used to stamp outgoing Savant messages with the matching protocol version.

## Usage

```rust
use prost::Message as _;
use savant_protobuf::generated::{self, message, video_frame, Message, Rational32, VideoFrame};

fn main() {
    let frame = VideoFrame {
        source_id: "camera-1".into(),
        uuid: "frame-uuid".into(),
        width: 1920,
        height: 1080,
        fps: Some(Rational32 {
            numerator: 30,
            denominator: 1,
        }),
        time_base: Some(Rational32 {
            numerator: 1,
            denominator: 1_000_000,
        }),
        transcoding_method: generated::VideoFrameTranscodingMethod::Copy as i32,
        video_codec: generated::VideoCodec::H264 as i32,
        content: Some(video_frame::Content::None(generated::NoneFrame {})),
        ..Default::default()
    };

    let wire = Message {
        protocol_version: savant_protobuf::version().to_string(),
        seq_id: 1,
        system_id: "edge-node".into(),
        content: Some(message::Content::VideoFrame(frame)),
        ..Default::default()
    }
    .encode_to_vec();

    assert!(!wire.is_empty());
}
```

## Install

```toml
[dependencies]
savant-protobuf = "2"
```

## Documentation

- [docs.rs](https://docs.rs/savant-protobuf)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
