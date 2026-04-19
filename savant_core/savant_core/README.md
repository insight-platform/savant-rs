# savant-core

`savant-core` is the pure-Rust video analytics and computer vision foundation of Savant, defining frame primitives, object metadata, bounding box geometry, protobuf messaging, ZeroMQ transport, and OpenTelemetry tracing. It is the crate to reach for when you need `VideoFrame` and `VideoObject` handling, RBBox math, IoU and NMS helpers, symbol mapping, or pipeline logic without any DeepStream-specific dependencies.

## What's inside

- **Primitives and metadata:** `primitives` is the heart of `savant-core`. It defines `RBBox`, `Point`, `PolygonalArea`, `Segment`, `Intersection`, `Attribute`, `AttributeValue`, `AttributeSet`, `VideoObject`, `VideoObjectBuilder`, `VideoFrameProxy`, `VideoFrameContent`, `VideoFrameTransformation`, `VideoFrameBatch`, `VideoFrameUpdate`, `MiscTrackData`, `EndOfStream`, `Shutdown`, `UserData`, and `VideoCodec`.
- **Messaging and protobuf:** `message` provides the high-level `Message`, `MessageEnvelope`, `MessageMeta`, and `SeqStore` types used to package frames, updates, EOS, shutdown, and user data for transport. `protobuf` bridges those Rust types to `savant-protobuf` with `ToProtobuf`, `from_pb`, `serialize()`, and `deserialize()`.
- **Geometry and postprocessing:** `geometry::Affine2D` composes frame transformation chains so object coordinates can be remapped between letterboxed, padded, cropped, and original spaces. `converters::nms` exposes `iou_xcycwh`, `nms_class_agnostic`, and `nms_class_aware`, while `converters::yolo::YoloDetectionConverter` turns raw YOLO tensors into structured detections.
- **Pipeline execution and querying:** `pipeline` defines `Pipeline`, `PipelineConfiguration`, `PipelineStageFunction`, `PipelinePayload`, and stage statistics for multi-stage analytics flows. `match_query::MatchQuery` and its helper expressions let you filter `VideoObject` collections by IDs, labels, attributes, bounding boxes, frame fields, and eval-based predicates.
- **Integration layers:** `transport::zeromq` includes `ReaderConfig`, `WriterConfig`, `SyncReader`, `SyncWriter`, `NonBlockingReader`, `NonBlockingWriter`, `ReaderResult`, `WriterResult`, and `TopicPrefixSpec` for Savant ZMQ reader/writer patterns. `symbol_mapper::SymbolMapper` and `RegistrationPolicy` keep stable numeric IDs for models and labels. `telemetry::TelemetryConfiguration` and `Configurator` set up OTLP exporters and context propagation, while `utils::rtp_pts_mapper::RtpPtsMapper`, `metrics`, and `webserver` support timing, observability, and control-plane APIs.

## Usage

```rust
use savant_core::primitives::bbox::RBBox;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::protobuf::ToProtobuf;

fn main() -> anyhow::Result<()> {
    let frame = VideoFrameProxy::new(
        "camera-1",
        (30, 1),
        1920,
        1080,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        None,
        Some(true),
        (1, 1_000_000),
        0,
        None,
        None,
    )?;

    let object = VideoObjectBuilder::default()
        .id(0)
        .namespace("detector".to_string())
        .label("person".to_string())
        .detection_box(RBBox::new(960.0, 540.0, 120.0, 240.0, None))
        .confidence(Some(0.98))
        .build()?;

    let person = frame.add_object(object, IdCollisionResolutionPolicy::GenerateNewId)?;
    let wire = frame.to_pb()?;

    assert_eq!(person.get_label(), "person");
    assert!(!wire.is_empty());
    Ok(())
}
```

## Install

```toml
[dependencies]
savant-core = "2"
```

## Documentation

- [docs.rs](https://docs.rs/savant-core)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
