# savant-core-py

`savant-core-py` is the PyO3 layer for `savant-core`: it exposes Savant video metadata, bounding box geometry, ZeroMQ transport, protobuf serialization, pipeline statistics, and OpenTelemetry helpers to Python. It is the binding crate re-exported by the `savant-rs` wheel, so Python users typically meet these APIs as `import savant_rs`.

## What's inside

- Root helpers: `version()`, `is_release_build()`, and the process-wide Python handler registry via `register_handler()`, `unregister_handler()`, and `clear_all_handlers()`.
- `primitives`: core Python bindings for `VideoFrame`, `VideoFrameBatch`, `VideoObject`, `BorrowedVideoObject`, `VideoObjectTree`, `Attribute`, `AttributeValue`, `EndOfStream`, `Shutdown`, `UserData`, and misc tracker types such as `TrackUpdate` and `MiscTrackData`.
- `primitives.geometry`: `RBBox`, `BBox`, `Point`, `Segment`, `PolygonalArea`, `Intersection`, plus geometry helpers such as `associate_bboxes()` for bounding box and area matching.
- `pipeline`: `PipelineConfiguration`, `Pipeline`, `StageFunction`, `StageLatencyStat`, `StageProcessingStat`, and frame-processing statistics for pipeline observability.
- `utils` and `symbol_mapper`: expression evaluation, UUID helpers, `ByteBuffer`, `AtomicCounter`, model/object symbol mapping, and protobuf message load/save helpers.
- `zmq`: `ReaderConfig`, `WriterConfig`, blocking and non-blocking ZMQ reader/writer classes, and typed result objects for Savant ZeroMQ transport.
- `telemetry`, `metrics`, and `webserver`: OpenTelemetry initialization, Prometheus metric families, service status endpoints, and attribute KVS helpers.
- Optional `gst` feature: `gstreamer` and `retina_rtsp` bindings such as `Mp4Muxer`, `Mp4Demuxer`, `VideoInfo`, `RtspSourceGroup`, and `RetinaRtspService`.
- Optional `deepstream` feature: `deepstream`, `nvinfer`, `nvtracker`, and `picasso` bindings including `SharedBuffer`, `SurfaceView`, `FlexibleDecoder`, `NvInfer`, `NvInferBatchingOperator`, `NvTracker`, and GPU-oriented encoder/decoder configuration types.

## Usage

```python
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.primitives.geometry import RBBox

frame = VideoFrame(
    "camera-1",
    (30, 1),
    1920,
    1080,
    VideoFrameContent.none(),
)
obj = frame.create_object("detector", "person", detection_box=RBBox(960.0, 540.0, 200.0, 400.0))
print(frame.uuid, obj.label)
```

## Install

```toml
[dependencies]
savant-core-py = "2"
```

## System requirements

Building this crate requires a working Python toolchain for PyO3. Enabling `gst` additionally needs GStreamer development libraries, and enabling `deepstream` requires the NVIDIA DeepStream SDK plus the GPU runtime dependencies used by the DeepStream-backed modules.

## Documentation

- [docs.rs](https://docs.rs/savant-core-py)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).