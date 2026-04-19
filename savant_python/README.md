# savant-rs

`savant-rs` is the root PyO3 + maturin crate that builds the `savant_rs` Python package published as the `savant-rs` wheel. It packages Rust primitives for real-time video analytics, frame metadata, pipeline orchestration, ZeroMQ transport, and optional NVIDIA DeepStream and GStreamer integrations behind a single `pip install savant-rs` entry point.

## What's inside

- Root package functions: `version()`, `is_release_build()`, and the global handler registry via `register_handler()`, `unregister_handler()`, and `clear_all_handlers()`.
- `savant_rs.primitives`: `VideoFrame`, `VideoFrameBatch`, `VideoObject`, `BorrowedVideoObject`, `Attribute`, `AttributeValue`, `EndOfStream`, `Shutdown`, `UserData`, and `RBBox`-based geometry helpers used to manipulate frame metadata in Python.
- `savant_rs.pipeline`: `VideoPipeline`, `VideoPipelineConfiguration`, stage payload typing, and per-stage latency/throughput statistics for Savant pipeline instrumentation.
- `savant_rs.utils`: UUID helpers, expression evaluation, `ByteBuffer`, `AtomicCounter`, protobuf serialization helpers, and the symbol mapper registry for model/object IDs.
- `savant_rs.zmq`: reader/writer config builders plus blocking and non-blocking ZeroMQ clients for Savant message transport.
- `savant_rs.telemetry`, `savant_rs.metrics`, and `savant_rs.webserver`: OpenTelemetry setup, Prometheus metric families, service health/status hooks, and the KVS webserver utilities.
- Optional `gst` feature: `savant_rs.gstreamer` and `savant_rs.retina_rtsp` expose MP4 mux/demux helpers and the embeddable RTSP client service types.
- Optional `deepstream` feature: `savant_rs.deepstream`, `savant_rs.nvinfer`, `savant_rs.nvtracker`, and `savant_rs.picasso` add NVIDIA DeepStream buffers, decode/encode helpers, inference pipelines, tracking, and rendering APIs.

## Install

```sh
pip install savant-rs
```

## Usage

```python
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.primitives.geometry import RBBox

frame = VideoFrame("camera-1", (25, 1), 1280, 720, VideoFrameContent.none())
frame.create_object("detector", "car", detection_box=RBBox(640.0, 360.0, 220.0, 120.0))
print(frame.json_pretty)
```

## System requirements

The published wheel is built with maturin and PyO3. CPU-only builds need a supported Python environment; enabling the optional `gst` feature requires GStreamer development packages at build time, and enabling `deepstream` additionally requires the NVIDIA DeepStream SDK and compatible GPU runtime libraries.

## Documentation

- [Savant project site](https://insight-platform.github.io/savant-rs/)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/savant_python)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).