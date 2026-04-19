# savant-services-common

`savant-services-common` collects the shared building blocks used by Savant microservices: config loader-friendly socket types, async ZeroMQ wrappers, lightweight metrics helpers, and service utilities that show up in router, replay, buffer-ng, meta-merge, and retina-rtsp. It is the crate that standardizes how services describe sources, sinks, FPS counters, timestamps, and graceful data movement across tokio-based components.

## What's inside

- `source`: `SourceConfiguration`, `SourceOptions`, and `TopicPrefixSpec` describe ingress sockets, timeouts, topic prefix filters, IPC permission fixes, and reader inflight capacity.
- `job_writer`: `SinkConfiguration`, `SinkOptions`, `JobWriter`, and `job_writer::cache::JobWriterCache` wrap Savant non-blocking writers for service egress, replay jobs, and cached downstream connections.
- `fps_meter`: `FpsMeter` is a compact counter used by services to log throughput over time.
- Crate helpers: `topic_to_string()` converts binary ZeroMQ topics into readable text, and `systime_ms()` provides a simple wall-clock timestamp helper.

## Usage

```rust
use savant_services_common::job_writer::{JobWriter, SinkConfiguration};
use savant_services_common::source::SourceConfiguration;

let source = SourceConfiguration {
    url: "router+bind:tcp://0.0.0.0:5555".into(),
    options: None,
};

let sink = SinkConfiguration {
    url: "dealer+connect:tcp://127.0.0.1:5556".into(),
    options: None,
};

let _ = (source, JobWriter::new((&sink).try_into()?));
```

## Install

```toml
[dependencies]
savant-services-common = "2"
```

## System requirements

No extra system packages are required beyond the Rust toolchain and whatever transport/runtime dependencies your service already uses. The crate is designed to be embedded in async tokio services that talk to Savant ZeroMQ endpoints.

## Documentation

- [docs.rs](https://docs.rs/savant-services-common)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
