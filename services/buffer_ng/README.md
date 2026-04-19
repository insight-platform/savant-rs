# savant-buffer-ng

`savant-buffer-ng` is Savant's disk buffer and persistent queue service for handling backpressure between pipeline stages. It stores incoming ZeroMQ messages in RocksDB, keeps delivery moving through a separate egress worker, and can invoke Python hooks either after receive or before send.

## What it does

The library half exposes `run_service_loop()`, `ServiceConfiguration`, `PersistentQueueWithCapacity`, `MessageWriter`, and `MessageHandler` for services that need an on-disk queue with configurable capacity and full-threshold tracking. The binary initializes optional Python message handlers from `common.message_handler_init`, starts the telemetry webserver, applies extra Prometheus/OpenTelemetry labels, and then drains ingress to RocksDB while a background thread replays queued messages downstream. This is the service you place in front of slower or intermittently unavailable consumers when you want durable buffering instead of in-memory drops.

## Install / run

```sh
cargo install savant-buffer-ng
# or: docker pull ghcr.io/insight-platform/savant-buffer-ng-x86:latest
buffer_ng path/to/configuration.json
```

## Configuration

See `services/buffer_ng/assets/configuration.json` for the shipped sample.

- `ingress`: source socket to read Savant messages from.
- `egress`: sink socket used to forward buffered messages.
- `common.message_handler_init`: optional Python module/function and invocation context for custom processing.
- `common.telemetry`: management port, stats interval, and extra metric labels.
- `common.buffer`: RocksDB path, max queue length, full-threshold percentage, and reset-on-start behavior.
- `common.idle_sleep`: loop backoff when ingress is idle.

## Documentation

- [Full service docs](https://insight-platform.github.io/savant-rs/services/buffer_ng/index.html)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/services/buffer_ng)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
