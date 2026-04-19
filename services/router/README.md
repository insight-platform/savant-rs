# savant-router

`savant-router` is a Python-extensible message routing service for Savant streams. It consumes one or more ZeroMQ ingress sockets, evaluates label-based and conditional routing rules, and forwards messages to multiple egress sockets with optional topic and source remapping.

## What it does

The service loads `ServiceConfiguration`, initializes a Python module declared in `common.init`, and then connects `Ingress` readers to `Egress` processors in a tight routing loop. Each ingress can declare a named handler, while each egress can apply a `matcher`, `source_mapper`, `topic_mapper`, and `high_watermark` so routing decisions can depend on message labels, custom Python logic, and downstream backpressure thresholds. This makes the router a natural fan-out hub between pipeline stages, model branches, and post-processing services.

## Install / run

```sh
cargo install savant-router
# or: docker pull ghcr.io/insight-platform/savant-router-x86:latest
router path/to/configuration.json
```

## Configuration

See `services/router/assets/configuration.json` for a complete sample.

- `ingress`: a list of named source sockets with optional per-ingress handler names.
- `egress`: a list of named sink sockets with `matcher`, `source_mapper`, `topic_mapper`, and `high_watermark` controls.
- `common`: Python module initialization, name-cache settings, and idle sleep tuning.

## Documentation

- [Full service docs](https://insight-platform.github.io/savant-rs/services/router/index.html)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/services/router)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
