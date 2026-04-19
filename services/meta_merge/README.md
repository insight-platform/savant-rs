# savant-meta-merge

`savant-meta-merge` is a Python-extensible metadata merge service for Savant fan-in pipelines. It collects frames with the same UUID from multiple ingress sockets, synchronizes them in a bounded queue, and emits one merged frame downstream after Python callbacks decide how to combine metadata and how to handle late arrivals.

## What it does

The crate exposes `run_service_loop()`, `process_ingress_messages()`, `ProcessResult`, and `ServiceConfiguration` for applications that need UUID-based synchronization across multiple sources. Internally, `Ingress` normalizes incoming messages, `EgressProcessor` accumulates frame heads until they are ready or expired, and the configured callback set drives merge behavior for `on_merge`, `on_head_expire`, `on_head_ready`, `on_late_arrival`, optional `on_unsupported_message`, and optional `on_send`. EOS handling is configured per ingress with `allow` or `deny`, so multi-ingress shutdown behavior stays explicit.

## Install / run

```sh
cargo install savant-meta-merge
# or: docker pull ghcr.io/insight-platform/savant-meta-merge-x86:latest
meta_merge path/to/configuration.json
```

## Configuration

See `services/meta_merge/assets/configuration.json` for the shipped sample.

- `ingress`: named source sockets plus optional `handler` names and per-ingress `eos_policy`.
- `egress`: downstream sink socket.
- `common.init`: Python module bootstrap for registering handlers.
- `common.callbacks`: names of the merge, expiry, readiness, late-arrival, unsupported-message, and send callbacks.
- `common.queue`: queue timing, currently `max_duration` for head synchronization.
- `common.idle_sleep`: polling backoff when no messages arrive.

## Documentation

- [Full service docs](https://insight-platform.github.io/savant-rs/services/meta_merge/index.html)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/services/meta_merge)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
