# savant-replay

`savant-replay` is the replay service for Savant pipelines: it archives incoming video streams in RocksDB and exposes a REST API for keyframe lookup, replay job creation, and time-synchronized or fast re-streaming. It is built on `actix-web` and `savant-replaydb`, so one configuration file wires storage, ingress, optional egress, and replay job defaults together.

## What it does

The service starts a `RocksDbService`, ingests Savant ZeroMQ traffic into an on-disk frame store, and serves management endpoints under `/api/v1`. From there you can create replay jobs (`PUT /job`), inspect running and stopped jobs, search keyframes, fetch a keyframe as multipart metadata plus payload, patch stop conditions, and shut the service down. Replay jobs are configured with `JobQuery`, `JobConfiguration`, `JobStopCondition`, and `JobOffset`, which let you choose the anchor keyframe, playback speed, routing labels, metadata-only mode, and stop criteria.

## Install / run

```sh
cargo install savant-replay
# or: docker pull ghcr.io/insight-platform/savant-replay-x86:latest
replay path/to/configuration.json
```

## Configuration

The binary loads the same JSON schema defined by `replaydb::service::configuration::ServiceConfiguration`; the shipped examples are `services/replay/replaydb/assets/rocksdb.json` and `services/replay/replaydb/assets/rocksdb_opt_out.json`.

- `common`: REST management port, stats period, writer-cache settings, job eviction TTL, and optional default sink options for dynamically created replay jobs.
- `in_stream`: Savant ZeroMQ ingress to archive.
- `out_stream`: optional static downstream egress for pass-through delivery.
- `storage`: RocksDB path, data TTL, and storage tuning such as WAL and compaction options.

## Documentation

- [Full service docs](https://insight-platform.github.io/savant-rs/services/replay/index.html)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/services/replay/replay)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
