# savant-replaydb

`savant-replaydb` is the RocksDB-backed frame store behind Savant Replay. It persists Savant Protobuf messages on disk, maintains keyframe indexes per source, applies TTL-based retention, and provides the job/runtime pieces needed to replay video streams with optional compression-friendly RocksDB settings.

## What's inside

- `service`: `ServiceConfiguration` describes the replay service runtime with `common`, `in_stream`, `out_stream`, and `storage` sections, while `rocksdb_service::RocksDbService` manages ingestion, job lifecycles, keyframe queries, and stopped-job eviction.
- `store`: the `Store` trait defines the storage contract, `JobOffset` selects replay start points by keyframe blocks or seconds, and `KeyframeRecord` returns frame metadata plus payload chunks for keyframe lookup.
- `store::rocksdb::RocksDbStore`: the concrete on-disk RocksDB implementation used for frame persistence and keyframe indexing.
- `stream_processor`: `RocksDbStreamProcessor` ingests Savant messages from ZeroMQ, stores video frames, user data, and EOS records, and optionally forwards them to a static egress.
- `job`: `RocksDbJob`, `JobConfiguration`, `JobQuery`, `JobStopCondition`, and `RoutingLabelsUpdateStrategy` drive replay sessions, time travel offsets, stop conditions, routing label rewrites, metadata-only delivery, and per-job sinks.

## Usage

```rust
use replaydb::service::configuration::ServiceConfiguration;
use replaydb::service::rocksdb_service::RocksDbService;

let conf = ServiceConfiguration::new("services/replay/replaydb/assets/rocksdb.json")?;
let mut service = RocksDbService::new(&conf)?;
let keyframes = service.find_keyframes("camera-1", None, None, 10).await?;
println!("{keyframes:?}");
```

## Install

```toml
[dependencies]
savant-replaydb = "2"
```

## System requirements

`savant-replaydb` depends on RocksDB and is intended for services that ingest Savant ZeroMQ traffic and persist frames on disk. In practice you need the Rust toolchain plus the native build prerequisites for `librocksdb-sys`; storage sizing and filesystem performance directly affect replay retention and throughput.

## Documentation

- [docs.rs](https://docs.rs/savant-replaydb)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
