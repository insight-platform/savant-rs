# savant_core — Dependencies

## Core Dependencies

| Crate | Usage |
|-------|-------|
| `anyhow` | Error handling: `Result<T>` = `anyhow::Result<T>` throughout |
| `thiserror` | Typed error enums (pipeline, protobuf) |
| `serde` / `serde_json` / `serde_yaml` | JSON/YAML serialization for MatchQuery, Attribute, config files |
| `prost` / `tonic` | Protobuf encoding/decoding (via `savant_protobuf`) |
| `parking_lot` | Fast RwLock/Mutex (used in VideoFrameProxy, Pipeline, SymbolMapper) |
| `hashbrown` | HashMap/HashSet (faster than std) |
| `lazy_static` | Global singletons (BBOX_UNDEFINED, async runtime, etc.) |
| `log` | Logging facade |
| `uuid` | UUID v4/v7 generation (frame IDs) |
| `rand` | Random number generation |

## Concurrency

| Crate | Usage |
|-------|-------|
| `tokio` | Async runtime (shared via `get_or_init_async_runtime()`) |
| `crossbeam` | Lock-free data structures (used in pipeline internals) |
| `rayon` | Parallel iterators (batch operations) |

## Geometry & Math

| Crate | Usage |
|-------|-------|
| `geo` | Polygon intersection/union for IoU/IoS/IoO calculations |
| `crc32fast` | `fast_hash()` — CRC32 for frame compatibility hashing |

## Query & Evaluation

| Crate | Usage |
|-------|-------|
| `evalexpr` | Expression evaluation in `MatchQuery::EvalExpr` |
| `jmespath` | JMESPath queries on attribute JSON (`AttributesJMESQuery`) |
| `globset` | Glob patterns in MatchQuery string expressions |
| `nom` | Parser combinators (label filter parser) |
| `regex` | Regular expressions in string matching |

## Networking & HTTP

| Crate | Usage |
|-------|-------|
| `zmq` | ZeroMQ transport (blocking sockets) |
| `actix-web` / `actix-ws` | Embedded HTTP server + WebSocket (webserver module) |
| `reqwest` | HTTP client (telemetry, etcd) |
| `futures-util` | Stream utilities for async operations |

## Telemetry

| Crate | Usage |
|-------|-------|
| `opentelemetry` / `opentelemetry_sdk` | Tracing API and SDK |
| `opentelemetry-otlp` | OTLP exporter |
| `opentelemetry-jaeger-propagator` | Jaeger context propagation |
| `opentelemetry-stdout` | Debug exporter |
| `opentelemetry-semantic-conventions` | Standard span attributes |

## Metrics

| Crate | Usage |
|-------|-------|
| `prometheus-client` | Prometheus metric families (Counter, Gauge) |

## Configuration

| Crate | Usage |
|-------|-------|
| `twelf` | Layered config loading (env + file + defaults) |
| `derive_builder` | Builder pattern for PipelineConfiguration, VideoFrame, etc. |

## Other

| Crate | Usage |
|-------|-------|
| `lru` | LRU cache (eval_cache, SeqStore) |
| `moka` | Concurrent cache |
| `nix` | Unix signals |
| `libloading` | Dynamic library loading (pipeline stage function plugins) |
| `savant_protobuf` | Generated protobuf types (workspace sibling) |
| `savant_etcd` | etcd client wrapper (workspace sibling) |
| `etcd-client` | etcd gRPC client |

## Dev Dependencies

| Crate | Usage |
|-------|-------|
| `criterion` | Benchmarks |
| `serial_test` | Serial test execution (shared global state) |
| `tempfile` | Temporary files in tests |
| `bollard` | Docker API (container tests) |
