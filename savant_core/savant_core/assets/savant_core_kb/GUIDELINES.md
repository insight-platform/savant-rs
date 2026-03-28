# savant_core KB — Agent Guidelines

## Purpose
Self-contained reference for agents working on the `savant_core` Rust crate — the
foundational library powering Savant's video-analytics pipeline.

## Files
| File | Content |
|---|---|
| `api.md` | Public API: key structs, enums, traits, functions per module |
| `arch.md` | Module tree, architecture overview, threading model |
| `deps.md` | External dependencies and what they are used for |
| `patterns.md` | Common patterns, idioms, testing tips |
| `errors.md` | Error handling conventions, `anyhow` vs typed errors |

## Usage
1. Read `arch.md` first to understand module layout and data flow.
2. Read `api.md` for the specific module you are modifying.
3. Read `deps.md` to know which external crates are available.
4. Read `patterns.md` for coding idioms, testing, and benchmarking.
5. Consult `errors.md` for error handling conventions.

## Critical Pitfalls (read before writing code)
- **`RBBox` is `Arc`-backed** — `Clone` is thin (shared data). Use `.copy()` for
  deep copies. Setters (`set_xc`, etc.) mutate through `AtomicF32` and mark
  `has_modifications = true`.
- **`VideoFrameProxy` uses interior mutability** — wraps
  `Arc<RwLock<Box<VideoFrame>>>`. Clone shares state. Use `smart_copy()` for a
  deep, independent copy.
- **`Message` serialization** — `save_message` / `load_message` go through
  protobuf. Version is checked on deserialization; mismatches produce errors.
- **Pipeline payload lifetimes** — frames added to a `Pipeline` stage are owned
  by the pipeline. Extracting them (`get_independent_frame`,
  `move_and_unpack_batch`) transfers ownership back.
- **ZeroMQ sockets** — `Reader` / `Writer` are not `Send`; the `NonBlocking*`
  variants use internal `tokio` tasks and are `Send + Sync`.
- **Telemetry** — must call `telemetry::init(...)` or `init_from_file(...)` early.
  `shutdown()` flushes spans; omitting it may lose data.
- **MatchQuery** — the DSL supports short-circuit (`StopIfFalse`, `StopIfTrue`)
  via `ControlFlow`. Combinators `And` / `Or` respect this.

## Conventions
- `pub mod rust { ... }` blocks re-export types for direct Rust consumers
  (as opposed to the Python binding layer in `savant_core_py`).
  Current re-exports include:
  - `otlp::PropagatedContext`
  - `pipeline::stats::{FrameProcessingStatRecord, FrameProcessingStatRecordType,
     StageLatencyMeasurements, StageLatencyStat, StageProcessingStat}`
  - `pipeline::{Pipeline, PipelineConfiguration, PipelineConfigurationBuilder,
     PipelineStagePayloadType}`
  - `symbol_mapper::{RegistrationPolicy, SymbolMapper}`
- `EPS = 0.00001` is the crate-wide epsilon for floating-point comparisons.
- `round_2_digits(v)` rounds to two decimal places (used in bbox serialization).
- Unit tests live alongside their module (`#[cfg(test)] mod tests { ... }`).
- Integration tests live in `tests/` at the crate root.
- Benchmarks use `criterion` and live in `benches/`.

## Building and Testing

```bash
# Run all tests
cargo test -p savant_core

# Run a specific test
cargo test -p savant_core -- test_name

# Run benchmarks
cargo bench -p savant_core

# Clippy
cargo clippy -p savant_core -- -D warnings

# Format
cargo fmt -p savant_core
```
