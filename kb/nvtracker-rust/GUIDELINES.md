# NvTracker Rust KB — Agent Guidelines

## Purpose

Self-contained reference for agents to write Rust code/tests for the `nvtracker`
crate without reading the whole source tree. Manual multi-stream tracking is the
first-class production design (no `nvstreammux`).

## Files

| File | Content |
|------|---------|
| `arch.md` | Pipeline layout, mixed batch model, DeepStream contract invariants, mux reference delta table, CUDA cleanup |
| `api.md` | Public API: `NvTracker`, `NvTrackerConfig`, `TrackedFrame`, `TrackerOutput`, `Roi`, `SavantIdMetaKind`, `attach_detection_meta`, `extract_tracker_output` |
| `patterns.md` | Tests, buffer setup, IOU YAML asset, test-to-pattern mapping |
| `errors.md` | `NvTrackerError` variants |
| `usage-guide.md` | Code samples for all batch patterns: single-source, multi-source, temporal, mixed, heterogeneous, stream reset |

## Usage order

1. `arch.md` — **read first** — pipeline topology, mixed batch model, frame numbering, DeepStream invariants
2. `usage-guide.md` — concrete code samples for every pattern
3. `api.md` — types and method signatures
4. `patterns.md` — integration tests, GPU helpers
5. `errors.md` — error mapping

## Conventions

- `SIG:` = function/method signature
- `DEF:` = default value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = CUDA + GStreamer + DeepStream + `nvtracker` plugin
