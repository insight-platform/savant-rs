# NvTracker Python KB — Agent Guidelines

## Purpose

Reference for agents writing Python tests or samples against `savant_rs.nvtracker`
without spelunking the Rust sources. Manual multi-stream tracking is the first-class
production design (no `nvstreammux`).

## Files

| File | Content |
|------|---------|
| `api.md` | Classes, constructors, methods |
| `deps.md` | Imports from `deepstream`, `nvinfer`, geometry |
| `patterns.md` | Build/test commands, test-to-pattern mapping |
| `enums.md` | `TrackingIdResetMode`, `TrackState` |
| `errors.md` | Runtime failures |
| `usage-guide.md` | Code samples for all batch patterns: single-source, multi-source, temporal, mixed, heterogeneous, stream reset |

## Usage

1. `usage-guide.md` — **read first** — concrete code samples for every pattern
2. `api.md` — all Python-facing types
3. `deps.md` — `SharedBuffer`, `VideoFormat`, `Roi`
4. `patterns.md` — Makefile, pytest
5. `enums.md` / `errors.md` as needed

## Critical pitfalls

- **Native submodule** — `savant_rs.nvtracker` is registered from the extension module; `nvtracker/__init__.py` is not executed at import time for symbols. Stubs: `savant_rs/nvtracker/nvtracker.pyi`.
- **Feature gate** — Module exists only when the wheel is built with **`deepstream`** (`SAVANT_FEATURES=deepstream make dev install`).
- **GIL** — Async callback runs on a GStreamer thread with GIL attached (`Python::attach`); keep callbacks light.
- **`TrackedFrame`** — The `SharedBuffer` is consumed when constructing a `TrackedFrame`. Do not reuse the buffer.
- **`track` / `track_sync`** — Accept `List[TrackedFrame]` and `List[Tuple[SavantIdMetaKind, int]]`. The tracker builds the internal `NonUniformBatch` automatically.

## Build & test

From repo root:

```bash
SAVANT_FEATURES=deepstream make dev install
SAVANT_FEATURES=deepstream make sp-pytest
```

## Conventions

Same as nvinfer Python KB: `SIG:`, `DEF:`, `REQ:`, `OPT:`, `⚠`, `GPU`.
