# NvBufSurface Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code/tests for the
`deepstream_buffers` crate without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, functions, signatures |
| `arch.md` | Architecture: module tree, memory model, pool lifecycle, FFI layer |
| `patterns.md` | Test patterns, helpers, templates, common test fixtures |
| `errors.md` | NvBufSurfaceError / TransformError variants and when they trigger |
| `caveats.md` | Critical design decisions, known pitfalls, memory safety invariants |

## Usage Order
1. `caveats.md` — **read first** — the most important pitfalls and invariants
2. `api.md` — public types, constructors, method signatures
3. `arch.md` — internals: module tree, pool model, FFI, GstParentBufferMeta
4. `patterns.md` — test scaffolding, helpers, ready-to-copy templates
5. `errors.md` — error variants and how to trigger/test them

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA/NvBufSurface runtime
- `pool_size=2` = use small pools in leak smoke tests to catch leaked refs

## Cargo
Third-party crates (including optional `gl` and dev-only `env_logger`) are declared in the repo root `[workspace.dependencies]` and referenced here as `{ workspace = true }` so versions stay centralized.
