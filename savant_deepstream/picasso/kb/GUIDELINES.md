# Picasso Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code/tests for the picasso crate without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, traits, methods, signatures |
| `deps.md` | External crate dependencies, re-exports, key types from deepstream_encoders/nvbufsurface/savant_core |
| `patterns.md` | Test patterns, helpers, GPU/non-GPU split, lifecycle idioms, ready-to-copy code |
| `arch.md` | Architecture: module tree, data flow, threading model, internal pipeline stages |
| `errors.md` | PicassoError variants, error conditions |

## Usage Order
1. `api.md` — picasso's own public types
2. `deps.md` — imports from external crates
3. `patterns.md` — test scaffolding, helpers, templates
4. `arch.md` — only when understanding internals or writing integration tests touching pipeline stages
5. `errors.md` — negative-path tests

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA/NvBufSurface runtime
- `NOGPU` = works without GPU (gstreamer::init() still needed for Buffer)
- `pub(crate)` items are listed only in `arch.md`
