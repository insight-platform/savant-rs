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
1. `api.md` — picasso's own public types + GPU utilities from deepstream_nvbufsurface
2. `deps.md` — imports from external crates (includes `buffer_gpu_id`, `EncoderConfig.gpu_id()`)
3. `patterns.md` — test scaffolding, helpers, templates (includes async drain + benchmark patterns)
4. `arch.md` — internals: threading model, async drain, GPU affinity, shared encoder state, render omission fast path
5. `errors.md` — all PicassoError variants including `GpuMismatch`

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA/NvBufSurface runtime
- `NOGPU` = works without GPU (gstreamer::init() still needed for Buffer)
- `pub(crate)` items are listed only in `arch.md`
