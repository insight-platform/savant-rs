# DeepStream Decoders Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code/tests for the
`deepstream_decoders` crate without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, methods, signatures |
| `arch.md` | Architecture: module tree, GStreamer pipeline variants, frame-ID propagation, auto-restart |
| `patterns.md` | Test patterns, helpers, platform-aware config builders, ready-to-copy code |
| `errors.md` | DecoderError variants, error conditions, testing patterns |
| `caveats.md` | Critical Jetson/platform caveats, NVDEC detection, decoder pool behavior |

## Usage Order
1. `caveats.md` — read first for platform-specific pitfalls
2. `api.md` — public types, constructors, method signatures
3. `arch.md` — internals: pipelines, bridge meta, PTS map, restart flow
4. `patterns.md` — test scaffolding and templates
5. `errors.md` — full DecoderError reference

## Conventions
- `SIG:` = function/method signature
- `DEF:` = default value
- `RET:` = return type
- `GPU` = requires CUDA runtime + GStreamer
- `NVDEC` = requires hardware decode support
- `JPEG-GPU` = requires `nvjpegdec`
- `CPU` = no hardware decoder requirement (PNG, JPEG CPU, raw upload paths)
