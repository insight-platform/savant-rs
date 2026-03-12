# DeepStream Encoders Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code/tests for the `deepstream_encoders` crate without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, methods, signatures |
| `arch.md` | Architecture: module tree, GStreamer pipeline variants, format conversion |
| `patterns.md` | Test patterns, helpers, platform-aware config builders, ready-to-copy code |
| `errors.md` | EncoderError variants, error conditions, testing patterns |
| `caveats.md` | Critical Jetson/platform caveats, NVENC detection, VIC workarounds |

## Usage Order
1. `caveats.md` — **read first** — platform-specific pitfalls
2. `api.md` — public types, constructors, method signatures
3. `arch.md` — internals: pipeline variants, format conversion, B-frame enforcement
4. `patterns.md` — test scaffolding, platform-aware configs, templates
5. `errors.md` — all EncoderError variants

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA runtime + GStreamer
- `NVENC` = requires NVENC hardware (has_nvenc() guard needed)
- `JPEG` = requires nvjpegenc element (has_nvjpegenc() guard needed)
- `CPU` = works without any GPU encoding hardware (PNG, Raw)
