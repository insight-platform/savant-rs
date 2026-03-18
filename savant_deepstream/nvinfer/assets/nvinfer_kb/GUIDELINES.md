# NvInfer Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code/tests for the
`nvinfer` crate without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, methods, signatures |
| `arch.md` | Architecture: module tree, pipeline flow, ROI handling |
| `patterns.md` | Test patterns, helpers, model config builders |
| `errors.md` | NvInferError variants, error conditions |
| `caveats.md` | Critical design decisions, known pitfalls |

## Usage Order
1. `caveats.md` — **read first** — buffer ownership, TensorView lifetime
2. `api.md` — public types, constructors, method signatures
3. `arch.md` — internals: pipeline, batch meta, ROI attachment
4. `patterns.md` — test scaffolding, model properties, ready-to-copy templates
5. `errors.md` — all NvInferError variants

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA + GStreamer + DeepStream runtime
