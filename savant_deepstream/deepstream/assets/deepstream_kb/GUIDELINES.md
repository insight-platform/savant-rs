# DeepStream metadata Rust KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write Rust code using the
`deepstream` crate (DeepStream metadata wrappers) without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Rust public API: structs, enums, methods, signatures |
| `arch.md` | Architecture: module tree, metadata hierarchy, FFI layer |
| `errors.md` | DeepStreamError variants, error conditions |

## Usage Order
1. `api.md` — public types, constructors, method signatures
2. `arch.md` — internals: metadata hierarchy, lock model
3. `errors.md` — all DeepStreamError variants

## Conventions
- `SIG:` = function/method signature
- `DEF:` = Default impl value
- `RET:` = return type
- `→` separates input from output
- `⚠` = important caveat
- `unsafe` = caller must uphold safety invariants

## Note
This crate is a **low-level wrapper** around DeepStream C metadata. Most
users interact with it indirectly via `nvinfer`, which consumes `BatchMeta`,
`InferTensorMeta`, `InferDims` for inference output extraction.
