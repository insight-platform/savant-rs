# Picasso KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write picasso Python tests without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Python API: classes, constructors, methods, properties, signatures |
| `deps.md` | External type dependencies (draw_spec, deepstream, gstreamer, primitives) |
| `patterns.md` | Test patterns, helpers, GPU/non-GPU split, lifecycle idioms |
| `enums.md` | All enum types with variants |
| `errors.md` | Error conditions, RuntimeError/ValueError triggers |

## Usage
1. Read `api.md` first — it has all picasso types.
2. Read `deps.md` for imports from sibling modules.
3. Read `patterns.md` for test scaffolding and ready-to-copy helpers.
4. Consult `enums.md` only when building encoder configs.
5. Consult `errors.md` for negative-path tests.

## Conventions
- `SIG:` = constructor/method signature (Python typing)
- `DEF:` = default value
- `RET:` = return type
- `REQ:` = required parameter
- `OPT:` = optional parameter
- `→` separates input from output
- `⚠` = important caveat
- `GPU` = requires CUDA/DeepStream runtime
- `NOGPU` = works without GPU
