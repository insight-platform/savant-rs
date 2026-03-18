# NvInfer KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write nvinfer Python tests without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Python API: classes, constructors, methods, properties, signatures |
| `deps.md` | External type dependencies, buffer interop, import map |
| `patterns.md` | Test patterns, helpers, GPU/non-GPU split, lifecycle idioms |
| `enums.md` | All enum types with variants |
| `errors.md` | Error conditions, RuntimeError/ValueError triggers |

## Usage
1. Read `api.md` first — it has all nvinfer types.
2. Read `deps.md` for imports from sibling modules + third-party (numpy, PIL, ctypes).
3. Read `patterns.md` for test scaffolding, E2E lifecycle, and ready-to-copy helpers.
4. Consult `enums.md` for MetaClearPolicy and DataType details.
5. Consult `errors.md` for negative-path tests.

## Critical Pitfalls (read before writing code)
- **Native module shadows Python packages** — `savant_rs.nvinfer` is a native PyO3 submodule registered directly in `sys.modules` by the `.so`. The `nvinfer/__init__.py` is **never loaded at runtime**. Type stubs live at `nvinfer/nvinfer.pyi`.
- **Tensor lifetime / Arc** — `TensorView` holds a reference to the parent `BatchInferenceOutput` via `Arc`. Tensor data (host pointers) remains valid as long as either the `BatchInferenceOutput` or any child `TensorView`/`ElementOutput` is alive. Once all references are dropped, tensor data is freed. Always call `as_bytes()` or `as_numpy()` to copy data before dropping the output.
- **GIL in callbacks** — The async callback fires on a GStreamer thread with the GIL acquired via `Python::attach()`. Keep callbacks fast; heavy work should be offloaded.
- **Buffer ownership** — `submit()` and `infer_sync()` consume the `gst::Buffer` from the `SharedBuffer` (via `take_inner()`). After calling either, the `SharedBuffer` is consumed (`is_consumed` is true). All outstanding references (SurfaceView, batch objects) must be deleted before consumption, otherwise the engine raises `RuntimeError: SharedBuffer has outstanding references`.
- **PTS correlation** — Pipeline correlation is auto-generated internally via `SavantIdMeta` attached to the buffer. There is no separate `batch_id` parameter.

## Building and Testing savant_python

**Always use the Makefile from the project root.** Do not invoke `maturin`,
`pip install`, or `cargo build` manually.

### Build + install with deepstream features (nvinfer + picasso + IDE stubs)
```bash
SAVANT_FEATURES=deepstream make build_savant install
```

### Build + test with deepstream features enabled
```bash
SAVANT_FEATURES=deepstream make sp-pytest
```

### What the command does (do NOT replicate manually)
1. `build_savant` — runs `utils/build.sh debug` which:
   - resolves `./venv/bin/python3` as the interpreter
   - invokes `maturin build` with `--features=deepstream`
   - runs `auditwheel repair` on the resulting wheel
2. `install` — installs the wheel into `./venv` with `pip install --force-reinstall`
3. Runs `pytest` from `savant_python/pytests/` using `./venv/bin/python3 -m pytest`

### ⚠ Common mistakes to avoid
- **Do NOT run `maturin develop`** — it skips `auditwheel repair`.
- **Do NOT run `cargo build -p savant_python`** — use the Makefile.
- **E2E test** (`test_nvinfer.py`) requires GPU + DeepStream + model assets; auto-skips otherwise.

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
