# Picasso KB — Agent Guidelines

## Purpose
Self-contained reference for agents to write picasso Python tests without reading source files.

## Files
| File | Content |
|---|---|
| `api.md` | Full Python API: classes, constructors, methods, properties, signatures |
| `deps.md` | External type dependencies, Rust/Python mixed-import architecture, GPU drawing helpers |
| `patterns.md` | Test patterns, helpers, GPU/non-GPU split, lifecycle idioms |
| `enums.md` | All enum types with variants |
| `errors.md` | Error conditions, RuntimeError/ValueError triggers |

## Usage
1. Read `api.md` first — it has all picasso types.
2. Read `deps.md` for imports from sibling modules + third-party GPU drawing.
3. Read `patterns.md` for test scaffolding, callback drawing recipes, and ready-to-copy helpers.
4. Consult `enums.md` only when building encoder configs.
5. Consult `errors.md` for negative-path tests.

## Critical Pitfalls (read before writing code)
- **Native module shadows Python packages** — `savant_rs.deepstream` and `savant_rs.picasso` are native PyO3 submodules registered directly in `sys.modules` by the `.so`. Any Python files in the corresponding `deepstream/` or `picasso/` directories are **never loaded at runtime**. Pure-Python helpers (e.g. `nvgstbuf_as_gpu_mat`, `nvbuf_as_gpu_mat`, `SkiaCanvas`) must live at the `savant_rs` package root as `_ds_*.py` files and be injected via `savant_rs/__init__.py`. See `deps.md` → Rust/Python Mixed-Import Architecture for the full pattern.
- **Render pipeline order**: draw spec → `on_render` → `on_gpumat` → encode. `on_render` draws **on top of** bboxes; do NOT `canvas.clear()` if using draw spec. Pre-fill input NvBufSurface for backgrounds. See `api.md` → Encode Pipeline Execution Order.
- **EncoderConfig builder methods shadow property setters** — `cfg.gpu_id(0)` ✅, `cfg.gpu_id = 0` ❌ (read-only at runtime). Same for `format()`. See `api.md` → EncoderConfig.
- **Callbacks fire on Rust worker threads** — `Mp4Muxer` is `Send` and can be used directly from callbacks. Other `unsendable` PyO3 objects would panic if accessed from a different thread; check the class annotation.
- **Skia GL context on headless EGL** — `skia.GrDirectContext.MakeGL()` returns `None`; use `skia.GrGLInterface.MakeEGL()` first. See `deps.md` → skia-python / `patterns.md` → on_render. The `SkiaCanvas` helper handles this automatically.
- **Skia + draw spec GL state conflict** — When both `draw` (ObjectDrawSpec) and `on_render` are used, Picasso's internal Skia context and the user's Skia context share the same GL context but cache state independently. Call `gr_context.resetContext()` at the start of every `on_render` invocation to force Skia to re-query GL state; without it, rendering is corrupted.
- **RBBox is centre-based** — `RBBox(xc, yc, width, height)`, not top-left corner. See `patterns.md` → Draw Spec + Callbacks Composition.

## Building and Testing savant_python

**Always use the Makefile from the project root.** Do not invoke `maturin`,
`pip install`, or `cargo build` manually — the Makefile's `sp-pytest` target
handles the full build→wheel→install→pytest chain correctly using the
project-local `./venv`.

### Build + install with deepstream features (picasso + IDE stubs)
```bash
SAVANT_FEATURES=deepstream make build_savant install
```
This builds the wheel with the `deepstream` Cargo feature and installs it into
`./venv`. The `savant_rs.picasso` type stubs (`.pyi`) live in
`python/savant_rs/picasso/picasso.pyi` and are packaged with the wheel, so IDE
symbol resolution works after install.

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

### Other useful feature combinations
```bash
SAVANT_FEATURES=deepstream make sp-pytest          # deepstream + picasso
SAVANT_FEATURES=gst make sp-pytest                 # gstreamer only (Mp4Muxer, Codec)
SAVANT_FEATURES=gst,deepstream make sp-pytest      # both
make sp-pytest                                     # no optional features
```

### ⚠ Common mistakes to avoid
- **Do NOT run `maturin develop`** — it does not produce a proper wheel and
  skips `auditwheel repair`, causing `libsavant_core_py.so` not-found errors.
- **Do NOT run `pip install` on a `.whl` directly** — the Makefile `install`
  target handles wheel selection and `--force-reinstall` correctly.
- **Do NOT set `LD_LIBRARY_PATH` manually** — the `auditwheel`-repaired wheel
  bundles or resolves all needed shared objects.
- **Do NOT run `cargo build -p savant_python`** — `savant_python` is a
  maturin-managed PyO3 crate; use the Makefile which calls `maturin build`.

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
