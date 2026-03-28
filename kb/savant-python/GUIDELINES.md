# savant_python KB — Agent Guidelines

## Purpose
Self-contained reference for agents working on `savant_python` — the maturin
crate that produces the `savant_rs` Python wheel.

## Files
| File | Content |
|---|---|
| `api.md` | Python module layout, submodule registration, class inventory |
| `arch.md` | Architecture: thin-wrapper pattern, feature flags, build pipeline |
| `deps.md` | Rust and Python dependencies, import paths |
| `patterns.md` | Build, test, and development patterns |
| `errors.md` | Common build/test issues and how to fix them |

## Usage
1. Read `arch.md` first — understand the thin-wrapper architecture.
2. Read `api.md` for the Python module tree and what each submodule exposes.
3. Read `deps.md` for import maps and dependency chains.
4. Read `patterns.md` for build commands, testing, and dev workflow.
5. Consult `errors.md` for common pitfalls.

## Critical Pitfalls (read before writing code)
- **savant_python is a thin wrapper** — it contains almost no logic. All class
  implementations live in `savant_core_py`. Editing savant_python's `lib.rs` is
  only needed when adding/removing submodules or changing feature gates.
- **Native module shadows Python packages** — PyO3 submodules are injected into
  `sys.modules` by `init_all()`. The `__init__.py` files under
  `python/savant_rs/<submod>/` are never loaded at runtime. Type stubs (`.pyi`)
  are the source of truth for IDE autocomplete.
- **Feature gates** — `deepstream` (which implies `gst`) gates nvinfer, picasso,
  and deepstream modules. Tests for these modules auto-skip without the feature.
- **Always use the Makefile** — do not run `maturin develop`, `cargo build`, or
  `pip install` manually. The Makefile handles `auditwheel repair` and correct
  feature flags.
- **PYI files must stay in sync** — whenever you change a Python-exposed class
  or function in `savant_core_py`, update the corresponding `.pyi` stub in
  `savant_python/python/savant_rs/`.

## Building and Testing

### Build + install with all features
```bash
SAVANT_FEATURES=deepstream make build_savant install
```

### Build + install without DeepStream (pure CPU)
```bash
make build_savant install
```

### Run Python tests
```bash
SAVANT_FEATURES=deepstream make sp-pytest
```

### What the Makefile does (do NOT replicate manually)
1. `build_savant` → `utils/build.sh debug`:
   - Resolves `./venv/bin/python3` as interpreter
   - Invokes `maturin build` with `--features=<SAVANT_FEATURES>`
   - Runs `auditwheel repair` on the resulting wheel
2. `install` → `pip install --force-reinstall dist/*.whl`
3. `sp-pytest` → builds, installs, runs `pytest savant_python/pytests/ -v --tb=short`

### Common mistakes to avoid
- **Do NOT run `maturin develop`** — skips auditwheel repair.
- **Do NOT run `cargo build -p savant_rs`** — use the Makefile.
- **E2E tests** (nvinfer, picasso) require GPU + DeepStream; auto-skip otherwise.

## Conventions
- `SIG:` = constructor/method signature
- `NOGPU` = works without GPU
- `GPU` = requires CUDA/DeepStream runtime
