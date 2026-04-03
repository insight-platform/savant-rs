# Savant Rust Optimized Algorithms, Data Structures and Services

Documentation is available at: https://insight-platform.github.io/savant-rs/

# Installation options

To install a standalone version with clientsdk dependencies use `[clientsdk]`.

# Python bindings (wheel)

The `Makefile` uses the virtualenv at **`/opt/venv`** by default (`VENV_DIR`, `pip`, `pytest`, and **maturin** via `PYTHON_INTERPRETER` in `utils/build.sh`).

DeepStream-enabled bindings:

```bash
SAVANT_FEATURES=deepstream make dev install
```

Other feature sets:

```bash
make dev install                                 # savant_rs default features
SAVANT_FEATURES=gst make dev install            # GStreamer-related APIs
```

Override the interpreter or venv location if needed:

```bash
make VENV_DIR=/path/to/venv dev install
make PYTHON_INTERPRETER= dev install            # use build.sh fallback (./venv or maturin -f)
```

