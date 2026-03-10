# savant_python — Patterns & Development Workflow

## Build Commands

All commands run from the **project root** (`/workspaces/savant-rs`).

### Debug build (with DeepStream)
```bash
SAVANT_FEATURES=deepstream make build_savant install
```

### Debug build (CPU only)
```bash
make build_savant install
```

### Release build
```bash
SAVANT_FEATURES=deepstream make build_savant_release install
```

### Run Python tests
```bash
SAVANT_FEATURES=deepstream make sp-pytest
```

### Full dev cycle (format + clippy + build + install)
```bash
SAVANT_FEATURES=deepstream make all-dev
```

## Adding a New Python Class

1. **Implement** the `#[pyclass]` in `savant_core_py/src/<module>.rs`
2. **Register** it in the module's `register_classes()` function
3. **Expose** it in `savant_python/src/lib.rs` (import + add to the submodule)
4. **Create/update** the `.pyi` stub in `savant_python/python/savant_rs/<module>/`
5. **Write** pytests in `savant_python/pytests/test_<module>.py`

## Adding a New Submodule

1. Create the module in `savant_core_py/src/` with a `register_classes()` fn
2. In `savant_python/src/lib.rs`:
   - Add `#[pymodule]` function for the submodule
   - Call `register_classes()` from `savant_core_py`
   - Add to `init_all()` submodule registration
   - Add to `sys.modules` injection block
3. Create `.pyi` stub directory under `python/savant_rs/`
4. Create `__init__.py` if needed (usually empty or re-exporting)

## Test Patterns

### Standard test file header (no GPU)
```python
import pytest
from savant_rs.primitives import VideoFrame
from savant_rs.primitives.geometry import RBBox

def test_something():
    frame = gen_frame()
    # ...
```

### GPU/DeepStream test header
```python
import pytest

try:
    from savant_rs.deepstream import init_cuda, DsNvSurfaceBufferGenerator
    HAS_DS = True
except ImportError:
    HAS_DS = False

pytestmark = pytest.mark.skipif(not HAS_DS, reason="DeepStream not available")
```

### Using gen_frame / gen_empty_frame
```python
from savant_rs.utils import gen_frame, gen_empty_frame

def test_with_generated_frame():
    frame = gen_frame()  # has sample objects and attributes
    assert frame.source_id != ""

def test_with_empty_frame():
    frame = gen_empty_frame()  # no objects, no attributes
```

## PYI Stub Conventions

Stubs follow standard Python typing conventions:

```python
class MyClass:
    """Brief docstring."""

    def __init__(self, x: int, y: str, *, z: Optional[float] = None) -> None: ...

    @property
    def x(self) -> int: ...

    @x.setter
    def x(self, value: int) -> None: ...

    def method(self, arg: List[str]) -> Dict[str, int]: ...

    @staticmethod
    def from_json(s: str) -> "MyClass": ...
```

## Project Directory Layout (test assets)

Test assets for DeepStream modules live outside `savant_python`:

```
savant_deepstream/nvinfer/assets/       # model files, face images
savant_deepstream/picasso/assets/       # picasso-specific test assets
```

In pytests, reference them relative to the test file:
```python
ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "savant_deepstream", "nvinfer", "assets"
)
```

## GIL Conventions

- All `#[pymodule]` functions use `gil_used = false`.
- Release GIL (`py.allow_threads(|| ...)`) for compute-heavy operations.
- Do not release/acquire GIL multiple times in tight loops — the overhead
  can exceed the parallelism benefit.
- Use `py.detach()` for long-running synchronous calls that wait on
  external events (e.g., `infer_sync`).

## Wheel Structure

The built wheel contains:
```
savant_rs/
├── __init__.py              # pure-Python
├── savant_rs.so             # native extension
├── _ds_gpumat.py            # pure-Python helper
├── _ds_skia_canvas.py       # pure-Python helper
├── primitives/              # stubs + __init__.py
├── picasso/                 # stubs
├── nvinfer/                 # stubs
├── deepstream/              # stubs
├── py/                      # pure-Python packages
└── ...                      # other submodules
```
