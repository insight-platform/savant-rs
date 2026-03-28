# savant_python — Common Errors & Troubleshooting

## Build Errors

### `maturin build` fails with missing features
**Symptom**: Classes like `NvInfer`, `PicassoEngine` not found at runtime.
**Cause**: Built without `deepstream` feature.
**Fix**: `SAVANT_FEATURES=deepstream make build_savant install`

### `auditwheel: cannot repair`
**Symptom**: Build succeeds but install fails.
**Cause**: Shared library dependencies not found on the system.
**Fix**: Ensure all DeepStream/CUDA libraries are installed. Use the provided
Docker container.

### Linker errors for `libcuda.so` / `libnvinfer.so`
**Symptom**: `cargo build` fails with undefined symbols.
**Cause**: DeepStream SDK not installed or not on `LD_LIBRARY_PATH`.
**Fix**: Build inside the DS container; check `nvidia-smi` availability.

## Import Errors

### `ModuleNotFoundError: No module named 'savant_rs.nvinfer'`
**Cause 1**: Built without `deepstream` feature.
**Cause 2**: Wheel not reinstalled after rebuild.
**Fix**: `SAVANT_FEATURES=deepstream make build_savant install`

### `ImportError: ... undefined symbol`
**Cause**: Wheel built for different Python version or platform.
**Fix**: Rebuild with `make build_savant install`. Ensure `./venv/bin/python3`
matches the target Python.

## Test Errors

### Tests skip with "DeepStream not available"
**Cause**: Expected behavior when running without GPU/DS runtime.
**Note**: These tests auto-skip via `pytestmark`.

### `RuntimeError: TensorView data has been freed`
**Cause**: `BatchInferenceOutput` was dropped before accessing tensor data.
**Fix**: Keep a reference to `BatchInferenceOutput` while using `TensorView`.

### `RuntimeError: init_cuda must be called before any GPU operation`
**Cause**: Forgot to call `init_cuda(gpu_id)` before creating buffers.
**Fix**: Call `init_cuda(0)` at the start of GPU test functions.

### `RuntimeError: SharedBuffer has outstanding references`
**Cause**: Calling `submit()` or `infer_sync()` while other Python objects
(SurfaceView, batch, source buffer) still hold Arc references.
**Fix**: `del` all intermediate objects before passing the SharedBuffer to
the engine. E.g.: `del batch, view, src_buf` before `engine.infer_sync(buf)`.

## PYI Stubs Out of Sync

### IDE shows wrong signature / missing method
**Cause**: `.pyi` stub not updated after Rust API change.
**Fix**: Update the corresponding `.pyi` file in
`savant_python/python/savant_rs/<module>/`.

**How to detect**: Compare the `#[pymethods]` block in `savant_core_py` with
the `.pyi` file. Every `#[new]`, `#[getter]`, `#[setter]`, `#[pyo3(name=...)]`
method should have a corresponding stub.

## GIL-related Issues

### Deadlock in callback
**Cause**: Python callback acquires GIL while Rust code holds a lock that the
GIL-holding thread also needs.
**Fix**: Release GIL before calling into Rust code that may invoke callbacks.
Use `py.allow_threads()` or `py.detach()`.

### Performance regression after GIL changes
**Cause**: Releasing/acquiring GIL in tight loop.
**Fix**: Batch operations to minimize GIL transitions. Profile with
`estimate_gil_contention()` utility.
