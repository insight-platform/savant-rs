# Critical Caveats & Design Decisions

## 1. submit / infer_sync Consume the Batch

Both `submit` and `infer_sync` call `batch.into_buffer()` internally. The
`SharedBuffer` must be the **sole owner** — no outstanding `SurfaceView`s,
clones, or batch structs. Drop all references before calling.

```rust
let shared = batch.shared_buffer();
drop(batch);
let view = SurfaceView::from_buffer(&shared, 0)?;
// ... use view ...
drop(view);
nvinfer.submit(shared, None)?;  // OK: sole owner
```

---

## 2. TensorView Lifetime vs MetaClearPolicy

When `MetaClearPolicy::After` or `Both`, dropping `BatchInferenceOutput`
calls `nvds_clear_obj_meta_list` on every frame. This **invalidates** the
raw pointers inside `TensorView` (host_ptr, device_ptr).

**Rule:** Consume or drop every `TensorView` **before** dropping the owning
`BatchInferenceOutput`.

```rust
let output = nvinfer.infer_sync(shared, None)?;
for elem in output.elements() {
    for t in &elem.tensors {
        let slice = unsafe { t.as_slice::<f32>() };
        // use slice
    }
}
drop(output);  // safe: all TensorViews consumed
```

---

## 3. process-mode=2 (Secondary) Only

NvInfer requires `process-mode=2` (secondary/object mode). Each buffer must
carry `NvDsObjectMeta` entries. The crate attaches them automatically via
`attach_batch_meta_with_rois`; callers supply ROIs or use full-frame fallback.

---

## 4. Flexible Config Requires Explicit ROIs

`NvInferConfig::new_flexible` has no `input_width`/`input_height`. The
full-frame fallback cannot compute dimensions. **Must** supply explicit
`rois` for every slot.

---

## 5. Jetson VIC and Small Surfaces

On Jetson (aarch64), VIC requires surfaces ≥ 16×16. For ROI crops smaller
than that, set `scaling-compute-hw=1` in nvinfer properties to force GPU
compute. Tests use `inject_jetson_scaling` for this.

---

## 6. Config File Lifetime

`validate_and_materialize()` returns a `NamedTempFile`. NvInfer stores it
in `_config_file` so it persists for the pipeline lifetime. Do not drop
the config file while the pipeline is running.

---

## 7. infer_sync Timeout

Blocks up to 30 seconds. If the pipeline is slow or stuck, `PipelineError`
with "infer_sync timed out" is returned.
