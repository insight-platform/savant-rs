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

---

## 8. OperatorInferenceOutput Drop Clears Buffer

`OperatorInferenceOutput::drop` **unconditionally** calls
`clear_all_frame_objects` on the output buffer. The `output_buffer` field is
declared last in the struct so Rust's field drop order destroys `frames`
(and their `TensorView` raw pointers) before the buffer that backs them.

Do not hold references to tensor data after `OperatorInferenceOutput` is dropped.

---

## 9. OperatorElement Lazy Scaler

`OperatorElement` uses `std::cell::OnceCell` (not `OnceLock`) for the lazy
`CoordinateScaler`. This means `OperatorElement` is `Send` but **not `Sync`**.
In PyO3 bindings, extract the scaler by value via `coordinate_scaler()` before
calling `py.detach()` — do not capture `&OperatorElement` in the GIL-released
closure.

```rust
let scaler = elem.coordinate_scaler(); // Copy, Send+Sync
let result = py.detach(|| scaler.scale_points(&points));
```

---

## 10. CoordinateScaler and Non-Uniform Scaling of Rotated Boxes

Under non-uniform scaling (Fill mode where `scale_x ≠ scale_y`), transforming
a rotated bounding box is an approximation — the true shape is a parallelogram,
not a rotated rectangle. This is documented and accepted because
`KeepAspectRatio` / `KeepAspectRatioSymmetric` modes always use uniform scale.

---

## 11. ROI Matching in Operator Callback

The operator callback matches `ElementOutput` to its ROI via `roi_id`:
- If `roi_id` is `Some(id)` and a matching ROI exists → the ROI's clamped
  bounding box (via `rbbox_to_rect_params`) is used for the scaler.
- If `roi_id` is `None` or no matching ROI found → full frame
  `(0, 0, frame_w, frame_h)` is used.

The clamped ROI rect is computed using the same `rbbox_to_rect_params` function
that DeepStream uses at ingress, ensuring consistency between the crop geometry
and the inverse transform.

---

## 12. PendingBatch Stores ROIs by Value

`submit.rs` clones the `Vec<RoiKind>` returned by `BatchFormationCallback`
into the `PendingBatch` (because the original `rois` are also destructured
into a `HashMap` for `NvInfer::submit`). The `rois` field in `submit.rs` is
iterated by reference (`.iter()`) and cloned per-entry when building the
`rois_map`.
