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

## 4. Full-Frame Inference When rois = None

When `rois = None` is passed to `submit`/`infer_sync`, the pipeline reads
actual slot dimensions from the `NvBufSurface` and creates synthetic
full-frame ROIs automatically (`(0, 0, slot_width, slot_height)` per slot).
There is no separate "flexible" constructor -- `NvInferConfig::new` is the
only constructor, and it always requires `model_width` and `model_height`
(used for nvinfer config generation and coordinate scaling).

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

## 8. OperatorInferenceOutput Drop — Three-Step Cleanup

`OperatorInferenceOutput::drop` performs three steps in order:

1. Calls `clear_all_frame_objects` on the output buffer (invalidates tensor pointers).
2. Drops `output_buffer` via `self.output_buffer.take()` — this buffer is a parent to the per-frame buffers in deliveries and must be fully released before downstream is unblocked.
3. Calls `self.seal.release()` to unblock any `SealedDeliveries::unseal()` call.

The `output_buffer` is wrapped in `Option<SharedBuffer>` specifically so that step 2 can eagerly release it before step 3.  Do not hold references to tensor data after `OperatorInferenceOutput` is dropped.

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

---

## 13. SealedDeliveries Condvar and GIL

`SealedDeliveries::unseal()` blocks on a `parking_lot::Condvar` until `OperatorInferenceOutput` is dropped.  In PyO3 bindings, the GIL **must** be released before calling `unseal()` (via `py.detach()`), otherwise deadlock occurs:

- `unseal()` holds the GIL and waits on the Condvar.
- `OperatorInferenceOutput::drop()` (running on the Rust callback thread that uses `Python::attach`) needs the GIL to proceed.
- Neither can make progress → deadlock.

The binding implementation releases the GIL correctly:
```rust
let pairs = py.detach(move || sealed.unseal());
```

---

## 14. Buffer No Longer Accessible Inside the Result Callback

`OperatorFrameOutput` no longer exposes a `buffer` field.  The per-frame `SharedBuffer` is held privately inside `OperatorInferenceOutput` and extracted via `take_deliveries()`.  Use the sealed delivery pattern:

1. Process tensors via `output.frames()`.
2. Call `output.take_deliveries()` to get `SealedDeliveries`.
3. Drop (or let go out of scope) `output`.
4. Call `sealed.unseal()` to get `Vec<(VideoFrameProxy, SharedBuffer)>`.

---

## 15. Early Drop of SealedDeliveries is Safe

Dropping `SealedDeliveries` without calling `unseal()` is safe:

- The contained `SharedBuffer`s are freed normally.
- When `OperatorInferenceOutput` later drops, `seal.release()` + `notify_all()` runs against zero waiters (a harmless no-op).
- `take_deliveries()` returns `Option<SealedDeliveries>` — second call returns `None`, never panics.
- All drop orderings (`SealedDeliveries` first, `OperatorInferenceOutput` first, or never taken) are safe.
