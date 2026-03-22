# NvInfer Public API

Crate: `nvinfer`
Prelude: `use nvinfer::*;` (or explicit imports)

---

## Top-Level Exports

```rust
pub use batch_meta_builder::attach_batch_meta_with_rois;
pub use config::NvInferConfig;
pub use deepstream::{InferDims, InferTensorMeta};
pub use error::{NvInferError, Result};
pub use meta_clear_policy::MetaClearPolicy;
pub use nvinfer_types::DataType;
pub use output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use pipeline::NvInfer;
pub use roi::Roi;
```

---

## NvInfer

```rust
pub struct NvInfer { /* private */ }
```

GStreamer pipeline wrapper for DeepStream nvinfer secondary inference.
Builds `appsrc ! [queue] ! nvinfer ! appsink`. Operates in `process-mode=2`
(secondary/object): each submitted buffer must carry `NvDsObjectMeta` entries
(one per ROI). ROIs are attached automatically via [`attach_batch_meta_with_rois`].

| Method | Signature | Notes |
|---|---|---|
| `new` | `(config: NvInferConfig, callback: InferCallback) → Result<Self>` | Spawns pipeline; callback invoked when inference completes (async mode) |
| `submit` | `(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) → Result<()>` | **Consumes** batch. Requires sole ownership (`into_buffer()` succeeds). Async: results delivered to callback. |
| `infer_sync` | `(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) → Result<BatchInferenceOutput>` | **Consumes** batch. Blocks up to 30s. Same ROI semantics as `submit`. |

**InferCallback:** `Box<dyn FnMut(BatchInferenceOutput) + Send>`

---

## NvInferConfig

```rust
pub struct NvInferConfig {
    pub name: String,
    pub nvinfer_properties: HashMap<String, String>,
    pub element_properties: HashMap<String, String>,
    pub gpu_id: u32,
    pub queue_depth: u32,
    pub input_format: String,
    pub input_width: Option<u32>,
    pub input_height: Option<u32>,
    pub meta_clear_policy: MetaClearPolicy,
    pub disable_output_host_copy: bool,
}
```

| Constructor | Signature | Notes |
|---|---|---|
| `new` | `(nvinfer_properties, input_format, input_width, input_height) → Self` | Fixed dimensions; full-frame ROI fallback when no ROIs supplied |
| `new_flexible` | `(nvinfer_properties, input_format) → Self` | No width/height; **must** supply explicit ROIs per slot |

| Builder | Signature |
|---|---|
| `with_element_properties` | `(self, properties: HashMap) → Self` |
| `gpu_id` | `(self, gpu_id: u32) → Self` |
| `queue_depth` | `(self, depth: u32) → Self` — 0 = no queue (sync) |
| `name` | `(self, name: impl Into<String>) → Self` |
| `meta_clear_policy` | `(self, policy: MetaClearPolicy) → Self` |
| `disable_output_host_copy` | `(self, disable: bool) → Self` — skip D2H copy; only GPU pointers valid |

**Mandatory nvinfer keys** (auto-injected if missing): `process-mode=2`, `output-tensor-meta=1`.
`disable-output-host-copy` is controlled via `NvInferConfig.disable_output_host_copy` and must **not** be set in `nvinfer_properties`.
Use dotted notation `section.key` for per-class config (e.g. `class-attrs-0.nms-iou-threshold`).

---

## Roi

```rust
pub struct Roi {
    pub id: i64,
    pub bbox: RBBox,
}
```

Region of interest: identifier + bounding box. Passed per batch slot to `submit`/`infer_sync`.
`object_id = roi.id as u64` in `NvDsObjectMeta`; returned in `ElementOutput::roi_id`.
When `RBBox` has non-zero angle, axis-aligned wrapping box is computed automatically.

---

## BatchInferenceOutput

```rust
pub struct BatchInferenceOutput { /* private */ }
```

Owns the output `GstBuffer`. Tensor views borrow from it.

| Method | Signature |
|---|---|
| `elements` | `(&self) → &[ElementOutput]` |
| `num_elements` | `(&self) → usize` |
| `host_copy_enabled` | `(&self) → bool` — `false` when `disable_output_host_copy` was set |
| `buffer` | `(&self) → SharedBuffer` — clone for downstream (e.g. SavantIdMeta) |

⚠ **Drop semantics:** When `MetaClearPolicy::After` or `Both`, dropping clears all
`NvDsObjectMeta` from frames. **Consume all `TensorView`s before dropping** — pointers invalidated.

---

## ElementOutput

```rust
pub struct ElementOutput {
    pub frame_id: Option<i64>,
    pub roi_id: Option<i64>,
    pub tensors: Vec<TensorView>,
}
```

Per-ROI inference output. `frame_id` from SavantIdMeta; `roi_id` from `Roi::id` (or `None` for full-frame sentinel).

---

## TensorView

```rust
pub struct TensorView {
    pub name: String,
    pub dims: InferDims,
    pub data_type: DataType,
    pub host_ptr: *const c_void,
    pub device_ptr: *const c_void,
    pub byte_length: usize,
    pub host_copy_enabled: bool,
}
```

Zero-copy view into a single output tensor. Valid while parent `BatchInferenceOutput` is alive.
When `host_copy_enabled` is `false`, host pointers contain stale data — only `device_ptr` is usable.

| Method | Signature |
|---|---|
| `unsafe as_slice<T>` | `(&self) → &[T]` — interpret host data as typed slice; returns `&[]` when host copy disabled |

---

## MetaClearPolicy

```rust
pub enum MetaClearPolicy {
    None,
    Before,   // DEF: clear input before attaching ROIs
    After,    // clear output on BatchInferenceOutput drop
    Both,
}
```

| Method | Signature |
|---|---|
| `clear_before` | `(self) → bool` |
| `clear_after` | `(self) → bool` |

---

## DataType

```rust
pub enum DataType { Float, Half, Int8, Int32 }
```

| Method | Signature |
|---|---|
| `element_size` | `(self) → usize` |

---

## attach_batch_meta_with_rois

```rust
pub fn attach_batch_meta_with_rois(
    buffer: &mut gstreamer::BufferRef,
    num_frames: u32,
    max_batch_size: u32,
    policy: MetaClearPolicy,
    rois: Option<&HashMap<u32, Vec<Roi>>>,
    input_width: u32,
    input_height: u32,
) → Result<u32>
```

Attach `NvDsBatchMeta` for secondary-mode ROI inference. Returns number of ROIs dropped (0 = success).
Called internally by `NvInfer::submit` / `infer_sync`; rarely used directly.
