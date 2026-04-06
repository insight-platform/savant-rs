# NvInfer Public API

Crate: `nvinfer`
Prelude: `use nvinfer::*;` (or explicit imports)

---

## Top-Level Exports

```rust
pub use batch_meta_builder::attach_batch_meta_with_rois;
pub use batching_operator::{
    BatchFormationCallback, BatchFormationResult, CoordinateScaler,
    NvInferBatchingOperator, NvInferBatchingOperatorConfig,
    NvInferBatchingOperatorConfigBuilder, OperatorElement,
    OperatorFrameOutput, OperatorInferenceOutput, OperatorResultCallback,
    SealedDeliveries,
};
pub use config::NvInferConfig;
pub use deepstream::{InferDims, InferTensorMeta};
pub use deepstream_buffers::VideoFormat;
pub use deepstream_buffers::{DstPadding, Rect, SharedBuffer, SurfaceView, TransformConfig};
pub use error::{NvInferError, Result};
pub use meta_clear_policy::MetaClearPolicy;
pub use model_color_format::ModelColorFormat;
pub use model_input_scaling::ModelInputScaling;
pub use nvinfer_types::DataType;
pub use output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use pipeline::NvInfer;
pub use roi::{Roi, RoiKind};
```

### Prelude

`use nvinfer::prelude::*;` re-exports all of the above plus `VideoFormat`.

---

## ModelInputScaling

```rust
pub enum ModelInputScaling {
    Fill,                       // stretch to model input (maintain-aspect-ratio=0)
    KeepAspectRatio,            // AR + left/bottom padding (symmetric-padding=0)
    KeepAspectRatioSymmetric,   // AR + symmetric padding (symmetric-padding=1)
}
```

Set on [`NvInferConfig::scaling`]. The generated nvinfer config always includes
the corresponding `maintain-aspect-ratio` / `symmetric-padding` entries; callers
must not put those keys in `nvinfer_properties`.

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
| `infer_sync` | `(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) → Result<BatchInferenceOutput>` | **Consumes** batch. Blocks up to 30s. Delegates to `infer_sync_with_timeout`. |
| `infer_sync_with_timeout` | `(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>, timeout: Duration) → Result<BatchInferenceOutput>` | **Consumes** batch. Blocks up to `timeout`. Same ROI semantics as `submit`. |
| `shutdown` | `(&mut self) → Result<()>` | Graceful shutdown: sends EOS, waits up to 10s for drain, sets pipeline to Null. |

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
    pub input_format: VideoFormat,
    pub model_width: u32,
    pub model_height: u32,
    pub model_color_format: ModelColorFormat,
    pub meta_clear_policy: MetaClearPolicy,
    pub disable_output_host_copy: bool,
    pub scaling: ModelInputScaling,
}
```

| Constructor | Signature | Notes |
|---|---|---|
| `new` | `(nvinfer_properties, input_format: VideoFormat, model_width: u32, model_height: u32, model_color_format: ModelColorFormat) → Self` | Creates config with model dimensions; when rois = None, reads actual slot dimensions for full-frame inference |

| Builder | Signature |
|---|---|
| `with_element_properties` | `(self, properties: HashMap) → Self` |
| `gpu_id` | `(self, gpu_id: u32) → Self` |
| `queue_depth` | `(self, depth: u32) → Self` — 0 = no queue (sync) |
| `name` | `(self, name: impl Into<String>) → Self` |
| `meta_clear_policy` | `(self, policy: MetaClearPolicy) → Self` |
| `disable_output_host_copy` | `(self, disable: bool) → Self` — skip D2H copy; only GPU pointers valid |
| `scaling` | `(self, scaling: ModelInputScaling) → Self` — input resize / padding mode |
| `model_input_dimensions` | `(&self) → (u32, u32)` — returns `(model_width, model_height)` |

**Mandatory nvinfer keys** (auto-injected if missing): `process-mode=2`, `output-tensor-meta=1`, `network-type=100`, `gie-unique-id=1`.

**Always injected from `scaling`:** `maintain-aspect-ratio` (and `symmetric-padding` when aspect ratio is preserved).

**Forbidden nvinfer properties** (rejected with `InvalidConfig` error):

| Key | Reason |
|---|---|
| `operate-on-gie-id` | Filters objects by upstream GIE id; synthetic ROI sentinels use `unique_component_id = -1` and would be silently skipped. |
| `operate-on-class-ids` | Filters objects by class id; sentinels carry no meaningful class and would be skipped. |
| `secondary-reinfer-interval` | Controls re-inference cadence across frames in a multi-frame pipeline; meaningless in single-shot mode and can silently skip frames. |
| `num-detected-classes` | Only meaningful for detector `network-type` (0); misleading with `network-type=100`. |
| `disable-output-host-copy` | Controlled via `NvInferConfig.disable_output_host_copy`; must not be set in `nvinfer_properties`. |
| `maintain-aspect-ratio` | Controlled via `NvInferConfig.scaling`; must not be set in `nvinfer_properties`. |
| `symmetric-padding` | Controlled via `NvInferConfig.scaling`; must not be set in `nvinfer_properties`. |

Use dotted notation `section.key` for per-class config (e.g. `class-attrs-0.nms-iou-threshold`).

---

## Roi

```rust
pub struct Roi {
    pub id: i64,
    pub bbox: RBBox,
}
```

| Constructor | Signature |
|---|---|
| `new` | `(id: i64, bbox: RBBox) → Self` |

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
    pub roi_id: Option<i64>,
    pub slot_number: u32,
    pub tensors: Vec<TensorView>,
}
```

Per-ROI inference output. `slot_number` is `NvDsFrameMeta.batch_id` (index into `NvBufSurface.surfaceList`). User frame ids are read from `BatchInferenceOutput::buffer()` via `SharedBuffer::savant_ids()` (aligned with slots). `roi_id` from `Roi::id` (or `None` for full-frame sentinel).

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

| Method | Signature | Notes |
|---|---|---|
| `unsafe as_slice<T>` | `(&self) → &[T]` | Interpret host data as typed slice; returns `&[]` when host copy disabled |
| `as_f32s` | `(&self) → Result<&[f32]>` | Safe typed accessor for `DataType::Float` tensors |
| `as_i32s` | `(&self) → Result<&[i32]>` | Safe typed accessor for `DataType::Int32` tensors |
| `as_i8s` | `(&self) → Result<&[i8]>` | Safe typed accessor for `DataType::Int8` tensors |

`as_f32s` / `as_i32s` / `as_i8s` return `Err(NvInferError::TensorTypeMismatch)` if `data_type` does not match, or `Err(NvInferError::HostDataUnavailable)` if host copy is disabled / pointer is null / byte length is zero.

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

| Method | Signature | Notes |
|---|---|---|
| `element_size` | `(self) → usize` | Size in bytes |
| `name` | `(self) → &'static str` | `"float32"`, `"float16"`, `"int8"`, `"int32"` |

---

## attach_batch_meta_with_rois

```rust
pub fn attach_batch_meta_with_rois(
    buffer: &mut gstreamer::BufferRef,
    num_frames: u32,
    max_batch_size: u32,
    policy: MetaClearPolicy,
    rois: Option<&HashMap<u32, Vec<Roi>>>,
) → Result<usize>
```

Attach `NvDsBatchMeta` for secondary-mode ROI inference. Returns number of ROIs dropped (0 = success).
Called internally by `NvInfer::submit` / `infer_sync`; rarely used directly.

---

## RoiKind

```rust
pub enum RoiKind {
    FullFrame,
    Rois(Vec<Roi>),
}
```

Per-frame ROI specification for the batching operator. `FullFrame` infers on
the entire frame (no per-object ROIs). `Rois(vec)` infers on specific regions.

---

# Batching Operator Layer

## NvInferBatchingOperatorConfig

```rust
pub struct NvInferBatchingOperatorConfig {
    pub max_batch_size: usize,
    pub max_batch_wait: Duration,
    pub nvinfer: NvInferConfig,
}
```

| Field | Description |
|---|---|
| `max_batch_size` | Maximum batch size; triggers immediate submission when reached |
| `max_batch_wait` | Maximum time before submitting a partial batch |
| `nvinfer` | Forwarded to the inner `NvInfer` pipeline |

### `NvInferBatchingOperatorConfig::builder(nvinfer_config)`

Returns a `NvInferBatchingOperatorConfigBuilder`. Defaults: `max_batch_size = 1`,
`max_batch_wait = 50 ms`.

Builder methods (all return `Self`): `max_batch_size(usize)`,
`max_batch_wait(Duration)`, `build() → NvInferBatchingOperatorConfig`.

---

## BatchFormationResult

```rust
pub struct BatchFormationResult {
    pub ids: Vec<SavantIdMetaKind>,
    pub rois: Vec<RoiKind>,
}
```

Returned by the `BatchFormationCallback`. `ids` is per-frame Savant IDs for
`NonUniformBatch::finalize`. `rois` is per-frame ROI specification (parallel
to the frames slice passed to the callback).

---

## Callback Types

```rust
pub type BatchFormationCallback =
    Arc<dyn Fn(&[VideoFrameProxy]) -> BatchFormationResult + Send + Sync>;

pub type OperatorResultCallback = Box<dyn FnMut(OperatorInferenceOutput) + Send>;
```

---

## NvInferBatchingOperator

```rust
pub struct NvInferBatchingOperator { /* private */ }
```

Higher-level batching layer over `NvInfer`. Accumulates individual frames into
batches and delivers per-frame results via `OperatorResultCallback`.

| Method | Signature | Notes |
|---|---|---|
| `new` | `(config, batch_formation, result_callback) → Result<Self>` | Spawns inner `NvInfer` + timer thread |
| `add_frame` | `(&self, frame: VideoFrameProxy, buffer: SharedBuffer) → Result<()>` | Add frame; submits when batch full |
| `flush` | `(&self) → Result<()>` | Submit current partial batch immediately |
| `shutdown` | `(&mut self) → Result<()>` | Flush, stop timer, shut down `NvInfer` |

**Drop:** Signals shutdown flag + notifies condvar; joins timer thread.

---

## OperatorInferenceOutput

```rust
pub struct OperatorInferenceOutput { /* private */ }
```

Full batch inference result. Owns the output `GstBuffer`. `TensorView`
pointers in `frames[].elements[].tensors` borrow from the internal output
buffer — declared last in the struct so frames are dropped first.

| Method | Signature |
|---|---|
| `frames` | `(&self) → &[OperatorFrameOutput]` |
| `host_copy_enabled` | `(&self) → bool` |
| `take_deliveries` | `(&mut self) → Option<SealedDeliveries>` — first call returns `Some`, subsequent calls return `None` |

⚠ **Drop:** Three-step cleanup: (1) clears `NvDsObjectMeta` on the output buffer, (2) drops `output_buffer` via `Option::take()`, (3) calls `seal.release()` to unblock downstream `SealedDeliveries::unseal()`.

---

## SealedDeliveries

```rust
pub struct SealedDeliveries { /* private */ }
```

A batch of `(VideoFrameProxy, SharedBuffer)` pairs sealed until the associated
`OperatorInferenceOutput` is dropped.  Individual buffers are inaccessible
while sealed.

| Method | Signature | Notes |
|---|---|---|
| `len` | `(&self) → usize` | Number of frames |
| `is_empty` | `(&self) → bool` | |
| `is_released` | `(&self) → bool` | Non-blocking check |
| `unseal` | `(self) → Vec<(VideoFrameProxy, SharedBuffer)>` | **Blocks** on Condvar until seal released |
| `unseal_timeout` | `(self, timeout: Duration) → Result<Vec<…>, Self>` | Blocks up to `timeout`; `Err(self)` if timeout expires |
| `try_unseal` | `(self) → Result<Vec<…>, Self>` | Non-blocking; `Err(self)` if still sealed |

`unsafe impl Send for SealedDeliveries` — all fields are `Send`.

⚠ **Drop safety:** Dropping `SealedDeliveries` without calling `unseal()` is safe — contained `SharedBuffer`s are freed and `Condvar::notify_all` to zero waiters is a no-op.

---

## OperatorFrameOutput

```rust
pub struct OperatorFrameOutput {
    pub frame: VideoFrameProxy,
    pub elements: Vec<OperatorElement>,
}
```

Per-frame inference result (callback view — no buffer access).

Per-frame inference result. The per-frame `SharedBuffer` is held internally by the parent `OperatorInferenceOutput` and is only accessible after calling `take_deliveries()` then `SealedDeliveries::unseal()`.

---

## OperatorElement

```rust
pub struct OperatorElement { /* private fields */ }
```

Per-element inference output wrapped with lazy coordinate scaling.
Implements `Deref<Target = ElementOutput>` so `roi_id`, `slot_number`,
`tensors` are accessible directly.

| Method | Signature | Notes |
|---|---|---|
| `coordinate_scaler` | `(&self) → CoordinateScaler` | Returns a copy of the lazily-initialized scaler |
| `scale_points` | `(&self, &[(f32, f32)]) → Vec<(f32, f32)>` | Transform points from model space to frame coords |
| `scale_ltwh` | `(&self, &[[f32; 4]]) → Vec<[f32; 4]>` | Transform (left, top, width, height) boxes |
| `scale_ltrb` | `(&self, &[[f32; 4]]) → Vec<[f32; 4]>` | Transform (left, top, right, bottom) boxes |
| `scale_rbboxes` | `(&self, &[RBBox]) → Vec<RBBox>` | Transform rotated bounding boxes |

**Deref target:** `ElementOutput` (provides `roi_id`, `slot_number`, `tensors`)

---

## CoordinateScaler

```rust
#[derive(Debug, Clone, Copy)]
pub struct CoordinateScaler {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}
```

Precomputed affine coefficients for `frame_xy = offset + model_xy * scale`.

| Method | Signature |
|---|---|
| `new` | `(roi_left, roi_top, roi_w, roi_h, model_w, model_h, scaling) → Self` |
| `scale_point` | `(&self, x, y) → (f32, f32)` |
| `scale_points` | `(&self, &[(f32, f32)]) → Vec<(f32, f32)>` |
| `scale_ltwh` | `(&self, l, t, w, h) → (f32, f32, f32, f32)` |
| `scale_ltwh_batch` | `(&self, &[[f32; 4]]) → Vec<[f32; 4]>` |
| `scale_ltrb` | `(&self, l, t, r, b) → (f32, f32, f32, f32)` |
| `scale_ltrb_batch` | `(&self, &[[f32; 4]]) → Vec<[f32; 4]>` |
| `scale_rbbox` | `(&self, &RBBox) → RBBox` |
| `scale_rbboxes` | `(&self, &[RBBox]) → Vec<RBBox>` |

For rotated boxes, the trig from `RBBox::scale` is inlined to produce a
single `RBBox::new()` call (no intermediate atomics). Under non-uniform
scaling (Fill mode, scale_x ≠ scale_y), the angle is recomputed.

---

## NvInferError

```rust
pub enum NvInferError {
    PipelineError(String),
    ElementCreationFailed(String),
    LinkFailed(String),
    InvalidProperty(String),
    InvalidConfig(String),
    BatchMetaFailed(String),
    NullPointer(String),
    GstInit(String),
    BatchFormationFailed(String),
    OperatorShutdown,
    TensorTypeMismatch { expected: &'static str, actual: &'static str },
    HostDataUnavailable,
    Buffer(#[from] NvBufSurfaceError),
}
```

| Variant | Trigger |
|---|---|
| `TensorTypeMismatch` | `as_f32s()` / `as_i32s()` / `as_i8s()` called on wrong `DataType` |
| `HostDataUnavailable` | Typed accessor called when host copy disabled, null pointer, or zero bytes |
| `Buffer` | Wraps `deepstream_buffers::NvBufSurfaceError` via `From` |
