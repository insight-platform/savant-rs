# DeepStream Metadata Public API

Crate: `deepstream`
Re-exports: `pub use deepstream_sys as sys;`

---

## Top-Level Exports

```rust
pub use batch_meta::BatchMeta;
pub use error::DeepStreamError;
pub use frame_meta::FrameMeta;
pub use infer_tensor_meta::{InferDims, InferTensorMeta};
pub use object_meta::ObjectMeta;
pub use user_meta::UserMeta;

pub type Result<T> = std::result::Result<T, DeepStreamError>;
```

---

## BatchMeta

```rust
pub struct BatchMeta { /* private */ }
```

Safe wrapper for `NvDsBatchMeta`. Cloning is shallow; all clones share the
same underlying batch and keep the DS metadata lock alive.

| Method | Signature | Notes |
|---|---|---|
| `unsafe from_gst_buffer` | `(buffer: *mut GstBuffer) → Result<Self>` | Obtain batch meta from GstBuffer. Buffer must contain DeepStream batch metadata. |
| `as_raw` | `(&self) → *mut NvDsBatchMeta` | Raw pointer to C struct |
| `num_frames` | `(&self) → u32` | Frames in batch |
| `max_frames_in_batch` | `(&self) → u32` | Max capacity |
| `frames` | `(&self) → Vec<FrameMeta>` | Iterate frame metadata |
| `is_empty` | `(&self) → bool` | `num_frames == 0` |

---

## FrameMeta

```rust
pub struct FrameMeta { /* private */ }
```

Safe wrapper for `NvDsFrameMeta`. Holds reference to `BatchMeta` for lifetime.
Implements `Clone` (shallow), `Debug`.

| Method | Signature | Notes |
|---|---|---|
| `unsafe from_raw` | `(raw: *mut NvDsFrameMeta, batch_meta: &BatchMeta) → Result<Self>` | |
| `as_raw` | `(&self) → *mut NvDsFrameMeta` | |
| `unsafe as_ref` | `(&self) → &NvDsFrameMeta` | Raw C reference |
| `unsafe as_mut` | `(&mut self) → &mut NvDsFrameMeta` | Raw C mutable reference |
| **Basic properties** | | |
| `frame_num` | `(&self) → i32` | |
| `buf_pts` / `set_buf_pts` | `(&self) → u64` / `(&mut self, pts: u64)` | Presentation timestamp |
| `source_id` | `(&self) → u32` | |
| `ntp_timestamp` | `(&self) → u64` | |
| `num_surfaces_per_frame` | `(&self) → i32` | |
| `surface_type` | `(&self) → u32` | |
| `surface_index` | `(&self) → u32` | |
| `batch_id` | `(&self) → u32` | |
| `pad_index` | `(&self) → u32` | |
| `num_objects` | `(&self) → u32` | |
| `width` / `height` | `(&self) → u32` | Source frame dimensions |
| **Geometry helpers** | | |
| `is_empty` | `(&self) → bool` | `num_objects == 0` |
| `area` | `(&self) → u32` | `width * height` |
| `center` | `(&self) → (f32, f32)` | `(w/2, h/2)` |
| `aspect_ratio` | `(&self) → f32` | `w / h` (0.0 if h == 0) |
| `is_landscape` | `(&self) → bool` | |
| `is_portrait` | `(&self) → bool` | |
| `is_square` | `(&self) → bool` | |
| **Misc / reserved** | | |
| `misc_frame_info` | `(&self) → [i64; 4]` | Full array |
| `set_misc_frame_info` | `(&mut self, info: [i64; 4])` | Full array |
| `get_misc_frame_info_at` | `(&self, index: usize) → Option<i64>` | Returns `None` if index ≥ 4 |
| `set_misc_frame_info_at` | `(&mut self, index: usize, value: i64) → bool` | Returns `false` if index ≥ 4 |
| `pipeline_width` | `(&self) → u32` | |
| `pipeline_height` | `(&self) → u32` | |
| `reserved` | `(&self) → [i64; 4]` | Full array |
| `set_reserved` | `(&mut self, reserved: [i64; 4])` | Full array |
| `get_reserved_at` | `(&self, index: usize) → Option<i64>` | |
| `set_reserved_at` | `(&mut self, index: usize, value: i64) → bool` | |
| **Object management** | | |
| `objects` | `(&self) → Vec<ObjectMeta>` | |
| `add_object` | `(&mut self, obj_meta: &mut ObjectMeta, parent: Option<&ObjectMeta>) → Result<()>` | |
| `remove_object` | `(&mut self, obj_meta: &ObjectMeta) → Result<()>` | |
| `clear_objects` | `(&mut self) → Result<()>` | |
| **User metadata** | | |
| `user_meta` | `(&self) → Vec<UserMeta>` | From `frame_user_meta_list` |
| `has_user_meta` | `(&self) → bool` | |

---

## ObjectMeta

```rust
pub struct ObjectMeta { /* private */ }
```

Safe wrapper for `NvDsObjectMeta`. Used for ROI objects and detection results.
Implements `Clone` (shallow), `Debug`.

| Method | Signature | Notes |
|---|---|---|
| `from_batch` | `(batch_meta: &BatchMeta) → Result<Self>` | Acquire from pool |
| `unsafe from_raw` | `(raw: *mut NvDsObjectMeta, batch_meta: &BatchMeta) → Result<Self>` | |
| `as_raw` | `(&self) → *mut NvDsObjectMeta` | Raw C pointer; use for direct FFI access |
| `unique_component_id` / `set_unique_component_id` | `(&self) → i32` / `(&mut self, id: i32)` | |
| `class_id` | `(&self) → i32` | |
| `set_class_id` | `(&mut self, id: i32)` | |
| `object_id` | `(&self) → u64` | Typically assigned by tracker |
| `set_object_id` | `(&mut self, id: u64)` | |
| `confidence` | `(&self) → f32` | Detection confidence |
| `set_confidence` | `(&mut self, confidence: f32)` | |
| `tracker_confidence` | `(&self) → f32` | Tracker confidence |
| `set_tracker_confidence` | `(&mut self, confidence: f32)` | |
| `label` | `(&self) → Result<Option<String>>` | Reads fixed-size `obj_label` buffer; `None` when empty |
| `set_label` | `(&mut self, label: &str) → Result<()>` | Truncates + NUL-terminates if too long |
| `parent` | `(&self) → Option<ObjectMeta>` | Parent object, if any |
| `set_parent` | `(&mut self, parent: &ObjectMeta) → Result<()>` | Both must be in same batch |
| `clear_parent` | `(&mut self)` | Sets parent pointer to null |
| `has_parent` | `(&self) → bool` | |
| `misc_obj_info` | `(&self) → [i64; 4]` | Miscellaneous info array |
| `set_misc_obj_info` | `(&mut self, info: [i64; 4])` | |
| `reserved` | `(&self) → [i64; 4]` | Reserved field array |
| `set_reserved` | `(&mut self, reserved: [i64; 4])` | |
| `user_meta` | `(&self) → Vec<UserMeta>` | All user metadata on this object |
| `has_user_meta` | `(&self) → bool` | |

> **No `rect_params`**: The wrapper does not expose rect fields directly.
> Use `as_raw()` to access `NvDsObjectMeta` fields like `rect_params` via FFI.

---

## InferTensorMeta

```rust
pub struct InferTensorMeta(*mut NvDsInferTensorMeta);
```

Non-owning wrapper for `NvDsInferTensorMeta` attached by nvinfer element.
Valid while parent buffer/sample is alive.
Implements `Clone` (shallow), `Send`, `Debug`.

| Method | Signature | Notes |
|---|---|---|
| `unsafe from_raw` | `(raw: *mut NvDsInferTensorMeta) → Result<Self>` | |
| `as_raw` | `(&self) → *mut NvDsInferTensorMeta` | |
| `unique_id` | `(&self) → u32` | |
| `num_output_layers` | `(&self) → u32` | |
| `gpu_id` | `(&self) → i32` | |
| `maintain_aspect_ratio` | `(&self) → bool` | |
| `output_layers_info` | `(&self) → *mut NvDsInferLayerInfo` | Raw pointer to layer info array |
| `layer_names` | `(&self) → Vec<String>` | Logs error for unsupported data types |
| `layer_dimensions` | `(&self) → Vec<InferDims>` | |
| `layer_data_types` | `(&self) → Vec<NvDsInferDataType>` | Raw data type values per layer |
| `out_buf_ptrs_host` / `out_buf_ptrs_dev` | `(&self) → Vec<*mut c_void>` | |
| `network_info` | `(&self) → NvDsInferNetworkInfo` | Network info struct (width, height, channels) |

---

## InferDims

```rust
pub struct InferDims {
    pub dimensions: Vec<u32>,
    pub num_elements: u32,
}
```

Tensor dimensions from `NvDsInferDims`.

---

## UserMeta

```rust
pub struct UserMeta { /* private */ }
```

Safe wrapper for `NvDsUserMeta` (custom user metadata).
Implements `Clone` (shallow), `Debug`.

| Method | Signature | Notes |
|---|---|---|
| `unsafe from_raw` | `(raw: *mut NvDsUserMeta, batch_meta: &BatchMeta) → Result<Self>` | |
| `as_raw` | `(&self) → *mut NvDsUserMeta` | |
| `unsafe as_ref` | `(&self) → &NvDsUserMeta` | |
| `unsafe as_mut` | `(&mut self) → &mut NvDsUserMeta` | |
| `batch_meta` | `(&self) → BatchMeta` | Cloned `BatchMeta` reference |
| `meta_type` | `(&self) → i32` | From `NvDsBaseMeta.meta_type` |
| `user_meta_data` | `(&self) → *mut c_void` | Raw pointer to user-defined data |
| `unsafe user_meta_data_as<T>` | `(&self) → Option<&T>` | Cast data; `None` if null |
| `unsafe user_meta_data_as_mut<T>` | `(&mut self) → Option<&mut T>` | Mutable cast |
| `has_user_data` | `(&self) → bool` | |
| `as_infer_tensor_meta` | `(&self) → Option<InferTensorMeta>` | Checks for `NVDSINFER_TENSOR_OUTPUT_META` type |

---

## DeepStreamError

See `errors.md` for full variant list and helper constructors.
