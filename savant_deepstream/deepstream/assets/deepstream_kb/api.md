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

| Method | Signature |
|---|---|
| `unsafe from_raw` | `(raw: *mut NvDsFrameMeta, batch_meta: &BatchMeta) → Result<Self>` |
| `as_raw` | `(&self) → *mut NvDsFrameMeta` |
| `frame_num` | `(&self) → i32` |
| `buf_pts` | `(&self) → u64` |
| `set_buf_pts` | `(&mut self, pts: u64)` |
| `source_id` | `(&self) → u32` |
| `num_objects` | `(&self) → u32` |
| `width` / `height` | `(&self) → u32` |
| `objects` | `(&self) → Vec<ObjectMeta>` |
| `add_object` / `remove_object` / `clear_objects` | `(&mut self, ...) → Result<()>` |

---

## ObjectMeta

```rust
pub struct ObjectMeta { /* private */ }
```

Safe wrapper for `NvDsObjectMeta`. Used for ROI objects and detection results.

| Method | Signature |
|---|---|
| `from_batch` | `(batch_meta: &BatchMeta) → Result<Self>` — acquire from pool |
| `unsafe from_raw` | `(raw: *mut NvDsObjectMeta, batch_meta: &BatchMeta) → Result<Self>` |
| `as_raw` | `(&self) → *mut NvDsObjectMeta` |
| `unique_component_id` / `set_unique_component_id` | `(&self) → i32` / `(&mut self, id)` |
| `class_id` / `object_id` | `(&self) → i32` / `u64` |
| `rect_params` | Access rect (left, top, width, height) |

---

## InferTensorMeta

```rust
pub struct InferTensorMeta(*mut NvDsInferTensorMeta);
```

Non-owning wrapper for `NvDsInferTensorMeta` attached by nvinfer element.
Valid while parent buffer/sample is alive.

| Method | Signature |
|---|---|
| `unsafe from_raw` | `(raw: *mut NvDsInferTensorMeta) → Result<Self>` |
| `as_raw` | `(&self) → *mut NvDsInferTensorMeta` |
| `unique_id` | `(&self) → u32` |
| `num_output_layers` | `(&self) → u32` |
| `gpu_id` | `(&self) → i32` |
| `layer_names` | `(&self) → Vec<String>` |
| `layer_dimensions` | `(&self) → Vec<InferDims>` |
| `out_buf_ptrs_host` / `out_buf_ptrs_dev` | `(&self) → Vec<*mut c_void>` |

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

---

## DeepStreamError

See `errors.md` for full variant list and helper constructors.
