# DeepStream Metadata Architecture

## Module Tree
```
deepstream/src/
├── lib.rs           # Re-exports, Result type
├── batch_meta.rs    # BatchMeta: NvDsBatchMeta wrapper, lock guard
├── frame_meta.rs    # FrameMeta: NvDsFrameMeta wrapper
├── object_meta.rs   # ObjectMeta: NvDsObjectMeta wrapper
├── infer_tensor_meta.rs # InferTensorMeta, InferDims
├── user_meta.rs     # UserMeta: NvDsUserMeta wrapper
└── error.rs         # DeepStreamError
```

## Metadata Hierarchy

```
GstBuffer
  └── NvDsBatchMeta (BatchMeta)
        ├── frame_meta_list → NvDsFrameMeta (FrameMeta)
        │     ├── obj_meta_list → NvDsObjectMeta (ObjectMeta)
        │     │     └── obj_user_meta_list → NvDsUserMeta (UserMeta)
        │     └── frame_user_meta_list → NvDsUserMeta (UserMeta)
        │           └── may contain NvDsInferTensorMeta (InferTensorMeta)
        │               when meta_type == NVDSINFER_TENSOR_OUTPUT_META
        └── user_meta_list → NvDsUserMeta (UserMeta)
```

`InferTensorMeta` is **not** directly attached to frames — it is stored as
user data inside `NvDsUserMeta` entries on the frame's `frame_user_meta_list`.
Use `UserMeta::as_infer_tensor_meta()` to extract it.

## Lock Model

`BatchMeta` holds an `Arc<BatchMetaLock>` which acquires
`nvds_acquire_meta_lock` on construction and releases via
`nvds_release_meta_lock` on drop. The lock is held for the lifetime of
every `BatchMeta` (and its clones) to guarantee safe traversal of the
metadata linked lists.

## FFI Layer

The crate uses `deepstream-sys` for raw bindings. Key types:
- `NvDsBatchMeta`, `NvDsFrameMeta`, `NvDsObjectMeta`
- `NvDsInferTensorMeta`, `NvDsInferDims`, `NvDsInferLayerInfo`
- `gst_buffer_get_nvds_batch_meta`, `nvds_acquire_meta_lock`, etc.

## API Registration

`gst_buffer_get_nvds_batch_meta` reads a file-scoped GQuark that is only
set when `nvds_meta_api_get_type()` is called. The crate calls this
(via `ensure_nvds_meta_api_registered`) before any batch meta access to
avoid `gst_meta_api_type_has_tag: assertion 'tag != 0' failed`.

## Dependencies

| Crate | Role |
|---|---|
| `deepstream-sys` | Raw FFI bindings to DeepStream C API (local path) |
| `glib` | GLib types, used for `From<glib::Error>` conversion |
| `thiserror` | Derive macro for `DeepStreamError` |
| `log` | Logging unsupported data types in `InferTensorMeta::layer_names` |

## Usage Context

- **nvinfer** uses `BatchMeta::from_gst_buffer` when extracting inference
  output from appsink samples. It also uses `InferTensorMeta`, `InferDims`
  for tensor parsing.
- **batch_meta_builder** (nvinfer) uses `deepstream-sys` directly for
  `nvds_acquire_frame_meta_from_pool`, `nvds_add_obj_meta_to_frame`, etc.
