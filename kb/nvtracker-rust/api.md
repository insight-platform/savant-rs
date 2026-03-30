# NvTracker — Public API (Rust)

Crate: `nvtracker` (`savant_deepstream/nvtracker`)

## Top-level exports

```rust
pub use config::{NvTrackerConfig, TrackingIdResetMode};
pub use deepstream_buffers::SavantIdMetaKind;
pub use detection_meta::attach_detection_meta;
pub use error::{NvTrackerError, Result};
pub use output::{
    extract_tracker_output, MiscTrackData, MiscTrackFrame, TrackState, TrackedObject, TrackerOutput,
};
pub use pipeline::{default_ll_lib_path, NvTracker, TrackedFrame, TrackerCallback};
pub use roi::Roi;
```

`TrackState` is re-exported from the `deepstream` crate (maps DeepStream `TRACKER_STATE`).
`SavantIdMetaKind` is re-exported from `deepstream_buffers`.

---

## Constants

| Name | Type | Value | Notes |
|------|------|-------|-------|
| `DEFAULT_TRACKER_WIDTH` | `u32` | `640` | `tracker-width` default |
| `DEFAULT_TRACKER_HEIGHT` | `u32` | `384` | `tracker-height` default |
| `DEFAULT_MAX_BATCH_SIZE` | `u32` | `16` | Default `max_batch_size` for batch queries + meta sizing |

---

## `TrackingIdResetMode`

`#[repr(u32)]` — matches `nvtracker` `tracking-id-reset-mode`.

| Variant | Discriminant |
|---------|--------------|
| `None` | 0 |
| `OnStreamReset` | 1 |
| `OnEos` | 2 |
| `OnStreamResetAndEos` | 3 |

`SIG: fn as_u32(self) -> u32`

---

## `NvTrackerConfig`

| Field | Type | Default (via `new`) |
|-------|------|---------------------|
| `name` | `String` | `""` |
| `tracker_width` | `u32` | `DEFAULT_TRACKER_WIDTH` |
| `tracker_height` | `u32` | `DEFAULT_TRACKER_HEIGHT` |
| `ll_lib_file` | `String` | required |
| `ll_config_file` | `String` | required |
| `gpu_id` | `u32` | `0` |
| `input_format` | `VideoFormat` (buffers crate) | `RGBA` |
| `element_properties` | `HashMap<String, String>` | empty |
| `tracking_id_reset_mode` | `TrackingIdResetMode` | `None` |
| `max_batch_size` | `u32` | `DEFAULT_MAX_BATCH_SIZE` |

`SIG: fn new(ll_lib_file, ll_config_file) -> Self`

`SIG: fn validate(&self) -> Result<()>` — requires existing files for lib + YAML; non-zero width/height; non-zero max_batch_size.

---

## `Roi`

Detection input (same shape as nvinfer ROI for interop).

| Field | Type |
|-------|------|
| `id` | `i64` — stored in `misc_obj_info[0]` on `NvDsObjectMeta` |
| `bbox` | `RBBox` (savant_core) |

---

## `TrackedFrame`

Input frame for tracking. Callers build one per source frame; the tracker assembles them into a `NonUniformBatch` internally.

| Field | Type | Notes |
|-------|------|-------|
| `source` | `String` | Stream name (e.g. `"cam-1"`); hashed to `pad_index` via crc32 |
| `buffer` | `SharedBuffer` | Single-surface NVMM buffer from `BufferGenerator` |
| `rois` | `HashMap<i32, Vec<Roi>>` | Detections keyed by `class_id` (`i32` matches `NvDsObjectMeta.class_id`) |

---

## `NvTracker`

Pipeline: `appsrc → nvtracker → appsink` (Playing on `new`).

`pub type TrackerCallback = Box<dyn FnMut(TrackerOutput) + Send>;`

| Method | Signature | Notes |
|--------|-----------|-------|
| `new` | `(config: NvTrackerConfig, callback: TrackerCallback) -> Result<Self>` | Validates config, inits GST, installs pad probe for batch-size queries |
| `track` | `(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()>` | Async; builds `NonUniformBatch` internally from frame buffers |
| `track_sync` | `(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<TrackerOutput>` | Blocks until output or 30 s timeout |
| `reset_stream` | `(&self, source_id: &str) -> Result<()>` | Sends `GST_NVEVENT_STREAM_RESET` on `appsrc` with pad id `crc32(source_id)`; resets frame counter |
| `shutdown` | `(&mut self) -> Result<()>` | EOS + drain bus + `Null`; idempotent (safe to call twice) |

`SIG: pub fn default_ll_lib_path() -> String` — typical DeepStream `.so` path.

---

## `attach_detection_meta`

`SIG: fn attach_detection_meta(buf: &mut gst::BufferRef, num_filled: u32, max_batch_size: u32, slots: &[(u32, Vec<(i32, Roi)>)], frame_nums: &[i32]) -> Result<()>`

- One `NvDsFrameMeta` per filled slot; `pad_index` = first tuple element (caller uses crc32 of `source_id`).
- Each `(class_id, Roi)` → `NvDsObjectMeta` with pre-track id `u64::MAX`, the given `class_id`, `confidence = 1.0`.
- Sets `bInferDone = 1` on each `NvDsFrameMeta` so the low-level tracker processes the detections.
- Sets `surface_index = 0` explicitly on every frame.
- Empty ROI list per slot is allowed (frame meta created, no objects attached).

---

## `extract_tracker_output`

`SIG: fn extract_tracker_output(buffer: gst::Buffer, resolve_source_id: impl Fn(u32) -> String) -> Result<TrackerOutput>`

Walks batch meta for current objects and batch user meta for shadow / terminated / past-frame lists. Wraps outgoing buffer in `SharedBuffer` (shallow ref).

---

## Output structs

### `TrackedObject`

`object_id`, `class_id`, `bbox_*`, `confidence`, `tracker_confidence`, `label`, `slot_number`, `source_id`.

### `MiscTrackFrame`

`frame_num`, bbox, `confidence`, `age`, `state: TrackState`, `visibility`.

### `MiscTrackData`

`object_id`, `class_id`, `label`, `source_id`, `frames: Vec<MiscTrackFrame>`.

### `TrackerOutput`

`buffer: SharedBuffer`, `current_tracks`, `shadow_tracks`, `terminated_tracks`, `past_frame_data`.
