# NvTracker — Public API (Rust)

Crate: `nvtracker`

## Core exports

- Runtime: `NvTracker`, `NvTrackerOutput`, `TrackedFrame`
- Config: `NvTrackerConfig`, `TrackingIdResetMode`
- Data model: `Roi`, `TrackerOutput`, `TrackedObject`, `MiscTrackData`, `MiscTrackFrame`, `TrackState`
- Helpers: `attach_detection_meta`, `extract_tracker_output`, `default_ll_lib_path`
- Batching layer: `NvTrackerBatchingOperator` and related config/output types

## Runtime API

`NvTracker` constructor:

- `NvTracker::new(config: NvTrackerConfig) -> Result<NvTracker>`

I/O methods:

- `submit(&self, frames: &[TrackedFrame], ids: Vec<SavantIdMetaKind>) -> Result<()>`
- `recv(&self) -> Result<NvTrackerOutput>`
- `recv_timeout(&self, timeout: Duration) -> Result<Option<NvTrackerOutput>>`
- `try_recv(&self) -> Result<Option<NvTrackerOutput>>`
- `send_event(&self, event: gst::Event) -> Result<()>`
- `send_eos(&self, source_id: &str) -> Result<()>`
- `reset_stream(&self, source_id: &str) -> Result<()>`
- `is_failed(&self) -> bool`
- `graceful_shutdown(&self, timeout: Duration) -> Result<Vec<NvTrackerOutput>>`
- `shutdown(&self) -> Result<()>`

`NvTrackerOutput`:

```rust
pub enum NvTrackerOutput {
    Tracking(TrackerOutput),
    Event(gst::Event),
    Eos { source_id: String },
    Error(NvTrackerError),
}
```

## Input model

`TrackedFrame`:

- `source: String`
- `buffer: SharedBuffer`
- `rois: HashMap<i32, Vec<Roi>>`

`Roi`:

- `id: i64`
- `bbox: RBBox`

## Output data model (numeric widths)

All tracker output ids and counters use `i64` on savant-owned structs.
DeepStream-native narrow widths (`u16 class_id`, `u32 batch_id`,
`u32 frame_num`, `u32 age`, `i32 obj.class_id`) are widened **losslessly**
at the FFI seam inside `extract_tracker_output` / `append_misc_batch`.

`TrackedObject`:

- `object_id: u64`, `class_id: i64`, `slot_number: i64`
- `bbox_left / top / width / height: f32`
- `confidence: f32`, `tracker_confidence: f32`
- `label: Option<String>`, `source_id: String`

`MiscTrackData`:

- `object_id: u64`, `class_id: i64`
- `label: Option<String>`, `source_id: String`
- `category: MiscTrackCategory`, `frames: Vec<MiscTrackFrame>`

`MiscTrackFrame`:

- `frame_num: i64`, `age: i64`
- `bbox_*: f32`, `confidence: f32`, `visibility: f32`
- `state: TrackState`

See `kb/savant-core/patterns.md` § *Numeric widths on public structs*
for the cross-crate convention.

## Config model

`NvTrackerConfig::new(ll_lib_file, ll_config_file)` with defaults for dimensions, channel capacities, timeout and format.

Mutable fields include:

- `name`, `tracker_width`, `tracker_height`, `max_batch_size`
- `gpu_id`, `input_format`, `element_properties`
- `tracking_id_reset_mode`, `operation_timeout`
- `input_channel_capacity`, `output_channel_capacity`, `drain_poll_interval`

Validation:

- `validate(&self) -> Result<()>`

`TrackingIdResetMode` values:

- `None`
- `OnStreamReset`
- `OnEos`
- `OnStreamResetAndEos`

## Batching operator API

`NvTrackerBatchingOperator` callbacks:

- `TrackerBatchFormationCallback = Arc<dyn Fn(&[VideoFrameProxy]) -> TrackerBatchFormationResult + Send + Sync>`
- `TrackerOperatorResultCallback = Box<dyn FnMut(TrackerOperatorOutput) + Send>`

Runtime methods:

- `new(config, batch_formation, result_callback) -> Result<Self>`
- `add_frame`, `flush`, `send_eos`, `reset_stream`, `graceful_shutdown`, `shutdown`

Output model:

- `TrackerOperatorOutput::{Tracking(TrackerOperatorTrackingOutput), Eos { source_id }, Error(NvTrackerError)}`
- `TrackerOperatorTrackingOutput::{frames, take_deliveries}`
- `SealedDeliveries::{len, is_empty, is_released, unseal, unseal_timeout, try_unseal}`

## Errors

See `errors.md` for full `NvTrackerError` variants and failure semantics.
