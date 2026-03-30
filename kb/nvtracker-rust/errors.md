# NvTracker — Errors (`NvTrackerError`)

`pub type Result<T> = std::result::Result<T, NvTrackerError>;`

| Variant | Fields | When |
|---------|--------|------|
| `GstInit(String)` | `reason` | `gst::init()` failed |
| `ElementCreationFailed` | `element: String`, `reason: String` | Failed to create GStreamer element |
| `PipelineError(String)` | `message` | Generic pipeline errors |
| `LinkFailed` | `chain: String` | Could not link element chain |
| `InvalidProperty` | `key: String`, `reason: String` | Unknown property or parse failure |
| `ConfigError(String)` | `message` | Config validation failure |
| `ResolutionMismatch` | `source_id`, `slot_a`, `w_a`, `h_a`, `slot_b`, `w_b`, `h_b` | Same source different resolutions |
| `BBoxConversion` | `slot_index: u32`, `roi_index: usize`, `message: String` | `RBBox::as_ltwh()` failed |
| `SlotIndexOutOfBounds` | `index: u32`, `num_filled: u32`, `operation: String` | Slot index >= num_filled |
| `FrameNumOverflow` | `pad_index: u32`, `source_id: String` | i32 counter overflow |
| `TrackSyncTimeout` | `timeout_secs: u64`, `pts_key: u64` | `track_sync` timed out |
| `TrackSyncDisconnected` | `pts_key: u64` | Channel disconnected |
| `BufferNotWritable` | `operation: String` | `get_mut()` returned None |
| `BufferOwnership` | `operation: String` | `into_buffer()` failed (outstanding refs) |
| `NullPointer` | `function: String` | Unexpected null pointer |
| `DeepStream(DeepStreamError)` | inner error | From deepstream crate |

`DeepStreamError` originates from safe wrappers in `savant_deepstream/deepstream` (e.g. batch meta, tracker misc parsing).
