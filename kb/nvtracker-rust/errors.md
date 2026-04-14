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
| `BatchFormationFailed(String)` | `message` | Batch formation callback failed |
| `OperatorShutdown` | — | Batching operator is shut down; operation rejected |
| `BatchMetaFailed` | `operation: String`, `detail: String` | Batch metadata error during a specific operation |
| `NullPointer` | `function: String` | Unexpected null pointer from DeepStream FFI |
| `ResolutionMismatch` | `source_id`, `slot_a`, `w_a`, `h_a`, `slot_b`, `w_b`, `h_b` | Same source different resolutions |
| `BBoxConversion` | `roi_id: i64`, `slot: u32`, `reason: String` | `RBBox::as_ltwh()` failed for a ROI |
| `SlotIndexOutOfBounds` | `index: u32`, `num_filled: u32`, `operation: String` | Slot index >= num_filled |
| `FrameNumOverflow` | `pad_index: u32`, `source_id: String` | i32 counter overflow |
| `PipelineFailed` | — | Pipeline entered a terminal failed state (e.g. operation timeout expired, sync or async); the tracker must be recreated |
| `OperatorFailed` | — | The batching operator's inner tracker pipeline has failed |
| `BufferNotWritable` | `operation: String` | `get_mut()` returned None |
| `BufferOwnership` | `operation: String` | `into_buffer()` failed (outstanding refs) |
| `DeepStream(DeepStreamError)` | inner error | From deepstream crate (`#[from]`) |
| `FrameworkError(PipelineError)` | inner error | From `savant_gstreamer::pipeline` (`#[from]`) |
| `ChannelDisconnected` | — | Input/output channel closed (`submit` / `recv` / etc.) |
| `ShuttingDown` | — | After `graceful_shutdown` begins: `submit`, `send_event`, `send_eos`, `reset_stream` |

Helper: `NvTrackerError::batch_meta(operation, detail)` constructs `BatchMetaFailed` with `Into<String>` arguments.

`DeepStreamError` originates from safe wrappers in `savant_deepstream/deepstream` (e.g. batch meta, tracker misc parsing).
