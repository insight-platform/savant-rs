# NvTracker — Python API (`savant_rs.nvtracker`)

All symbols require the **`deepstream`** build feature.

## `TrackingIdResetMode`

PyO3 enum; `int(mode)` matches DeepStream `tracking-id-reset-mode`.

Class attributes: `NONE`, `ON_STREAM_RESET`, `ON_EOS`, `ON_STREAM_RESET_AND_EOS`.

## `TrackState`

Maps tracker misc state: `EMPTY`, `ACTIVE`, `INACTIVE`, `TENTATIVE`, `PROJECTED`.

## `NvTrackerConfig`

`SIG: __init__(ll_lib_file: str, ll_config_file: str, input_format: VideoFormat, *, name: str = "", tracker_width: int = 640, tracker_height: int = 384, max_batch_size: int = 16, gpu_id: int = 0, element_properties: Optional[dict[str, str]] = None, tracking_id_reset_mode: TrackingIdResetMode = NONE, queue_depth: int = 0, operation_timeout_ms: int = 30000, input_channel_capacity: int = 16, output_channel_capacity: int = 16, drain_poll_interval_ms: int = 100)`

**Getters:** `ll_lib_file`, `ll_config_file`, `queue_depth`, `operation_timeout_ms`, `input_channel_capacity`, `output_channel_capacity`, `drain_poll_interval_ms`, `name`, `tracker_width`, `tracker_height`, `max_batch_size`, `gpu_id`; `__repr__`.

Rust `validate()` runs inside `NvTracker.__init__` (via `NvTracker::new`).

## `TrackedFrame`

Input frame for tracking. Callers build one per source frame; the tracker assembles them into a `NonUniformBatch` internally.

`SIG: __init__(source: str, buffer: SharedBuffer, rois: Dict[int, List[Roi]])`

| Parameter | Type | Notes |
|-----------|------|-------|
| `source` | `str` | Stream name (e.g. `"cam-1"`) |
| `buffer` | `SharedBuffer` | Single-surface NVMM buffer (consumed on construction) |
| `rois` | `Dict[int, List[Roi]]` | Detections keyed by `class_id` |

**Properties:** `source` (read-only).

`__repr__`: `TrackedFrame(source="cam-1", classes=2, rois=5)`.

## Output types (read-only properties)

### `TrackedObject`

`object_id`, `class_id`, `bbox_left`, `bbox_top`, `bbox_width`, `bbox_height`, `confidence`, `tracker_confidence`, `label`, `slot_number`, `source_id`.

### `MiscTrackFrame`

`frame_num`, bbox fields, `confidence`, `age`, `state` (`TrackState`), `visibility`.

### `MiscTrackData`

`object_id`, `class_id`, `label`, `source_id`, `frames` (`List[MiscTrackFrame]`).

### `TrackerOutput`

`current_tracks`, `shadow_tracks`, `terminated_tracks`, `past_frame_data`.

`SIG: def buffer(self) -> SharedBuffer` — returns a Python wrapper holding a clone of the inner `SharedBuffer`.

`__repr__` summarizes list lengths.

## `NvTrackerOutput`

Discriminated union from `recv` / `recv_timeout` / `try_recv` (mirrors `NvInferOutput`).

**Properties:** `is_tracking`, `is_event`, `is_eos`, `is_error`, `event_summary`, `eos_source_id`, `error_message`.

**Methods:** `as_tracking() -> Optional[TrackerOutput]`.

## `NvTracker`

Pipeline: `appsrc → nvtracker → appsink` via the shared GStreamer framework (Playing on `new`). Optional queue from `queue_depth`.

`SIG: __init__(config: NvTrackerConfig)`

| Method | Signature | Notes |
|--------|-----------|-------|
| `submit(frames, ids)` | `List[TrackedFrame]`, `List[Tuple[SavantIdMetaKind, int]]` | GIL released; backpressure if input channel full |
| `recv()` | `-> NvTrackerOutput` | Blocks; `RuntimeError` on channel disconnect |
| `recv_timeout(timeout_ms)` | `-> Optional[NvTrackerOutput]` | `None` on timeout |
| `try_recv()` | `-> Optional[NvTrackerOutput]` | |
| `send_eos(source_id)` | Logical per-source EOS (custom downstream event) |
| `send_custom_downstream_event(structure_name, string_fields=None)` | String-keyed custom downstream event |
| `is_failed()` | `-> bool` |
| `reset_stream(source_id: str)` | Stream reset + internal counters |
| `graceful_shutdown(timeout_ms: int)` | `-> List[NvTrackerOutput]`; GIL released; takes inner engine |
| `shutdown()` | Abrupt shutdown; takes inner engine; second call raises `RuntimeError` |

`__repr__`: `NvTracker(running)` / `NvTracker(shut_down)`.

## `NvTrackerBatchingOperatorConfig`

`SIG: __init__(max_batch_size: int, max_batch_wait_ms: int, nvtracker_config: NvTrackerConfig)`

Read-only properties:

- `max_batch_size`
- `max_batch_wait_ms`
- `nvtracker_config`

## `TrackerBatchFormationResult`

`SIG: __init__(ids: List[Tuple[SavantIdMetaKind, int]], rois: List[Dict[int, List[Roi]]])`

- `ids` can be empty (operator always injects internal `Batch(batch_id)`).
- `rois` length must match the number of input `VideoFrame` values.

Read-only property:

- `ids`

## `TrackerOperatorFrameOutput`

Per-frame callback view:

- `frame: VideoFrame`
- `tracked_objects: List[TrackedObject]`
- `shadow_tracks: List[MiscTrackData]`
- `terminated_tracks: List[MiscTrackData]`
- `past_frame_data: List[MiscTrackData]`

`shadow/terminated/past` lists are pre-grouped by `frame.source_id`.

## `SealedDeliveries`

Represents sealed `List[Tuple[VideoFrame, SharedBuffer]]` released when parent output is dropped.

Methods:

- `__len__()`
- `is_empty()`
- `is_released()`
- `unseal(timeout_ms: Optional[int] = None) -> List[Tuple[VideoFrame, SharedBuffer]]`
- `try_unseal() -> Optional[List[Tuple[VideoFrame, SharedBuffer]]]`

## `TrackerOperatorOutput`

Callback payload: tracking batch, logical EOS, or error.

**Properties:** `is_tracking`, `is_eos`, `is_error`, `eos_source_id`, `error_message`, `frames`, `num_frames`.

**Methods:** `take_deliveries() -> Optional[SealedDeliveries]` (only for `is_tracking`; otherwise `None`).

## `NvTrackerBatchingOperator`

High-level frame+buffer batching layer over `NvTracker`.

`SIG: __init__(config: NvTrackerBatchingOperatorConfig, batch_formation_callback: Callable[[List[VideoFrame]], TrackerBatchFormationResult], result_callback: Callable[[TrackerOperatorOutput], None])`

Methods:

- `add_frame(frame: VideoFrame, buffer: SharedBuffer | int)`
- `flush()`
- `reset_stream(source_id: str)`
- `send_eos(source_id: str)` — forwards to inner `NvTracker.send_eos`
- `graceful_shutdown(timeout_ms: int) -> List[TrackerOperatorOutput]` — GIL released
- `shutdown()`
