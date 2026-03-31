# NvTracker Python — Patterns

## Pytest location

`savant_python/pytests/test_nvtracker.py`

- Skips entirely if `savant_rs` was built without deepstream (`ImportError`).
- All seven tests pass when assets and hardware are available:

| Test | Pattern | What it checks |
|------|---------|----------------|
| `test_enums_and_track_state` | — | Enum sanity |
| `test_nv_tracker_config_paths` | — | Config asset check |
| `test_single_source_tracking_py` | A (single-source) | Single source E2E + ID stability |
| `test_multi_source_py` | B (multi-source) | Two-stream isolation |
| `test_same_source_multi_frame_py` | C (temporal batch) | Two frames from same source |
| `test_reset_stream_py` | Stream reset | Stream reset + new IDs |
| `test_class_id_tracking_py` | E (class-ID) | Detections under different `class_id` propagate correctly |

## Config path test

Uses:

- `DEFAULT_LL = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"`
- IOU YAML under `savant_deepstream/nvtracker/assets/config_tracker_IOU.yml` (relative to repo root from pytest file)

## Callback

```python
def _cb(out: TrackerOutput) -> None:
    _ = out.current_tracks
```

Callback is invoked from a GStreamer thread with GIL held. Keep logic minimal.

## Buffer lifecycle

Each `TrackedFrame` wraps a single-surface `SharedBuffer` from `BufferGenerator`. The tracker builds a `NonUniformBatch` internally — callers never construct batches manually.

```python
def _make_frame(
    source: str,
    rois: dict[int, list[Roi]],
    w: int = 320,
    h: int = 240,
) -> TrackedFrame:
    gen = BufferGenerator(
        VideoFormat.RGBA, w, h, gpu_id=0, mem_type=MemType.DEFAULT
    )
    buf = gen.acquire(None)
    return TrackedFrame(source, buf, rois)
```

The `SharedBuffer` is consumed when `TrackedFrame` is constructed. Do not reuse the same buffer.

## Wheel build

Always use **`SAVANT_FEATURES=deepstream make dev install`** from project root so `nvtracker` is linked into `savant_rs`.

## Batching operator pattern

`NvTrackerBatchingOperator` accepts `(VideoFrame, SharedBuffer)` pairs directly (same shape as nvinfer batching operator) and handles frame accumulation internally.

```python
def batch_formation(frames: list[VideoFrame]) -> TrackerBatchFormationResult:
    rois = []
    for frame in frames:
        # Build class-keyed detections for this frame.
        rois.append({0: [Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))]})
    return TrackerBatchFormationResult(ids=[], rois=rois)

def on_tracking(output: TrackerOperatorOutput) -> None:
    for frame_out in output.frames:
        frame = frame_out.frame
        tracks = frame_out.tracked_objects
        # shadow/terminated/past are already grouped for this frame source.
        _ = frame_out.shadow_tracks
        _ = frame_out.terminated_tracks
        _ = frame_out.past_frame_data

    sealed = output.take_deliveries()
    if sealed is None:
        return
    pairs = sealed.unseal()  # wait until output is dropped
    for frame, buffer in pairs:
        # feed downstream stage
        pass
```

### Important constraints

- `TrackerBatchFormationResult.rois` length must exactly match incoming `frames` length.
- `ids` is optional (`[]` is valid); operator injects internal `Batch(batch_id)`.
- `SealedDeliveries.unseal()` releases with GIL detached in Rust, so callback thread can progress safely.

## Stage chaining pattern

The delivery API is intentionally compatible with the nvinfer batching operator:

1. Inference callback: `take_deliveries()` -> `unseal()` -> pass pairs downstream.
2. Tracking callback: `take_deliveries()` -> `unseal()` -> pass pairs to next stage.

This keeps `(VideoFrame, SharedBuffer)` ownership explicit across pipeline stages.
