# NvTracker — Python Usage Guide

Comprehensive code samples for all batch patterns. Every example assumes DeepStream 7.1, GPU 0, and the IOU low-level tracker.

## Prerequisites

```python
from savant_rs.deepstream import (
    BufferGenerator,
    MemType,
    SavantIdMetaKind,
    SharedBuffer,
    VideoFormat,
    init_cuda,
)
from savant_rs.nvinfer import Roi
from savant_rs.nvtracker import (
    NvTracker,
    NvTrackerConfig,
    TrackedFrame,
    TrackingIdResetMode,
    TrackState,
    TrackerOutput,
    TrackedObject,
)
from savant_rs.primitives.geometry import RBBox
```

Build the wheel with DeepStream support:

```bash
SAVANT_FEATURES=deepstream make dev install
```

## CUDA initialization

Call once before creating any buffers:

```python
init_cuda(0)
```

## Configuration

```python
cfg = NvTrackerConfig(
    ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    ll_config_file="/path/to/config_tracker_IOU.yml",
    input_format=VideoFormat.RGBA,
    tracker_width=320,
    tracker_height=240,
    max_batch_size=4,
    tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET,
    queue_depth=0,  # 0 = synchronous; >0 = insert GStreamer queue
)
```

`max_batch_size` must be >= the maximum number of frames in any single `track`/`track_sync` call. It controls the pad probe response to DeepStream's batch-size queries.

`queue_depth` inserts a GStreamer `queue` element between `appsrc` and `nvtracker` when set to a value greater than zero. This decouples the push thread from tracker processing, which can help absorb latency spikes. Set to `0` (default) for synchronous operation.

> **Note:** The `sub-batches` element property is **rejected** by `NvTracker`. Each instance is its own isolated tracker — create separate `NvTracker` instances for different tracking workloads instead of sub-batching within one element.

## Creating the tracker

```python
def on_output(output: TrackerOutput) -> None:
    for t in output.current_tracks:
        print(f"{t.source_id}: obj {t.object_id} class {t.class_id} at "
              f"({t.bbox_left},{t.bbox_top} {t.bbox_width}x{t.bbox_height})")

tracker = NvTracker(cfg, on_output)
```

## Buffer helper

Each `TrackedFrame` requires a single-surface `SharedBuffer`. Use `BufferGenerator`:

```python
def make_frame(
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

The tracker builds a `NonUniformBatch` internally from the per-frame buffers — callers never need to construct batches manually.

Always use keyword arguments for `BufferGenerator` after the width/height positional args. Use `TransformConfig()` where needed (there is no `.default()` method in the Python binding).

## Pattern A: Single-source tracking

```python
frame = make_frame("cam-1", {
    0: [
        Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0)),
        Roi(2, RBBox.ltwh(180.0, 100.0, 70.0, 70.0)),
    ],
})
ids = [(SavantIdMetaKind.FRAME, 1)]
out = tracker.track_sync([frame], ids)
assert len(out.current_tracks) == 2
id_a = out.current_tracks[0].object_id
id_b = out.current_tracks[1].object_id

# Second frame — IDs persist
frame2 = make_frame("cam-1", {
    0: [
        Roi(1, RBBox.ltwh(42.0, 40.0, 80.0, 60.0)),
        Roi(2, RBBox.ltwh(182.0, 100.0, 70.0, 70.0)),
    ],
})
out2 = tracker.track_sync([frame2], [(SavantIdMetaKind.FRAME, 2)])
ids_out = {t.object_id for t in out2.current_tracks}
assert id_a in ids_out and id_b in ids_out
```

## Pattern B: Multi-source (stream isolation)

Two cameras in one call. Each gets independent track IDs.

```python
roi = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
frame_a = make_frame("cam-a", {0: [roi]})
frame_b = make_frame("cam-b", {0: [roi]})
out = tracker.track_sync(
    [frame_a, frame_b],
    [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
)
assert len(out.current_tracks) == 2
by_src = {t.source_id: t.object_id for t in out.current_tracks}
assert by_src["cam-a"] != by_src["cam-b"]
```

## Pattern C: Multi-frame per source (temporal batch)

Two frames from the **same** camera in one call. The tracker processes them sequentially.

```python
r0 = Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))
r1 = Roi(1, RBBox.ltwh(55.0, 52.0, 60.0, 60.0))
f0 = make_frame("cam-a", {0: [r0]})
f1 = make_frame("cam-a", {0: [r1]})
out = tracker.track_sync(
    [f0, f1],
    [(SavantIdMetaKind.FRAME, 1), (SavantIdMetaKind.FRAME, 2)],
)
assert len(out.current_tracks) >= 1
```

## Pattern D: Mixed batch (multi-source + multi-frame, heterogeneous resolutions)

Four frames: cam-a×2 (320×240) + cam-b×2 (640×480). The tracker builds a `NonUniformBatch` internally.

```python
frames = [
    make_frame("cam-a", {0: [Roi(1, RBBox.ltwh(40.0, 40.0, 50.0, 50.0))]}, 320, 240),
    make_frame("cam-a", {0: [Roi(1, RBBox.ltwh(45.0, 45.0, 50.0, 50.0))]}, 320, 240),
    make_frame("cam-b", {0: [Roi(1, RBBox.ltwh(200.0, 200.0, 100.0, 100.0))]}, 640, 480),
    make_frame("cam-b", {0: [Roi(1, RBBox.ltwh(210.0, 210.0, 100.0, 100.0))]}, 640, 480),
]
ids = [(SavantIdMetaKind.FRAME, i) for i in range(1, 5)]
out = tracker.track_sync(frames, ids)
assert len(out.current_tracks) == 4
```

Frame order matters: it determines the surface slot order. Frames for the same source must have the same resolution.

## Pattern E: Class-ID tracking

Detections grouped by `class_id` are propagated to the tracker. Each `TrackedObject` carries the specified `class_id`.

```python
frame = make_frame("cam-1", {
    0: [Roi(1, RBBox.ltwh(40.0, 40.0, 80.0, 60.0))],
    1: [Roi(2, RBBox.ltwh(180.0, 100.0, 70.0, 70.0))],
})
out = tracker.track_sync([frame], [(SavantIdMetaKind.FRAME, 1)])
class_ids = {t.class_id for t in out.current_tracks}
assert class_ids == {0, 1}
```

## Stream reset

Reset clears the tracker's internal state for one stream. After reset (with `TrackingIdResetMode.ON_STREAM_RESET`), the same object receives a new track ID.

```python
# Tracker created with tracking_id_reset_mode=TrackingIdResetMode.ON_STREAM_RESET
roi = Roi(1, RBBox.ltwh(60.0, 60.0, 90.0, 90.0))
ids = [(SavantIdMetaKind.FRAME, 1)]

o0 = tracker.track_sync([make_frame("cam-1", {0: [roi]})], ids)
id_before = o0.current_tracks[0].object_id

# Confirm stability
o1 = tracker.track_sync([make_frame("cam-1", {0: [roi]})], ids)
assert o1.current_tracks[0].object_id == id_before

# Reset
tracker.reset_stream("cam-1")

# New ID assigned
o2 = tracker.track_sync([make_frame("cam-1", {0: [roi]})], ids)
assert o2.current_tracks[0].object_id != id_before
```

In a multi-source scenario, resetting one stream does not affect other streams.

## Output consumption

```python
output: TrackerOutput = tracker.track_sync(frames, ids)

# Current tracked objects
for obj in output.current_tracks:
    print(f"source={obj.source_id} id={obj.object_id} class={obj.class_id} "
          f"bbox=({obj.bbox_left},{obj.bbox_top} {obj.bbox_width}x{obj.bbox_height}) "
          f"conf={obj.confidence:.2f} tracker_conf={obj.tracker_confidence:.2f}")

# Shadow / terminated / past-frame (NvDCF/DeepSORT only; empty with IOU tracker)
for s in output.shadow_tracks:
    print(f"shadow: obj {s.object_id} from {s.source_id}")

for t in output.terminated_tracks:
    print(f"terminated: obj {t.object_id} from {t.source_id}")

# Output buffer
buf: SharedBuffer = output.buffer()
```

## Error handling

All errors surface as `RuntimeError` with self-explaining messages from structured Rust error variants.

```python
try:
    out = tracker.track_sync(frames, ids)
except RuntimeError as e:
    msg = str(e)
    if "timed out" in msg:
        print(f"Timeout: {msg}")
    elif "resolution mismatch" in msg:
        print(f"Resolution error: {msg}")
    elif "outstanding references" in msg:
        print(f"Buffer still borrowed: {msg}")
    else:
        print(f"Tracker error: {msg}")
```

## Shutdown

Always shut down the tracker when done. `shutdown()` is idempotent. Further calls after shutdown raise `RuntimeError`.

```python
tracker.shutdown()
```

## IOU tracker limitations (DS 7.1)

- Object IDs start from **0** (not 1).
- Shadow, terminated, and past-frame misc lists are **not** populated by the IOU tracker. Use NvDCF or DeepSORT for these features.
- `probationAge: 0` in the IOU config means objects are immediately confirmed (no tentative state).

## Async tracking

For fire-and-forget tracking, use `track` instead of `track_sync`. Results arrive in the callback:

```python
results = []

def on_output(output: TrackerOutput) -> None:
    results.append(output)

tracker = NvTracker(cfg, on_output)
frame = make_frame("cam-1", {0: [Roi(1, RBBox.ltwh(50.0, 50.0, 60.0, 60.0))]})
tracker.track([frame], [(SavantIdMetaKind.FRAME, 1)])  # non-blocking
# ... results arrive asynchronously in on_output callback
```

The callback runs on a GStreamer thread with the GIL held. Keep callback logic minimal.
