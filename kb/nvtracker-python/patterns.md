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
