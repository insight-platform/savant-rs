# NvTracker — Patterns & Tests

## Asset: IOU low-level config

Path: `savant_deepstream/nvtracker/assets/config_tracker_IOU.yml`

Minimal multi-object tracker YAML for tests. **`probationAge`** is currently **`0`** in-tree (faster activation during development); the original plan used `1` — adjust if you need stricter probation semantics.

Point `NvTrackerConfig::ll_config_file` at this file and `ll_lib_file` at  
`/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so` (or `default_ll_lib_path()`).

## Unit tests

- `config::tests::validate_rejects_missing_files`
- `config::tests::validate_rejects_zero_dimensions` (uses temp files)

## Integration tests

| Test binary | What it checks |
|-------------|----------------|
| `test_detection_meta_count` | After `attach_detection_meta`, object count in batch meta (no live `nvtracker` needed) |
| `test_iou_tracker` | E2E IOU tracker suite (all run by default, not `#[ignore]`) |

**`test_iou_tracker` cases**

| Test | Pattern | What it checks |
|------|---------|----------------|
| `test_single_source_id_stability` | A (single-source) | ID persistence across frames |
| `test_multi_source_isolation` | B (multi-source) | Independent IDs per stream |
| `test_multi_source_nonuniform` | E (heterogeneous) | Different resolutions per source via `TrackedFrame` |
| `test_same_source_multi_frame` | C (temporal batch) | Two frames from same source in one call |
| `test_mixed_batch` | D (mixed) | Multi-source + multi-frame + heterogeneous resolutions |
| `test_class_id_tracking` | E (class-ID) | Detections under different `class_id` propagate correctly |
| `test_reset_stream_reassigns_ids` | Stream reset | Stream reset gives new IDs |
| `test_reset_stream_only_affects_target` | Stream reset | Reset only resets the target stream |
| `test_shadow_tracks_on_disappearance` | A (single-source) | Object removal from `current_tracks` |
| `test_source_id_roundtrip` | A+B | `source_id` preserved in output |
| `test_misc_track_state_mapping` | — | Enum sanity check |
| `test_mux_reference_equivalence` | `#[ignore]` | Diagnostic placeholder for mux comparison |

Shadow / terminated / past-frame misc lists are **not** populated by the IOU tracker in DS 7.1. Tests verify object disappearance from `current_tracks` only.

The IOU tracker assigns object IDs starting from **0**. Use `Option<u64>` when checking if a track was found (do not use `0` as a sentinel).

Run default suite:

```bash
cargo test -p nvtracker
```

Run E2E IOU tests (use single thread — GStreamer / GPU):

```bash
cargo test -p nvtracker --test test_iou_tracker -- --test-threads=1
```

## Buffer prep (GPU)

Tests use `BufferGenerator` to create single-surface NVMM buffers. Each `TrackedFrame` wraps one such buffer. The tracker builds a `NonUniformBatch` internally — callers never construct batches manually.

```rust
fn make_buffer(w: u32, h: u32) -> SharedBuffer {
    BufferGenerator::new(
        VideoFormat::RGBA, w, h, 30, 1, 0, NvBufSurfaceMemType::Default,
    ).unwrap()
    .acquire(None)
    .unwrap()
}
```

Frame order in the `&[TrackedFrame]` slice determines the surface slot order in the internal batch.

## `SharedBuffer` ownership

Each `TrackedFrame` holds one `SharedBuffer`. The tracker consumes them during `prepare_batch`. Drop all external `SurfaceView` / batch handles before `track`/`track_sync` so `into_buffer()` succeeds. Otherwise: `BufferOwnership` error.

## Sync timeout

`track_sync` waits up to `operation_timeout` (default **30 seconds**, configurable via `NvTrackerConfig::operation_timeout`). When the timeout expires — whether in `track_sync` or detected by the watchdog thread for async buffers — the pipeline enters a terminal failed state and returns `PipelineFailed`. The tracker instance must be recreated after this error.
