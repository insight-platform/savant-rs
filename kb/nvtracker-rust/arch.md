# NvTracker — Architecture

## Design principle: manual multi-stream is first-class

The `NvTracker` pipeline operates **without `nvstreammux`**. All stream multiplexing, frame numbering, and resolution management are handled explicitly in Rust. This is not a workaround — it is the production design. The manual path gives full control over batching semantics, avoids nvstreammux's opaque scheduling, and supports heterogeneous resolutions natively via `NonUniformBatch`.

## Module layout

| File | Role |
|------|------|
| `config.rs` | `NvTrackerConfig`, `TrackingIdResetMode`, validation |
| `error.rs` | `NvTrackerError`, `Result` |
| `roi.rs` | `Roi` (id + `RBBox`) |
| `detection_meta.rs` | `attach_detection_meta` — build/update `NvDsBatchMeta` |
| `output.rs` | `TrackerOutput` + `extract_tracker_output` |
| `pipeline.rs` | `NvTracker` GStreamer wiring |
| `build.rs` | Links **`nvdsgst_helper`** so `gst_nvevent_new_stream_reset` and custom query FFI resolve |

## Pipeline topology

```
appsrc → nvtracker → appsink
```

No queue element — `nvtracker` is a `GstBaseTransform` operating in-place on the same buffer. No `bridge_savant_id_meta` — metadata (including `SavantIdMeta`) persists on the same buffer throughout.

## Pipeline flow

1. Caller supplies a **batched NVMM** `SharedBuffer` (same family as nvinfer batching): `NonUniformBatch` / surface batch with `num_filled` in the NvBufSurface header.
2. `track` / `track_sync`:
   - Takes exclusive ownership of the buffer (no outstanding views).
   - For each slot: `pad_index = crc32fast::hash(source_id.as_bytes())`, stored in LRU (capacity **4096**) for reverse lookup on output.
   - `attach_detection_meta` writes frames + untracked objects; sets `bInferDone = 1` on each `NvDsFrameMeta`.
   - May set `source_frame_width` / `source_frame_height` from surface list.
   - PTS set from an internal monotonic counter (used to pair sync callbacks).
   - `appsrc.push_buffer`.
3. **appsink** callback:
   - Uses `from_glib_none` on the sample buffer (same refcount pattern as nvinfer, no deep copy) to avoid corrupting batch meta / SIGSEGV.
   - `extract_tracker_output` → async callback or sync channel.

## Batch-size queries (pad probe)

The pipeline installs a pad probe on `appsrc`'s src pad that responds to DeepStream custom queries (`gst_nvquery_is_batch_size`, `gst_nvquery_is_numStreams_size`) with `config.max_batch_size`. Without this, `nvtracker` defaults batch size to **1** and multi-slot batches fail silently (objects get `UNTRACKED_OBJECT_ID` and are removed).

## Mixed batch model

A single batched buffer pushed to `track` supports **mixed** content:

- **Multi-source:** slots from different cameras (different `source_id`).
- **Multi-frame per source:** multiple temporal frames from the same camera in one batch.
- **Mixed:** both patterns in the same batch.

Example 4-slot batch:

| slot | source_id | content  | batch_id | pad_index      | frame_num (global) |
|------|-----------|----------|----------|----------------|---------------------|
| 0    | cam-a     | frame T0 | 0        | crc32("cam-a") | N                   |
| 1    | cam-a     | frame T1 | 1        | crc32("cam-a") | N+1                 |
| 2    | cam-b     | frame T0 | 2        | crc32("cam-b") | M                   |
| 3    | cam-b     | frame T1 | 3        | crc32("cam-b") | M+1                 |

**Resolution constraint:** Slots sharing the same `source_id` must have the same resolution. Different sources may have different resolutions. Use `NonUniformBatch` for heterogeneous resolutions; `UniformBatchGenerator` only when all sources share the same resolution.

## Frame numbering

Per-source frame counters live in `frame_counters: HashMap<u32, i32>`. Each `pad_index` (crc32 of `source_id`) gets a monotonically increasing frame number. `frame_nums` are passed to `attach_detection_meta` and written to `NvDsFrameMeta.frame_num`. Counters reset to 0 on `reset_stream`.

## DeepStream contract invariants

For each `NvDsFrameMeta` in the batch:

| Field                 | Required value                         | Implementation              |
|-----------------------|----------------------------------------|-----------------------------|
| `batch_id`            | slot index (0..N-1)                    | set to `i`                  |
| `frame_num`           | monotonic per source across batches    | global per-source counter   |
| `pad_index`           | `crc32(source_id)`                     | `crc32fast::hash`           |
| `source_id`           | same as `pad_index`                    | same                        |
| `surface_index`       | `0`                                    | explicitly set              |
| `source_frame_width`  | `surfaceList[batch_id].width`          | patched from NvBufSurface   |
| `source_frame_height` | `surfaceList[batch_id].height`         | patched from NvBufSurface   |
| `bInferDone`          | `1`                                    | set in `attach_detection_meta` |

## CUDA cleanup (`ManuallyDrop`)

The `pipeline`, `appsrc`, and `appsink` fields are wrapped in `ManuallyDrop`. After `shutdown()` transitions the pipeline to `Null`, these GStreamer objects are intentionally **not** freed. The DeepStream nvtracker plugin spawns CUDA worker threads during GObject finalization that may never terminate, causing the process to hang. Leaking the wrappers after shutdown is harmless — the OS reclaims all resources on process exit.

## Source ID resolution

- Input: string `source_id` → `pad_index` (crc32) on `NvDsFrameMeta`.
- Output: `pad_index` → string via LRU; if evicted, `"unknown-{pad:#x}"`.
- `reset_stream` hashes `source_id` to the same crc32 and sends NVIDIA stream-reset on `appsrc`.

## Mux reference trace: delta table

The table below documents behavioral differences between the standard `nvstreammux → nvtracker` pipeline and the manual `appsrc → nvtracker` path. The manual path is functionally equivalent for all production scenarios.

| Aspect | nvstreammux path | Manual path | Delta |
|--------|-----------------|-------------|-------|
| Batch size discovery | mux sets `batch-size` property; tracker queries upstream | Pad probe answers `gst_nvquery_batch_size` / `gst_nvquery_numStreams_size` | Equivalent |
| `pad_index` assignment | mux assigns from sink pad index | `crc32(source_id)` | Different values, same semantics (unique per stream) |
| `frame_num` | mux increments per-pad frame count | Global per-source counter in `NvTracker` state | Equivalent semantics |
| `surface_index` | mux sets to 0 | Explicitly set to 0 | Equivalent |
| `bInferDone` | Set by nvinfer in standard pipeline | Set in `attach_detection_meta` | Equivalent |
| `source_frame_width/height` | Copied from surface params by mux | Patched from `NvBufSurface.surfaceList` | Equivalent |
| `NvDsBatchMeta` lifecycle | Created by mux | Created by `attach_detection_meta` via `nvds_create_batch_meta` | Equivalent |
| Stream reset | `GST_NVEVENT_STREAM_RESET` from mux | Same event sent via `appsrc.send_event` | Equivalent |
| `sub-batches` | Configured on tracker element | **Rejected** — each `NvTracker` is its own instance; use separate instances instead | N/A |
| Heterogeneous resolution | Requires mux `width`/`height` + scaling | Native via `NonUniformBatch` — no scaling needed | Manual path is simpler |
| Shadow/terminated lists | Populated by NvDCF/DeepSORT; not by IOU | Same behavior — tracker-lib dependent, not pipeline-dependent | Equivalent |

## Element properties

Extra `nvtracker` GObject properties go through `NvTrackerConfig::element_properties` (`set_property_from_str`). Invalid keys or parse failures → `InvalidProperty`.
