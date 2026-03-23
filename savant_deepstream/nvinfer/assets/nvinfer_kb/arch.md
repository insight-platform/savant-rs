# NvInfer Architecture

## Module Tree
```
nvinfer/src/
├── lib.rs              # Re-exports
├── pipeline.rs         # NvInfer: GStreamer pipeline, submit, infer_sync
├── config.rs           # NvInferConfig: properties, validate_and_materialize
├── error.rs            # NvInferError, Result
├── output.rs           # BatchInferenceOutput, ElementOutput, TensorView
├── roi.rs              # Roi
├── meta_clear_policy.rs# MetaClearPolicy
├── nvinfer_types.rs    # DataType
└── batch_meta_builder.rs # attach_batch_meta_with_rois, clear_all_frame_objects (pub(crate))
```

## Pipeline Layout

```
appsrc (NVMM, video/x-raw) ! [queue] ! nvinfer ! appsink
```

- `queue_depth > 0`: queue inserted with `max-size-buffers = queue_depth`
- `queue_depth == 0`: no queue (synchronous)

## Data Flow

### submit (async)
```
SharedBuffer.into_buffer() → gst::Buffer (sole owner required)
  → attach_batch_meta_with_rois (ROIs or full-frame sentinel)
  → set PTS (monotonic counter)
  → appsrc.push_buffer()
  → nvinfer processes
  → appsink callback → InferCallback(BatchInferenceOutput)
```

### infer_sync
```
Same as submit, but:
  → register PTS in sync_tx map
  → block on mpsc::channel recv_timeout(30s)
  → return BatchInferenceOutput when appsink delivers
```

## ROI Handling

| Scenario | Behaviour |
|---|---|
| `rois = None` and fixed input dims | Synthetic full-frame ROIs per slot `(0,0,w,h)` |
| `rois = None` and flexible dims | Read slot dimensions from NvBufSurface; synthetic ROIs per slot |
| `rois = Some(map)` | One `NvDsObjectMeta` per `Roi`; slot with no entry → full-frame sentinel |

Full-frame sentinel uses `unique_component_id = FULL_FRAME_SENTINEL` (-1);
`ElementOutput::roi_id` is `None` for full-frame results.
`ElementOutput::slot_number` is `NvDsFrameMeta.batch_id` (surface slot). User ids
come from `SavantIdMeta` on `BatchInferenceOutput::buffer()` (`savant_ids()`), not
from `ElementOutput`.

## SavantIdMeta Bridge

`bridge_savant_id_meta(&nvinfer)` is called at construction. PTS-keyed
propagation ensures output buffers carry per-frame IDs. `BatchInferenceOutput::buffer()`
returns `SharedBuffer` with `savant_ids()` available.

Regression: `tests/test_memory.rs` (`test_nonuniform_slot_numbers`) asserts that
`SharedBuffer::savant_ids()` on the batch submitted to `infer_sync` matches
`output.buffer().savant_ids()` afterward (multi-slot non-uniform batch).

## Oversized batches vs engine `batch-size`

`gstnvinfer` may run multiple internal TensorRT passes when the number of
frames / ROIs in a single buffer exceeds the model’s configured `batch-size`.
The Rust `NvInfer` pipeline does not split buffers; it forwards the full batch.

Regression: `tests/test_oversized_batch.rs` uses the age/gender model with
`batch-size=1` and submits two frames with four ROIs each (eight inferences),
validating outputs and `SavantIdMeta` propagation for both uniform and
non-uniform batches.

## Config File

`validate_and_materialize()` writes config to a `NamedTempFile`. The file
must stay alive (NvInfer holds it via `_config_file`). Keys with `section.key`
are grouped under `[section]`; bare keys go to `[property]`.

## Jetson Scaling

On aarch64, small surfaces (< 16×16) can fail with VIC. Tests inject
`scaling-compute-hw=1` into nvinfer properties to force GPU compute.
