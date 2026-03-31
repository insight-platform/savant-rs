# NvInfer Architecture

## Module Tree
```
nvinfer/src/
├── lib.rs                # Re-exports
├── pipeline.rs           # NvInfer: GStreamer pipeline, submit, infer_sync
├── config.rs             # NvInferConfig: properties, validate_and_materialize
├── error.rs              # NvInferError, Result
├── output.rs             # BatchInferenceOutput, ElementOutput, TensorView
├── roi.rs                # Roi, RoiKind
├── model_input_scaling.rs# ModelInputScaling
├── meta_clear_policy.rs  # MetaClearPolicy
├── nvinfer_types.rs      # DataType
├── batch_meta_builder.rs # attach_batch_meta_with_rois, rbbox_to_rect_params (pub(crate))
└── batching_operator/    # Higher-level batching layer
    ├── mod.rs            # Re-exports: NvInferBatchingOperator, OperatorElement, etc.
    ├── config.rs         # NvInferBatchingOperatorConfig
    ├── operator.rs       # NvInferBatchingOperator: frame accumulation, timer, callback
    ├── output.rs         # OperatorElement, OperatorFrameOutput, OperatorInferenceOutput
    ├── scaler.rs         # CoordinateScaler: model→frame coordinate transform
    ├── submit.rs         # SubmitContext: batch formation + NvInfer submission
    ├── types.rs          # PendingBatch, PendingMap, callback type aliases
    └── tests.rs          # Unit + integration tests
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

---

## Batching Operator Layer

`NvInferBatchingOperator` is a higher-level wrapper over `NvInfer` that
accepts individual `(VideoFrameProxy, SharedBuffer)` pairs, accumulates them
into batches, forms a `NonUniformBatch`, and submits to `NvInfer`. Results are
mapped back to original frames via `OperatorInferenceOutput`.

### Batching Operator Data Flow

```
User calls add_frame(frame, buffer)
  → deepstream_buffers::BatchState accumulates (frame, buffer) pairs
  → When batch is full OR timer expires:
      → BatchFormationCallback(frames) → BatchFormationResult { ids, rois }
      → NonUniformBatch formed + finalized
      → PendingBatch stored in PendingMap (frames + rois + model config)
      → NvInfer.submit(batch_buffer, rois_map)
      → NvInfer callback fires with BatchInferenceOutput:
          → Extract PendingBatch from PendingMap by batch_id
          → Group ElementOutputs by slot_number
          → Match each element to its ROI via roi_id → compute clamped rect
          → Wrap each ElementOutput in OperatorElement (with lazy scaler)
          → Build OperatorFrameOutput per frame
          → Deliver OperatorInferenceOutput to OperatorResultCallback
```

### Sealed Delivery Flow

The batching operator callback delivers results in two phases:

1. **Callback phase**: `OperatorResultCallback` receives `OperatorInferenceOutput`
   containing `frames()` (tensor pointers alive, no buffer access).
2. **Delivery phase**: `take_deliveries()` → `SealedDeliveries` → `unseal()`
   (blocks until output dropped) → `Vec<(VideoFrameProxy, SharedBuffer)>`.

```
callback receives OperatorInferenceOutput
   ├── frames() — read tensors, modify VideoFrameProxy
   ├── take_deliveries() → SealedDeliveries (one Arc<Seal> per batch)
   ├── send(sealed) to downstream channel
   └── output drops → clear_all_frame_objects → drop output_buffer → seal.release()

downstream receives SealedDeliveries
   ├── unseal() — blocks on Condvar until seal released
   └── for (frame, buffer) in pairs { buffer.into_buffer() → sole owner }
```

### PendingBatch

When a batch is submitted, `PendingBatch` is stored in the `PendingMap`:
- `frames: Vec<(VideoFrameProxy, SharedBuffer)>` — original per-frame pairs
- `rois: Vec<RoiKind>` — per-frame ROI specification (parallel to frames)
- `model_width`, `model_height`, `scaling` — from `NvInferConfig`

When results arrive, the callback retrieves the `PendingBatch` by batch ID,
zips frames with ROIs, and builds `OperatorElement` wrappers with the ROI
geometry needed for coordinate scaling.

### Coordinate Scaler

`CoordinateScaler` maps model-space coordinates back to absolute frame
coordinates via a 2D affine transform: `frame_xy = offset + model_xy * scale`.

The four coefficients (`scale_x`, `scale_y`, `offset_x`, `offset_y`) depend
on the ROI crop rectangle, model input dimensions, and `ModelInputScaling` mode:

| Mode | scale_x | scale_y | offset_x | offset_y |
|---|---|---|---|---|
| Fill | RW/MW | RH/MH | roi_left | roi_top |
| KeepAspectRatio | 1/s | 1/s | roi_left | roi_top |
| KeepAspectRatioSymmetric | 1/s | 1/s | roi_left - pad_x/s | roi_top - pad_y/s |

Where `s = min(MW/RW, MH/RH)`, `pad_x = (MW - RW*s)/2`, `pad_y = (MH - RH*s)/2`.

For rotated bounding boxes (`RBBox`), the center is point-transformed, and
dimensions are scaled using the trigonometric formulas from `RBBox::scale`
(inlined for efficiency — single `RBBox::new()` call, no intermediate atomics).

### OperatorElement Lifecycle

`OperatorElement` wraps `ElementOutput` and stores the ROI geometry + model
config. It implements `Deref<Target = ElementOutput>` so `roi_id`, `tensors`,
etc. are accessible directly. The `CoordinateScaler` is lazily initialized
via `OnceCell` on first call to any `scale_*` method.

### Timer Thread

A dedicated timer thread (named `nvinfer-{name}-timer` or
`nvinfer-batching-operator-timer`) uses `Condvar` to sleep until the batch
deadline. The `Condvar` is notified when:
- The first frame arrives (sets the deadline)
- `shutdown()` is called (wakes to exit)

### Batch Submission Trigger

Batches are submitted when either:
1. `frames.len() >= max_batch_size` (immediate submit from `add_frame`)
2. Timer deadline expires (submit from timer thread)
3. `flush()` is called explicitly
