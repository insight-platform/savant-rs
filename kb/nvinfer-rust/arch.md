# NvInfer Architecture

## Module Tree

```
savant_deepstream/nvinfer/src/
├── lib.rs
├── pipeline.rs
├── config.rs
├── error.rs
├── output.rs
├── roi.rs
├── model_input_scaling.rs
├── model_color_format.rs
├── meta_clear_policy.rs
├── nvinfer_types.rs
├── batch_meta_builder.rs
├── batching_operator.rs
└── batching_operator/
    ├── config.rs
    ├── operator.rs
    ├── output.rs
    ├── scaler.rs
    ├── submit.rs
    └── types.rs
```

## Runtime pipeline

`NvInfer` uses the shared `savant_gstreamer::pipeline` framework:

```
appsrc (NVMM caps) -> nvinfer -> appsink
```

Inputs are submitted via `submit(...)`, outputs are pulled via `recv` APIs (`recv`, `recv_timeout`, `try_recv`) as `NvInferOutput`.

## Submission flow

1. Take exclusive ownership of `SharedBuffer` (`into_buffer`).
2. Attach `NvDsBatchMeta` + ROI metadata (`attach_batch_meta_with_rois`).
3. Assign internal monotonic PTS for framework correlation.
4. Push through framework input channel to `appsrc`.
5. Pull `NvInferOutput::Inference(BatchInferenceOutput)` from output channel.

## ROI handling

- `rois = Some(map)`: per-slot `Roi` lists are encoded as `NvDsObjectMeta`.
- `rois = None`: full-frame sentinel ROI is synthesized for each filled slot.
- Slot numbering in results is `NvDsFrameMeta.batch_id` (surface slot index).

## Config materialization

`NvInferConfig::validate_and_materialize` writes a temporary INI file consumed by the `nvinfer` element. The temp file is held inside `NvInfer` (`_config_file`) for pipeline lifetime safety.

## Failure model

The framework enforces `operation_timeout`; timeout transitions the runtime into terminal failed state (`PipelineFailed`) and requires recreation of the `NvInfer` instance.

## Batching operator layer

`NvInferBatchingOperator` wraps per-frame inputs (`VideoFrameProxy`, `SharedBuffer`) and:

1. accumulates frames in `deepstream_buffers::BatchState`,
2. forms `NonUniformBatch`,
3. submits to inner `NvInfer`,
4. joins inference outputs with pending frame context,
5. delivers `OperatorOutput` to user callback.

Delivery ownership uses a seal protocol:

- callback gets `OperatorInferenceOutput` for tensor processing,
- `take_deliveries()` returns `SealedDeliveries`,
- `unseal()` blocks until output drop releases buffer ownership.

## Coordinate scaling

`CoordinateScaler` converts model-space coordinates back to frame-space, with mode-dependent behavior (`Fill`, `KeepAspectRatio`, `KeepAspectRatioSymmetric`) and dedicated helpers for points, axis-aligned boxes, and rotated `RBBox`.
