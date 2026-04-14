# Critical Caveats & Design Decisions

## 1) `submit` consumes buffers

`NvInfer::submit` takes ownership via `SharedBuffer::into_buffer()`.  
Any outstanding `SurfaceView` / cloned `SharedBuffer` references cause a submission error.

## 2) Tensor view lifetime

`TensorView` pointers are valid only while owning output objects are alive (`BatchInferenceOutput` / `OperatorInferenceOutput`).  
Consume tensor data before dropping those owners.

## 3) Full-frame fallback behavior

If `rois` is `None` (or slot missing in map), full-frame sentinel ROI metadata is auto-generated from slot dimensions.

## 4) Timeout is terminal

When `operation_timeout` is exceeded, pipeline enters terminal failed state (`PipelineFailed`) and must be recreated.  
For batching operator, `pending_batch_timeout` similarly triggers terminal `OperatorFailed`.

## 5) Config ownership

`NvInferConfig::validate_and_materialize` produces a temporary config file that must outlive the running pipeline; `NvInfer` stores it internally for this reason.

## 6) Batching delivery seal

For operator mode, buffer deliveries stay sealed until `OperatorInferenceOutput` is dropped:

1. process `frames()`,
2. call `take_deliveries()`,
3. drop output,
4. `unseal()` deliveries.
