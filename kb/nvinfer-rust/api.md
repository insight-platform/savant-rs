# NvInfer Public API

Crate: `nvinfer`  
Prelude: `use deepstream_nvinfer::prelude::*;`

## Core exports

- Runtime: `NvInfer`, `NvInferOutput`
- Config: `NvInferConfig`, `ModelInputScaling`, `ModelColorFormat`, `MetaClearPolicy`
- Data model: `Roi`, `RoiKind`, `BatchInferenceOutput`, `ElementOutput`, `TensorView`, `DataType`
- Batching layer: `NvInferBatchingOperator` and related config/output types

## NvInfer runtime

Constructor:

- `NvInfer::new(config: NvInferConfig) -> Result<NvInfer>`

I/O methods (channel-based):

- `submit(&self, batch: SharedBuffer, rois: Option<&HashMap<u32, Vec<Roi>>>) -> Result<()>`
- `send_eos(&self, source_id: &str) -> Result<()>`
- `send_event(&self, event: gst::Event) -> Result<()>`
- `recv(&self) -> Result<NvInferOutput>`
- `recv_timeout(&self, timeout: Duration) -> Result<Option<NvInferOutput>>`
- `try_recv(&self) -> Result<Option<NvInferOutput>>`
- `is_failed(&self) -> bool`
- `graceful_shutdown(&self, timeout: Duration) -> Result<Vec<NvInferOutput>>`
- `shutdown(&self) -> Result<()>`

`NvInferOutput`:

```rust
pub enum NvInferOutput {
    Inference(BatchInferenceOutput),
    Event(gst::Event),
    Eos { source_id: String },
    Error(NvInferError),
}
```

## NvInferConfig

`NvInferConfig::new(nvinfer_properties, input_format, model_width, model_height, model_color_format) -> Self`

Important builder methods:

- `with_element_properties`
- `gpu_id`
- `name`
- `meta_clear_policy`
- `disable_output_host_copy`
- `scaling`
- `operation_timeout`
- `input_channel_capacity`
- `output_channel_capacity`
- `drain_poll_interval`
- `model_input_dimensions`
- `validate_and_materialize`

## Inference output types

`BatchInferenceOutput`:

- `elements(&self) -> &[ElementOutput]`
- `num_elements(&self) -> usize`
- `host_copy_enabled(&self) -> bool`
- `buffer(&self) -> SharedBuffer`

`ElementOutput` fields:

- `roi_id: Option<i64>`
- `slot_number: u32`
- `tensors: Vec<TensorView>`

`TensorView` typed readers:

- `as_f32s`, `as_i32s`, `as_i8s`

## ROI model

- `Roi { id: i64, bbox: RBBox }`
- `Roi::new(id, bbox)`
- `RoiKind::{FullFrame, Rois(Vec<Roi>)}`

Helper:

- `attach_batch_meta_with_rois(...) -> Result<usize>`

## Batching operator

Config:

- `NvInferBatchingOperatorConfig { max_batch_size, max_batch_wait, pending_batch_timeout, nvinfer }`
- `NvInferBatchingOperatorConfig::builder(...)`

Callbacks:

- `BatchFormationCallback = Arc<dyn Fn(&[VideoFrameProxy]) -> BatchFormationResult + Send + Sync>`
- `OperatorResultCallback = Box<dyn FnMut(OperatorOutput) + Send>`

Runtime:

- `NvInferBatchingOperator::new(config, batch_formation, result_callback) -> Result<Self>`
- `add_frame`, `flush`, `send_eos`, `graceful_shutdown`, `shutdown`

Result model:

- `OperatorOutput::{Inference(OperatorInferenceOutput), Eos { source_id }, Error(NvInferError)}`
- `OperatorInferenceOutput::{frames, host_copy_enabled, take_deliveries}`
- `SealedDeliveries::{len, is_empty, is_released, unseal, unseal_timeout, try_unseal}`
- `OperatorElement` coordinate helpers: `coordinate_scaler`, `scale_points`, `scale_ltwh`, `scale_ltrb`, `scale_rbboxes`

## Errors

`NvInferError` includes:

- runtime/framework failures (`PipelineError`, `FrameworkError`, `ChannelDisconnected`, `ShuttingDown`)
- terminal states (`PipelineFailed`, `OperatorFailed`)
- config/build failures (`InvalidConfig`, `InvalidProperty`, `ElementCreationFailed`, `LinkFailed`, `BatchMetaFailed`)
- tensor access failures (`TensorTypeMismatch`, `HostDataUnavailable`)
- buffer-level failures via `Buffer(#[from] NvBufSurfaceError)`
