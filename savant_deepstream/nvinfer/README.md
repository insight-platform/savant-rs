# savant-deepstream-nvinfer

`savant-deepstream-nvinfer` wraps DeepStream TensorRT inference in a Rust API for ROI-driven `nvinfer` pipelines, tensor output extraction, and batching-operator integration. Import it as `deepstream_nvinfer` (the Rust library module name is `deepstream_nvinfer`; the underlying GStreamer element string stays `"nvinfer"`) when you need `sgie`-style secondary classifier flows, pgie-like full-frame inference, inference meta handling, tensor output access, or precise ROI crop fidelity without switching to `nvinferserver`.

## What's inside

- `NvInfer` and `NvInferOutput` are the low-level runtime. `submit` consumes a `SharedBuffer`, attaches ROI metadata, and returns `Inference`, `Event`, `Eos`, or `Error` outputs through the pull API.
- `NvInferConfig` is the central config type. It stores the DeepStream `nvinfer` property map, element properties, input format, model tensor size, TensorRT-related color/scaling options, watchdog timeouts, and channel capacities.
- `Roi`, `RoiKind`, and `attach_batch_meta_with_rois` are the ROI layer. They map Savant `RBBox` values to DeepStream object metadata so the same runtime can run whole-frame or per-object inference.
- `BatchInferenceOutput`, `ElementOutput`, `TensorView`, and `DataType` are the output model. They expose raw tensor views, slot numbers, ROI ids, and host/device-copy behavior for downstream post-processing.
- `ModelColorFormat`, `ModelInputScaling`, and `MetaClearPolicy` capture common model-preparation concerns such as RGB/BGR interpretation, aspect-ratio policy, and object-meta clearing.
- `NvInferBatchingOperator`, `NvInferBatchingOperatorConfig`, `NvInferBatchingOperatorConfigBuilder`, `OperatorOutput`, `OperatorInferenceOutput`, `OperatorElement`, and `SealedDeliveries` provide a higher-level batching operator for sources that need frame grouping before inference.
- `InferDims` and `InferTensorMeta` are re-exported from the DeepStream crates for callers that need direct access to native inference metadata.

## Usage

```rust
use deepstream_nvinfer::{
    ModelColorFormat, NvInfer, NvInferConfig, NvInferOutput, Roi, VideoFormat,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;

let props = HashMap::from([
    ("onnx-file".to_string(), "/models/detector.onnx".to_string()),
    ("model-engine-file".to_string(), "/models/detector.engine".to_string()),
]);

let infer = NvInfer::new(NvInferConfig::new(
    props,
    VideoFormat::RGBA,
    640,
    640,
    ModelColorFormat::RGB,
))?;

let batch = /* SharedBuffer produced by deepstream_buffers */;
let rois = HashMap::from([(
    0u32,
    vec![Roi::new(7, RBBox::ltwh(32.0, 48.0, 256.0, 256.0).unwrap())],
)]);

infer.submit(batch, Some(&rois))?;
if let NvInferOutput::Inference(output) = infer.recv()? {
    for element in output.elements() {
        println!("slot={} tensors={}", element.slot_number, element.tensors.len());
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Install

```toml
[dependencies]
savant-deepstream-nvinfer = "2"
```

Cargo features: this crate does not expose crate-specific features.

## System requirements

- NVIDIA DeepStream SDK 7.x.
- GStreamer 1.24+ with the DeepStream `nvinfer` plugin.
- CUDA, TensorRT, and model artifacts compatible with your target platform; `NvInferConfig` expects model paths and generates the final DeepStream config file at runtime.
- Jetson and dGPU are both supported, but TensorRT engines, CUDA versions, and throughput characteristics are platform-specific.
- If your deployment is based on Triton / `nvinferserver`, use the corresponding DeepStream element instead; `savant-deepstream-nvinfer` is specifically for `nvinfer`.

## Documentation

- [docs.rs](https://docs.rs/savant-deepstream-nvinfer)
- [Savant project site](https://insight-platform.github.io/savant-rs/)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
