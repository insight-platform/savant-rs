# deepstream_nvinfer

Safe Rust API for NVIDIA DeepStream TensorRT inference.

This crate provides safe, idiomatic Rust wrappers around the DeepStream inference API, particularly focusing on TensorRT inference context and related functionality.

## Features

- Safe wrapper for `NvDsInferContext`
- Inference context initialization parameters
- Batch input/output handling
- Layer information management
- Tensor metadata support
- Comprehensive error handling

## Usage

```rust
use deepstream_nvinfer::{Context, InferContextInitParams, InferNetworkMode, InferTensorOrder, InferFormat};

// Create initialization parameters
let mut init_params = InferContextInitParams::new();
init_params
    .set_gpu_id(0)
    .set_max_batch_size(4)
    .set_unique_id(1)
    .set_network_mode(InferNetworkMode::FP16)
    .set_onnx_file_path("model.onnx")?
    .set_engine_file_path("model.engine")?
    .set_net_input_order(InferTensorOrder::NCHW)
    .set_net_input_format(InferFormat::RGB)
    .set_infer_input_dims(3, 224, 224);

// Create inference context
let mut infer_context = Context::new_with_default_logging(init_params)?;

// Use the context for inference...
```

## Migration from deepstream crate

This crate was extracted from the main `deepstream` crate to provide focused TensorRT inference functionality. If you were previously using inference features from the `deepstream` crate, you can now use this dedicated crate instead.

### Changes from the original implementation:

- All inference-related types are now in the `deepstream_nvinfer` crate
- Error handling uses `NvInferError` instead of `DeepStreamError`
- `InferTensorMeta::from_raw()` no longer requires a `BatchMeta` parameter
- The crate has its own `Result<T>` type alias

## Dependencies

- `deepstream-sys` - Raw DeepStream C API bindings
- `thiserror` - Error handling
- `log` - Logging support
