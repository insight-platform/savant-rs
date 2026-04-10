//! GStreamer pipeline wrapper for DeepStream nvinfer secondary inference.
//!
//! Built on `savant_gstreamer::pipeline` framework. Accepts batched NvBufSurface
//! buffers with ROI lists via [`NvInfer::submit`], delivers inference results
//! and EOS via [`NvInfer::recv`].

pub mod batch_meta_builder;
pub mod batching_operator;
pub mod config;
pub mod error;
pub mod meta_clear_policy;
pub mod model_color_format;
pub mod model_input_scaling;
pub mod nvinfer_types;
pub mod output;
pub mod pipeline;
pub mod prelude;
pub mod roi;

pub use batch_meta_builder::attach_batch_meta_with_rois;
pub use batching_operator::{
    BatchFormationCallback, BatchFormationResult, CoordinateScaler, NvInferBatchingOperator,
    NvInferBatchingOperatorConfig, NvInferBatchingOperatorConfigBuilder, OperatorElement,
    OperatorFrameOutput, OperatorInferenceOutput, OperatorOutput, OperatorResultCallback,
    SealedDeliveries,
};
pub use config::NvInferConfig;
pub use deepstream::{InferDims, InferTensorMeta};
pub use deepstream_buffers::VideoFormat;
pub use deepstream_buffers::{DstPadding, Rect, SharedBuffer, SurfaceView, TransformConfig};
pub use error::{NvInferError, Result};
pub use meta_clear_policy::MetaClearPolicy;
pub use model_color_format::ModelColorFormat;
pub use model_input_scaling::ModelInputScaling;
pub use nvinfer_types::DataType;
pub use output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use pipeline::{NvInfer, NvInferOutput};
pub use roi::{Roi, RoiKind};
