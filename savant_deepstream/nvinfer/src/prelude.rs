//! Convenience re-exports for all commonly used types.
//!
//! ```ignore
//! use nvinfer::prelude::*;
//! ```

pub use crate::batching_operator::{
    BatchFormationCallback, BatchFormationResult, CoordinateScaler, NvInferBatchingOperator,
    NvInferBatchingOperatorConfig, NvInferBatchingOperatorConfigBuilder, OperatorElement,
    OperatorFrameOutput, OperatorInferenceOutput, OperatorResultCallback, SealedDeliveries,
};
pub use crate::config::NvInferConfig;
pub use crate::error::{NvInferError, Result};
pub use crate::meta_clear_policy::MetaClearPolicy;
pub use crate::model_color_format::ModelColorFormat;
pub use crate::model_input_scaling::ModelInputScaling;
pub use crate::nvinfer_types::DataType;
pub use crate::output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use crate::pipeline::NvInfer;
pub use crate::roi::{Roi, RoiKind};
pub use deepstream_buffers::{
    DstPadding, Rect, SharedBuffer, SurfaceView, TransformConfig, VideoFormat,
};
