pub mod context;
pub mod init_params;
pub mod io;
pub mod layer_info;
pub mod output;
pub use context::Context;
use deepstream_sys::*;
pub use init_params::InferContextInitParams;
pub use io::{BatchInput, BatchOutput, NetworkType};
pub use layer_info::LayerInfo;

/// Data type enumeration for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit floating point
    Float,
    /// 16-bit floating point (half precision)
    Half,
    /// 8-bit signed integer
    Int8,
    /// 32-bit signed integer
    Int32,
}

impl From<DataType> for u32 {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Float => NvDsInferDataType_FLOAT,
            DataType::Half => NvDsInferDataType_HALF,
            DataType::Int8 => NvDsInferDataType_INT8,
            DataType::Int32 => NvDsInferDataType_INT32,
        }
    }
}

impl From<u32> for DataType {
    fn from(value: u32) -> Self {
        match value {
            x if x == NvDsInferDataType_FLOAT => DataType::Float,
            x if x == NvDsInferDataType_HALF => DataType::Half,
            x if x == NvDsInferDataType_INT8 => DataType::Int8,
            x if x == NvDsInferDataType_INT32 => DataType::Int32,
            _ => DataType::Float, // Default to Float for unknown values
        }
    }
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
}

impl From<u32> for LogLevel {
    fn from(level: u32) -> Self {
        match level {
            0 => LogLevel::Error,
            1 => LogLevel::Warning,
            2 => LogLevel::Info,
            3 => LogLevel::Debug,
            _ => LogLevel::Info,
        }
    }
}

impl From<LogLevel> for u32 {
    fn from(log_level: LogLevel) -> u32 {
        match log_level {
            LogLevel::Error => 0,
            LogLevel::Warning => 1,
            LogLevel::Info => 2,
            LogLevel::Debug => 3,
        }
    }
}

/// Safe wrapper for NvDsInferNetworkInfo
pub struct NetworkInfo {
    inner: NvDsInferNetworkInfo,
}

impl NetworkInfo {
    /// Create a new network info structure
    pub fn new() -> Self {
        Self {
            inner: unsafe { std::mem::zeroed() },
        }
    }

    /// Get the network width
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    /// Get the network height
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    /// Get the number of channels
    pub fn channels(&self) -> u32 {
        self.inner.channels
    }

    /// Get access to the raw structure
    ///
    /// # Safety
    /// This provides access to the raw C structure. Use with caution.
    pub fn as_raw(&self) -> &NvDsInferNetworkInfo {
        &self.inner
    }
}

impl Default for NetworkInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Input format enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferFormat {
    /// 24-bit interleaved R-G-B format
    RGB,
    /// 24-bit interleaved B-G-R format
    BGR,
    /// 8-bit Luma format
    Gray,
    /// NCHW planar tensor format
    Tensor,
}

impl From<InferFormat> for NvDsInferFormat {
    fn from(format: InferFormat) -> Self {
        match format {
            InferFormat::RGB => NvDsInferFormat_NvDsInferFormat_RGB,
            InferFormat::BGR => NvDsInferFormat_NvDsInferFormat_BGR,
            InferFormat::Gray => NvDsInferFormat_NvDsInferFormat_GRAY,
            InferFormat::Tensor => NvDsInferFormat_NvDsInferFormat_Tensor,
        }
    }
}

impl From<NvDsInferFormat> for InferFormat {
    fn from(format: NvDsInferFormat) -> Self {
        match format {
            x if x == NvDsInferFormat_NvDsInferFormat_RGB => InferFormat::RGB,
            x if x == NvDsInferFormat_NvDsInferFormat_BGR => InferFormat::BGR,
            x if x == NvDsInferFormat_NvDsInferFormat_GRAY => InferFormat::Gray,
            x if x == NvDsInferFormat_NvDsInferFormat_Tensor => InferFormat::Tensor,
            _ => unimplemented!("This format is not supported: {}", format),
        }
    }
}

/// Tensor order enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferTensorOrder {
    /// NCHW format: Number of batches, Channels, Height, Width
    NCHW,
    /// NHWC format: Number of batches, Height, Width, Channels
    NHWC,
    /// NC format: Number of batches, Channels (for 1D data)
    NC,
}

impl From<InferTensorOrder> for NvDsInferTensorOrder {
    fn from(order: InferTensorOrder) -> Self {
        match order {
            InferTensorOrder::NCHW => NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW,
            InferTensorOrder::NHWC => NvDsInferTensorOrder_NvDsInferTensorOrder_kNHWC,
            InferTensorOrder::NC => NvDsInferTensorOrder_NvDsInferTensorOrder_kNC,
        }
    }
}

impl From<NvDsInferTensorOrder> for InferTensorOrder {
    fn from(order: NvDsInferTensorOrder) -> Self {
        match order {
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW => InferTensorOrder::NCHW,
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNHWC => InferTensorOrder::NHWC,
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNC => InferTensorOrder::NC,
            _ => InferTensorOrder::NCHW, // Default to NCHW for unknown values
        }
    }
}

/// Network mode enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferNetworkMode {
    /// 32-bit floating point precision
    FP32,
    /// 8-bit integer precision
    INT8,
    /// 16-bit floating point precision
    FP16,
    /// Best precision available
    Best,
}

impl From<InferNetworkMode> for NvDsInferNetworkMode {
    fn from(mode: InferNetworkMode) -> Self {
        match mode {
            InferNetworkMode::FP32 => NvDsInferNetworkMode_NvDsInferNetworkMode_FP32,
            InferNetworkMode::INT8 => NvDsInferNetworkMode_NvDsInferNetworkMode_INT8,
            InferNetworkMode::FP16 => NvDsInferNetworkMode_NvDsInferNetworkMode_FP16,
            InferNetworkMode::Best => NvDsInferNetworkMode_NvDsInferNetworkMode_BEST,
        }
    }
}

impl From<NvDsInferNetworkMode> for InferNetworkMode {
    fn from(mode: NvDsInferNetworkMode) -> Self {
        match mode {
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_FP32 => InferNetworkMode::FP32,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_INT8 => InferNetworkMode::INT8,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_FP16 => InferNetworkMode::FP16,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_BEST => InferNetworkMode::Best,
            _ => InferNetworkMode::FP32, // Default to FP32 for unknown values
        }
    }
}
