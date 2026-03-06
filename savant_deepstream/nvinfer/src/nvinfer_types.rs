//! Inference-related types used by the NvInfer pipeline.

use deepstream_sys::{
    NvDsInferDataType, NvDsInferDataType_FLOAT, NvDsInferDataType_HALF, NvDsInferDataType_INT32,
    NvDsInferDataType_INT8,
};

/// Data type of a tensor element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit floating point.
    Float,
    /// 16-bit floating point (half precision).
    Half,
    /// 8-bit signed integer.
    Int8,
    /// 32-bit signed integer.
    Int32,
}

impl DataType {
    /// Size in bytes of a single element.
    pub fn element_size(self) -> usize {
        match self {
            DataType::Float | DataType::Int32 => 4,
            DataType::Half => 2,
            DataType::Int8 => 1,
        }
    }
}

impl From<NvDsInferDataType> for DataType {
    fn from(value: NvDsInferDataType) -> Self {
        match value {
            x if x == NvDsInferDataType_FLOAT => DataType::Float,
            x if x == NvDsInferDataType_HALF => DataType::Half,
            x if x == NvDsInferDataType_INT8 => DataType::Int8,
            x if x == NvDsInferDataType_INT32 => DataType::Int32,
            _ => DataType::Float,
        }
    }
}
