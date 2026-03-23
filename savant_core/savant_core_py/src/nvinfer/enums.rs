//! PyO3 enum wrappers for nvinfer types.

use nvinfer::DataType;
use nvinfer::MetaClearPolicy;
use nvinfer::ModelInputScaling;
use pyo3::prelude::*;

/// Controls when object metadata is erased from the batch buffer.
///
/// - ``NONE`` -- never clear automatically.
/// - ``BEFORE`` -- clear stale objects before attaching ROI objects (default).
/// - ``AFTER`` -- clear all objects when the output is dropped.
/// - ``BOTH`` -- clear before submission **and** after the output is dropped.
#[pyclass(
    from_py_object,
    name = "MetaClearPolicy",
    module = "savant_rs.nvinfer",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMetaClearPolicy {
    #[pyo3(name = "NONE")]
    None = 0,
    #[pyo3(name = "BEFORE")]
    Before = 1,
    #[pyo3(name = "AFTER")]
    After = 2,
    #[pyo3(name = "BOTH")]
    Both = 3,
}

impl From<PyMetaClearPolicy> for MetaClearPolicy {
    fn from(p: PyMetaClearPolicy) -> Self {
        match p {
            PyMetaClearPolicy::None => MetaClearPolicy::None,
            PyMetaClearPolicy::Before => MetaClearPolicy::Before,
            PyMetaClearPolicy::After => MetaClearPolicy::After,
            PyMetaClearPolicy::Both => MetaClearPolicy::Both,
        }
    }
}

impl From<MetaClearPolicy> for PyMetaClearPolicy {
    fn from(p: MetaClearPolicy) -> Self {
        match p {
            MetaClearPolicy::None => PyMetaClearPolicy::None,
            MetaClearPolicy::Before => PyMetaClearPolicy::Before,
            MetaClearPolicy::After => PyMetaClearPolicy::After,
            MetaClearPolicy::Both => PyMetaClearPolicy::Both,
        }
    }
}

/// How input frames are scaled to the model's fixed input dimensions.
///
/// - ``FILL`` -- stretch to model input (default).
/// - ``KEEP_ASPECT_RATIO`` -- preserve aspect ratio, padding on the right/bottom.
/// - ``KEEP_ASPECT_RATIO_SYMMETRIC`` -- preserve aspect ratio, symmetric padding.
#[pyclass(
    from_py_object,
    name = "ModelInputScaling",
    module = "savant_rs.nvinfer",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyModelInputScaling {
    #[pyo3(name = "FILL")]
    Fill = 0,
    #[pyo3(name = "KEEP_ASPECT_RATIO")]
    KeepAspectRatio = 1,
    #[pyo3(name = "KEEP_ASPECT_RATIO_SYMMETRIC")]
    KeepAspectRatioSymmetric = 2,
}

impl PyModelInputScaling {
    pub(crate) fn repr_str(&self) -> &'static str {
        match self {
            PyModelInputScaling::Fill => "ModelInputScaling.FILL",
            PyModelInputScaling::KeepAspectRatio => "ModelInputScaling.KEEP_ASPECT_RATIO",
            PyModelInputScaling::KeepAspectRatioSymmetric => {
                "ModelInputScaling.KEEP_ASPECT_RATIO_SYMMETRIC"
            }
        }
    }
}

#[pymethods]
impl PyModelInputScaling {
    fn __repr__(&self) -> &'static str {
        self.repr_str()
    }
}

impl From<PyModelInputScaling> for ModelInputScaling {
    fn from(p: PyModelInputScaling) -> Self {
        match p {
            PyModelInputScaling::Fill => ModelInputScaling::Fill,
            PyModelInputScaling::KeepAspectRatio => ModelInputScaling::KeepAspectRatio,
            PyModelInputScaling::KeepAspectRatioSymmetric => {
                ModelInputScaling::KeepAspectRatioSymmetric
            }
        }
    }
}

impl From<ModelInputScaling> for PyModelInputScaling {
    fn from(p: ModelInputScaling) -> Self {
        match p {
            ModelInputScaling::Fill => PyModelInputScaling::Fill,
            ModelInputScaling::KeepAspectRatio => PyModelInputScaling::KeepAspectRatio,
            ModelInputScaling::KeepAspectRatioSymmetric => {
                PyModelInputScaling::KeepAspectRatioSymmetric
            }
        }
    }
}

/// Data type of a tensor element.
///
/// - ``FLOAT`` -- 32-bit floating point (4 bytes).
/// - ``HALF`` -- 16-bit floating point (2 bytes).
/// - ``INT8`` -- 8-bit signed integer (1 byte).
/// - ``INT32`` -- 32-bit signed integer (4 bytes).
#[pyclass(
    from_py_object,
    name = "DataType",
    module = "savant_rs.nvinfer",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDataType {
    #[pyo3(name = "FLOAT")]
    Float = 0,
    #[pyo3(name = "HALF")]
    Half = 1,
    #[pyo3(name = "INT8")]
    Int8 = 2,
    #[pyo3(name = "INT32")]
    Int32 = 3,
}

#[pymethods]
impl PyDataType {
    /// Size in bytes of a single element of this type.
    fn element_size(&self) -> usize {
        DataType::from(*self).element_size()
    }

    fn __repr__(&self) -> &'static str {
        self.repr_str()
    }
}

impl PyDataType {
    pub(crate) fn repr_str(&self) -> &'static str {
        match self {
            PyDataType::Float => "DataType.FLOAT",
            PyDataType::Half => "DataType.HALF",
            PyDataType::Int8 => "DataType.INT8",
            PyDataType::Int32 => "DataType.INT32",
        }
    }
}

impl From<PyDataType> for DataType {
    fn from(d: PyDataType) -> Self {
        match d {
            PyDataType::Float => DataType::Float,
            PyDataType::Half => DataType::Half,
            PyDataType::Int8 => DataType::Int8,
            PyDataType::Int32 => DataType::Int32,
        }
    }
}

impl From<DataType> for PyDataType {
    fn from(d: DataType) -> Self {
        match d {
            DataType::Float => PyDataType::Float,
            DataType::Half => PyDataType::Half,
            DataType::Int8 => PyDataType::Int8,
            DataType::Int32 => PyDataType::Int32,
        }
    }
}
