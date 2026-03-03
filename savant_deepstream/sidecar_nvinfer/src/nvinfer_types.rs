//! Inference types extracted from the nvinfer crate for self-contained usage.
//!
//! Provides [`DataType`], [`InferDims`], and [`InferTensorMeta`] — the minimal
//! subset required by the sidecar pipeline to read output tensor metadata from
//! the nvinfer element.

use deepstream_sys::{
    NvDsInferDataType, NvDsInferDataType_FLOAT, NvDsInferDataType_HALF, NvDsInferDataType_INT32,
    NvDsInferDataType_INT8, NvDsInferDims, NvDsInferTensorMeta,
};
use std::ffi::{c_void, CStr};

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

/// Tensor dimensions and total element count.
#[derive(Debug, Clone)]
pub struct InferDims {
    /// Shape along each axis.
    pub dimensions: Vec<u32>,
    /// Total number of elements (product of dimensions).
    pub num_elements: u32,
}

impl From<&NvDsInferDims> for InferDims {
    fn from(raw: &NvDsInferDims) -> Self {
        Self {
            dimensions: raw.d[..raw.numDims as usize].to_vec(),
            num_elements: raw.numElements,
        }
    }
}

/// Read-only accessor for `NvDsInferTensorMeta` attached to a frame by nvinfer.
///
/// This is a non-owning wrapper: the underlying memory is managed by DeepStream
/// and remains valid as long as the parent `gst::Sample` is alive.
pub struct InferTensorMeta {
    raw: *mut NvDsInferTensorMeta,
}

// The pointer is valid as long as the owning gst::Sample lives; ownership is
// always transferred together, so Send is safe.
unsafe impl Send for InferTensorMeta {}

impl InferTensorMeta {
    /// Wrap a raw pointer.  Returns `None` if null.
    ///
    /// # Safety
    /// The caller must guarantee the pointer is valid for the lifetime of this value.
    pub unsafe fn from_raw(raw: *mut NvDsInferTensorMeta) -> Option<Self> {
        if raw.is_null() {
            None
        } else {
            Some(Self { raw })
        }
    }

    /// Number of output layers.
    pub fn num_output_layers(&self) -> u32 {
        unsafe { (*self.raw).num_output_layers }
    }

    /// Human-readable names for each output layer.
    pub fn layer_names(&self) -> Vec<String> {
        let n = self.num_output_layers() as usize;
        let mut names = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).output_layers_info };
        for _ in 0..n {
            let name = unsafe {
                CStr::from_ptr((*cur).layerName)
                    .to_string_lossy()
                    .into_owned()
            };
            names.push(name);
            cur = unsafe { cur.add(1) };
        }
        names
    }

    /// Dimensions per output layer.
    pub fn layer_dimensions(&self) -> Vec<InferDims> {
        let n = self.num_output_layers() as usize;
        let mut dims = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).output_layers_info };
        for _ in 0..n {
            dims.push(unsafe { InferDims::from(&(*cur).__bindgen_anon_1.dims) });
            cur = unsafe { cur.add(1) };
        }
        dims
    }

    /// Data types per output layer.
    pub fn layer_data_types(&self) -> Vec<DataType> {
        let n = self.num_output_layers() as usize;
        let mut types = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).output_layers_info };
        for _ in 0..n {
            types.push(DataType::from(unsafe { (*cur).dataType }));
            cur = unsafe { cur.add(1) };
        }
        types
    }

    /// Host-side buffer pointers (one per output layer).
    pub fn out_buf_ptrs_host(&self) -> Vec<*mut c_void> {
        let n = self.num_output_layers() as usize;
        let mut ptrs = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).out_buf_ptrs_host };
        for _ in 0..n {
            if cur.is_null() {
                break;
            }
            ptrs.push(unsafe { *cur });
            cur = unsafe { cur.add(1) };
        }
        ptrs
    }

    /// Device-side buffer pointers (one per output layer).
    pub fn out_buf_ptrs_dev(&self) -> Vec<*mut c_void> {
        let n = self.num_output_layers() as usize;
        let mut ptrs = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).out_buf_ptrs_dev };
        for _ in 0..n {
            if cur.is_null() {
                break;
            }
            ptrs.push(unsafe { *cur });
            cur = unsafe { cur.add(1) };
        }
        ptrs
    }
}
