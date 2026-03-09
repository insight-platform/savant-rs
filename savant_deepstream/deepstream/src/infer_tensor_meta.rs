//! Safe wrappers for DeepStream inference tensor metadata.
//!
//! Provides [`InferTensorMeta`] and [`InferDims`] for reading output tensor
//! metadata attached by the nvinfer element.

use crate::{DeepStreamError, Result};
use deepstream_sys::{
    NvDsInferDataType, NvDsInferDataType_FLOAT, NvDsInferDataType_HALF, NvDsInferDataType_INT32,
    NvDsInferDataType_INT8, NvDsInferDims, NvDsInferTensorMeta,
};
use std::ffi::{c_void, CStr};

const SUPPORTED_DATA_TYPES: [NvDsInferDataType; 4] = [
    NvDsInferDataType_FLOAT,
    NvDsInferDataType_INT8,
    NvDsInferDataType_HALF,
    NvDsInferDataType_INT32,
];

/// Tensor dimensions and total element count.
#[derive(Debug, Clone, PartialEq)]
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

/// Safe wrapper for `NvDsInferTensorMeta` attached to a frame by the nvinfer
/// element.
///
/// This is a non-owning wrapper: the underlying memory is managed by
/// DeepStream and remains valid as long as the parent `gst::Sample` (or
/// buffer) is alive.
pub struct InferTensorMeta {
    raw: *mut NvDsInferTensorMeta,
}

unsafe impl Send for InferTensorMeta {}

impl InferTensorMeta {
    /// Wrap a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the pointer is valid for the lifetime of this
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`DeepStreamError::NullPointer`] if `raw` is null.
    pub unsafe fn from_raw(raw: *mut NvDsInferTensorMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("InferTensorMeta::from_raw"));
        }
        Ok(Self { raw })
    }

    /// Get the raw pointer.
    pub fn as_raw(&self) -> *mut NvDsInferTensorMeta {
        self.raw
    }

    /// Get the unique identifier for this inference tensor metadata.
    pub fn unique_id(&self) -> u32 {
        unsafe { (*self.raw).unique_id }
    }

    /// Number of output layers.
    pub fn num_output_layers(&self) -> u32 {
        unsafe { (*self.raw).num_output_layers }
    }

    /// GPU ID.
    pub fn gpu_id(&self) -> i32 {
        unsafe { (*self.raw).gpu_id }
    }

    /// Whether aspect ratio is maintained.
    pub fn maintain_aspect_ratio(&self) -> bool {
        unsafe { (*self.raw).maintain_aspect_ratio != 0 }
    }

    /// Raw pointer to the output-layers info array.
    pub fn output_layers_info(&self) -> *mut deepstream_sys::NvDsInferLayerInfo {
        unsafe { (*self.raw).output_layers_info }
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
            let data_type = unsafe { (*cur).dataType };
            if !SUPPORTED_DATA_TYPES.contains(&data_type) {
                log::error!("Unsupported data type: {} for layer {}", data_type, name);
            }
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

    /// Data types per output layer (raw `NvDsInferDataType` values).
    pub fn layer_data_types(&self) -> Vec<NvDsInferDataType> {
        let n = self.num_output_layers() as usize;
        let mut types = Vec::with_capacity(n);
        let mut cur = unsafe { (*self.raw).output_layers_info };
        for _ in 0..n {
            types.push(unsafe { (*cur).dataType });
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

    /// Network info structure.
    pub fn network_info(&self) -> deepstream_sys::NvDsInferNetworkInfo {
        unsafe { (*self.raw).network_info }
    }
}

impl Clone for InferTensorMeta {
    fn clone(&self) -> Self {
        Self { raw: self.raw }
    }
}

impl std::fmt::Debug for InferTensorMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferTensorMeta")
            .field("unique_id", &self.unique_id())
            .field("num_output_layers", &self.num_output_layers())
            .field("gpu_id", &self.gpu_id())
            .field("layer_names", &self.layer_names())
            .field("layer_dimensions", &self.layer_dimensions())
            .field("has_host_buffers", &!self.out_buf_ptrs_host().is_empty())
            .field("has_device_buffers", &!self.out_buf_ptrs_dev().is_empty())
            .finish()
    }
}
