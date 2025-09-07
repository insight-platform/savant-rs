use crate::{BatchMeta, DeepStreamError, Result};
use deepstream_sys::{NvDsInferDims, NvDsInferTensorMeta};

const SUPPORTED_DATA_TYPES: [deepstream_sys::NvDsInferDataType; 4] = [
    deepstream_sys::NvDsInferDataType_FLOAT,
    deepstream_sys::NvDsInferDataType_INT8,
    deepstream_sys::NvDsInferDataType_HALF,
    deepstream_sys::NvDsInferDataType_INT32,
];

/// Safe wrapper for DeepStream inference tensor metadata
///
/// This struct provides safe access to inference tensor metadata while managing
/// the underlying C memory properly.
pub struct InferTensorMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsInferTensorMeta,
    _batch_meta: BatchMeta,
}

#[derive(Debug)]
pub struct InferDims {
    pub dimensions: Vec<u32>,
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

impl InferTensorMeta {
    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    /// This is typically used internally or when working with existing
    /// inference tensor metadata.
    pub unsafe fn from_raw(raw: *mut NvDsInferTensorMeta, batch_meta: &BatchMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("InferTensorMeta::from_raw"));
        }

        Ok(Self {
            raw,
            _batch_meta: batch_meta.clone(),
        })
    }

    /// Get the raw pointer
    ///
    /// # Safety
    /// This returns the raw C pointer. Use with caution.
    pub fn as_raw(&self) -> *mut NvDsInferTensorMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsInferTensorMeta {
        &*self.raw
    }

    /// Get the unique identifier for this inference tensor metadata
    pub fn unique_id(&self) -> u32 {
        unsafe { (*self.raw).unique_id }
    }

    /// Get the number of output layers
    pub fn num_output_layers(&self) -> u32 {
        unsafe { (*self.raw).num_output_layers }
    }

    /// Get the GPU ID
    pub fn gpu_id(&self) -> i32 {
        unsafe { (*self.raw).gpu_id }
    }

    /// Check if aspect ratio is maintained
    pub fn maintain_aspect_ratio(&self) -> bool {
        unsafe { (*self.raw).maintain_aspect_ratio != 0 }
    }

    /// Get the output layers info pointer
    ///
    /// # Returns
    /// * `*mut NvDsInferLayerInfo` - Raw pointer to output layers info
    pub fn output_layers_info(&self) -> *mut deepstream_sys::NvDsInferLayerInfo {
        unsafe { (*self.raw).output_layers_info }
    }

    pub fn layer_dimensions(&self) -> Vec<InferDims> {
        let mut output_layers_dims = Vec::new();
        let mut current = unsafe { (*self.raw).output_layers_info };
        let mut num_layers = self.num_output_layers();
        while num_layers > 0 && !current.is_null() {
            output_layers_dims.push(unsafe { InferDims::from(&(*current).__bindgen_anon_1.dims) });
            current = unsafe { current.offset(1) };
            num_layers -= 1;
        }
        output_layers_dims
    }

    pub fn layer_names(&self) -> Vec<String> {
        let mut layer_names = Vec::new();
        let mut current = unsafe { (*self.raw).output_layers_info };
        let mut num_layers = self.num_output_layers();
        while num_layers > 0 {
            let data_type = unsafe { (*current).dataType };
            // must be among supported data types
            let types = SUPPORTED_DATA_TYPES;
            let layer_name = unsafe {
                std::ffi::CStr::from_ptr((*current).layerName)
                    .to_string_lossy()
                    .to_string()
            };
            if !types.contains(&data_type) {
                log::error!(
                    "Unsupported data type: {} for layer {}",
                    data_type,
                    layer_name
                );
            }
            layer_names.push(unsafe {
                std::ffi::CStr::from_ptr((*current).layerName)
                    .to_string_lossy()
                    .to_string()
            });
            current = unsafe { current.offset(1) };
            num_layers -= 1;
        }
        layer_names
    }

    /// Get the host buffer pointers
    ///
    /// # Returns
    /// * `*mut *mut c_void` - Raw pointer to host buffer pointers
    pub fn out_buf_ptrs_host(&self) -> Vec<*mut std::ffi::c_void> {
        let mut out_buf_ptrs_host = Vec::new();
        let mut current = unsafe { (*self.raw).out_buf_ptrs_host };
        let mut num_layers = self.num_output_layers();
        while num_layers > 0 && !current.is_null() {
            out_buf_ptrs_host.push(unsafe { *current });
            current = unsafe { current.offset(1) };
            num_layers -= 1;
        }
        out_buf_ptrs_host
    }

    /// Get the device buffer pointers
    ///
    /// # Returns
    /// * `Vec<*mut c_void>` - Vector of device buffer pointers
    pub fn out_buf_ptrs_dev(&self) -> Vec<*mut std::ffi::c_void> {
        let mut out_buf_ptrs_dev = Vec::new();
        let mut current = unsafe { (*self.raw).out_buf_ptrs_dev };
        let mut num_layers = self.num_output_layers();
        while num_layers > 0 && !current.is_null() {
            out_buf_ptrs_dev.push(unsafe { *current });
            current = unsafe { current.offset(1) };
            num_layers -= 1;
        }
        out_buf_ptrs_dev
    }

    /// Get the network info pointer
    ///
    /// # Returns
    /// * `NvDsInferNetworkInfo` - Network info structure
    pub fn network_info(&self) -> deepstream_sys::NvDsInferNetworkInfo {
        unsafe { (*self.raw).network_info }
    }
}

impl Clone for InferTensorMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            _batch_meta: self._batch_meta.clone(),
        }
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
            .field("has_network_info", &true)
            .finish()
    }
}
