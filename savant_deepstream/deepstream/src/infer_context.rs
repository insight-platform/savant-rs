use crate::{DeepStreamError, Result};

pub mod init_params;
pub mod io;
use deepstream_sys::{
    NvDsInferContextHandle,
    NvDsInferContextLoggingFunc,
    NvDsInferContext_Create,
    NvDsInferContext_DequeueOutputBatch,
    NvDsInferContext_Destroy,
    NvDsInferContext_FillLayersInfo,
    NvDsInferContext_GetLabel,
    NvDsInferContext_GetNetworkInfo,
    NvDsInferContext_GetNumLayersInfo,
    NvDsInferContext_QueueInputBatch,
    // Data type constants
    NvDsInferDataType_FLOAT,
    NvDsInferDataType_HALF,
    NvDsInferDataType_INT32,
    NvDsInferDataType_INT8,
    // Format constants
    NvDsInferLayerInfo,
    NvDsInferNetworkInfo,
    // Network type constants
    NvDsInferStatus,
    NvDsInferStatus_NVDSINFER_SUCCESS,
    // Tensor order constants
};
pub use init_params::{InferContextInitParams, InferFormat, InferTensorOrder};
pub use io::{BatchInput, BatchOutput};
use std::{ffi::CStr, marker::PhantomData, os::raw::c_char, ptr};

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

impl Into<u32> for LogLevel {
    fn into(self) -> u32 {
        match self {
            LogLevel::Error => 0,
            LogLevel::Warning => 1,
            LogLevel::Info => 2,
            LogLevel::Debug => 3,
        }
    }
}

/// Safe wrapper for NvDsInferContext
///
/// This struct provides safe access to the DeepStream inference context,
/// managing the underlying C handle and ensuring proper cleanup.
pub struct InferContext {
    handle: NvDsInferContextHandle,
    _phantom: PhantomData<*mut ()>, // Ensure !Send and !Sync
}

impl InferContext {
    /// Create a new inference context
    ///
    /// # Arguments
    /// * `init_params` - Initialization parameters for the context
    /// * `user_ctx` - Optional user context pointer
    /// * `log_func` - Optional logging callback function
    ///
    /// # Returns
    /// A new `InferContext` instance or an error if creation failed
    pub fn new(
        init_params: &InferContextInitParams,
        user_ctx: Option<*mut std::ffi::c_void>,
        log_func: NvDsInferContextLoggingFunc,
    ) -> Result<Self> {
        let mut handle: NvDsInferContextHandle = ptr::null_mut();
        let user_ctx_ptr = user_ctx.unwrap_or(ptr::null_mut());

        let status = unsafe {
            NvDsInferContext_Create(
                &mut handle,
                &init_params.as_raw() as *const _ as *mut _,
                user_ctx_ptr,
                log_func,
            )
        };

        if status != NvDsInferStatus_NVDSINFER_SUCCESS {
            return Err(DeepStreamError::invalid_operation(&format!(
                "Failed to create inference context: status {}",
                status
            )));
        }

        if handle.is_null() {
            return Err(DeepStreamError::null_pointer("InferContext::new"));
        }

        Ok(Self {
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a new inference context with default logging
    ///
    /// # Arguments
    /// * `init_params` - Initialization parameters for the context
    ///
    /// # Returns
    /// A new `InferContext` instance or an error if creation failed
    pub fn new_with_default_logging(init_params: &InferContextInitParams) -> Result<Self> {
        Self::new(init_params, None, Some(logging_callback_wrapper))
    }

    /// Queue a batch of input frames for inference
    ///
    /// # Arguments
    /// * `batch_input` - The batch input containing frames to process
    ///
    /// # Returns
    /// `Ok(())` if successful, or an error if queueing failed
    pub fn queue_input_batch(&mut self, batch_input: &BatchInput) -> Result<()> {
        let status = unsafe {
            NvDsInferContext_QueueInputBatch(self.handle, &batch_input.inner as *const _ as *mut _)
        };

        if status != NvDsInferStatus_NVDSINFER_SUCCESS {
            return Err(DeepStreamError::invalid_operation(&format!(
                "Failed to queue input batch: status {}",
                status
            )));
        }

        Ok(())
    }

    /// Dequeue output for a batch of frames
    ///
    /// # Returns
    /// A `BatchOutput` containing the inference results, or an error if dequeueing failed
    pub fn dequeue_output_batch(&mut self) -> Result<BatchOutput> {
        let mut batch_output = BatchOutput::new();

        let status =
            unsafe { NvDsInferContext_DequeueOutputBatch(self.handle, batch_output.as_raw_mut()) };

        if status != NvDsInferStatus_NVDSINFER_SUCCESS {
            return Err(DeepStreamError::invalid_operation(&format!(
                "Failed to dequeue output batch: status {}",
                status
            )));
        }

        // Set the context handle for proper cleanup
        batch_output.context_handle = Some(self.handle);
        Ok(batch_output)
    }

    /// Get network information
    ///
    /// # Returns
    /// Network information structure
    pub fn get_network_info(&self) -> NetworkInfo {
        let mut network_info = NetworkInfo::new();

        unsafe {
            NvDsInferContext_GetNetworkInfo(self.handle, &mut network_info.inner);
        }

        network_info
    }

    /// Get the number of bound layers
    ///
    /// # Returns
    /// The number of bound layers in the inference engine
    pub fn get_num_layers_info(&self) -> u32 {
        unsafe { NvDsInferContext_GetNumLayersInfo(self.handle) }
    }

    /// Get information about all bound layers
    ///
    /// # Returns
    /// A vector of layer information structures
    pub fn get_layers_info(&self) -> Vec<LayerInfo> {
        let num_layers = self.get_num_layers_info();
        if num_layers == 0 {
            return Vec::new();
        }

        let mut layers_info = vec![LayerInfo::new(); num_layers as usize];

        unsafe {
            NvDsInferContext_FillLayersInfo(
                self.handle,
                layers_info.as_mut_ptr() as *mut NvDsInferLayerInfo,
            );
        }

        layers_info
    }

    /// Get the label for a class ID or attribute
    ///
    /// # Arguments
    /// * `id` - Class ID for detectors, or attribute ID for classifiers
    /// * `value` - Attribute value for classifiers; set to 0 for detectors
    ///
    /// # Returns
    /// The label string if available, or None if not found
    pub fn get_label(&self, id: u32, value: u32) -> Option<String> {
        unsafe {
            let label_ptr = NvDsInferContext_GetLabel(self.handle, id, value);
            if label_ptr.is_null() {
                None
            } else {
                CStr::from_ptr(label_ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }

    /// Get the raw handle
    ///
    /// # Safety
    /// This returns the raw C handle. Use with caution.
    pub fn as_raw(&self) -> NvDsInferContextHandle {
        self.handle
    }
}

impl Drop for InferContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                NvDsInferContext_Destroy(self.handle);
            }
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

/// Safe wrapper for NvDsInferLayerInfo
#[derive(Clone)]
pub struct LayerInfo {
    inner: NvDsInferLayerInfo,
}

impl LayerInfo {
    /// Create a new layer info structure
    pub fn new() -> Self {
        Self {
            inner: unsafe { std::mem::zeroed() },
        }
    }

    /// Get the data type
    pub fn data_type(&self) -> DataType {
        DataType::from(self.inner.dataType)
    }

    /// Get the data type as raw value
    pub fn data_type_raw(&self) -> u32 {
        self.inner.dataType
    }

    /// Get the binding index
    pub fn binding_index(&self) -> i32 {
        self.inner.bindingIndex
    }

    /// Get the layer name
    pub fn layer_name(&self) -> Option<String> {
        unsafe {
            if self.inner.layerName.is_null() {
                None
            } else {
                CStr::from_ptr(self.inner.layerName)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }

    /// Check if this is an input layer
    pub fn is_input(&self) -> bool {
        self.inner.isInput != 0
    }

    /// Get access to the raw structure
    ///
    /// # Safety
    /// This provides access to the raw C structure. Use with caution.
    pub fn as_raw(&self) -> &NvDsInferLayerInfo {
        &self.inner
    }
}

impl Default for LayerInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal callback wrapper for logging
unsafe extern "C" fn logging_callback_wrapper(
    _handle: NvDsInferContextHandle,
    unique_id: u32,
    log_level: u32,
    log_message: *const c_char,
    _user_ctx: *mut std::ffi::c_void,
) {
    if log_message.is_null() {
        return;
    }

    let message = match CStr::from_ptr(log_message).to_str() {
        Ok(s) => s,
        Err(_) => return,
    };

    let level = LogLevel::from(log_level);

    // For now, just log to the standard Rust logging infrastructure
    match level {
        LogLevel::Error => log::error!("[InferContext {}] {}", unique_id, message),
        LogLevel::Warning => log::warn!("[InferContext {}] {}", unique_id, message),
        LogLevel::Info => log::info!("[InferContext {}] {}", unique_id, message),
        LogLevel::Debug => log::debug!("[InferContext {}] {}", unique_id, message),
    }
}

/// Utility function to check if a status indicates success
pub fn is_success(status: NvDsInferStatus) -> bool {
    status == NvDsInferStatus_NVDSINFER_SUCCESS
}

/// Convert NvDsInferStatus to a Result
pub fn status_to_result(status: NvDsInferStatus) -> Result<()> {
    if is_success(status) {
        Ok(())
    } else {
        Err(DeepStreamError::invalid_operation(&format!(
            "NvDsInfer operation failed with status: {}",
            status
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_infer_context() -> Result<()> {
        // Initialize logging
        _ = env_logger::try_init();

        let mut init_params = InferContextInitParams::new();
        init_params
            .set_gpu_id(0)
            .set_max_batch_size(16)
            .set_unique_id(1)
            .set_model_file_path("savant_deepstream/deepstream/assets/adaface_ir50_webface4m.onnx")?
            .set_engine_file_path(
                "savant_deepstream/deepstream/assets/adaface_ir50_webface4m.engine",
            )?
            .set_network_scale_factor(0.007843137254902f32)
            .set_net_input_order(InferTensorOrder::Nchw)
            .set_net_input_format(InferFormat::Bgr)
            .set_infer_input_dims(3, 112, 112)
            .set_output_layer_names(&["feature"])?;

        let _infer_context = InferContext::new_with_default_logging(&init_params)?;

        Ok(())
    }
}
