use crate::{DeepStreamError, Result};

pub mod init_params;
pub mod io;
pub mod layer_info;
pub mod output;
use deepstream_sys::{
    NvDsInferContextHandle,
    NvDsInferContextLoggingFunc,
    NvDsInferContext_Create,
    NvDsInferContext_DequeueOutputBatch,
    NvDsInferContext_Destroy,
    NvDsInferContext_FillLayersInfo,
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
pub use init_params::{InferContextInitParams, InferFormat, InferNetworkMode, InferTensorOrder};
pub use io::{BatchInput, BatchOutput, NetworkType};
pub use layer_info::LayerInfo;
use std::{ffi::CStr, os::raw::c_char, ptr};

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

/// Safe wrapper for NvDsInferContext
///
/// This struct provides safe access to the DeepStream inference context,
/// managing the underlying C handle and ensuring proper cleanup.
pub struct InferContext {
    handle: NvDsInferContextHandle,
    init_params: InferContextInitParams,
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
        init_params: InferContextInitParams,
        user_ctx: Option<*mut std::ffi::c_void>,
        log_func: NvDsInferContextLoggingFunc,
    ) -> Result<Self> {
        let mut handle: NvDsInferContextHandle = ptr::null_mut();
        let user_ctx_ptr = user_ctx.unwrap_or(ptr::null_mut());

        //let init_params = Box::new(init_params);
        let init_params_ptr = init_params.as_raw() as *const _ as *mut _;
        //std::mem::forget(init_params);
        let status = unsafe {
            NvDsInferContext_Create(&mut handle, init_params_ptr, user_ctx_ptr, log_func)
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
            init_params,
        })
    }

    /// Create a new inference context with default logging
    ///
    /// # Arguments
    /// * `init_params` - Initialization parameters for the context
    ///
    /// # Returns
    /// A new `InferContext` instance or an error if creation failed
    pub fn new_with_default_logging(init_params: InferContextInitParams) -> Result<Self> {
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
        if batch_input.inner.numInputFrames > self.init_params.max_batch_size() {
            return Err(DeepStreamError::invalid_operation(&format!(
                "Enqueued batch size {} exceeds maximum network batch size {}",
                batch_input.inner.numInputFrames,
                self.init_params.max_batch_size()
            )));
        }

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
        let mut batch_output = BatchOutput::new(self.output_layers());

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
    pub fn layers(&self) -> Vec<LayerInfo> {
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

    pub fn output_layers(&self) -> Vec<LayerInfo> {
        self.layers()
            .into_iter()
            .filter(|layer| !layer.is_input())
            .collect()
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

    // split message by \n
    let messages = message.split('\n');
    let level = LogLevel::from(log_level);
    for message in messages {
        match level {
            LogLevel::Error => log::error!("[InferContext {}] {}", unique_id, message),
            LogLevel::Warning => log::warn!("[InferContext {}] {}", unique_id, message),
            LogLevel::Info => log::info!("[InferContext {}] {}", unique_id, message),
            LogLevel::Debug => log::debug!("[InferContext {}] {}", unique_id, message),
        }
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
    use std::ffi::c_void;

    use super::*;
    use anyhow::Result;
    use cudarc::runtime::sys as cuda;

    #[test]
    fn test_infer_context() -> Result<()> {
        const WIDTH: usize = 12;
        const HEIGHT: usize = 12;
        const CHANNELS: usize = 3;
        const BATCH_SIZE: u32 = 4;
        const GPU_ID: i32 = 0;
        // Initialize logging
        //_ = env_logger::try_init();
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();

        let mut init_params = InferContextInitParams::new();
        // let model_name = "age_gender_mobilenet_v2_dynBatch";
        let model_name = "identity";
        init_params
            .set_gpu_id(GPU_ID as u32)
            .set_max_batch_size(BATCH_SIZE)
            .set_unique_id(1)
            .set_network_mode(InferNetworkMode::FP16)
            .set_onnx_file_path(&format!("assets/{}.onnx", model_name))?
            .set_engine_file_path(&format!(
                "assets/{}.onnx_b{}_gpu{}_fp16.engine",
                model_name, BATCH_SIZE, GPU_ID
            ))?
            .set_net_input_order(InferTensorOrder::NCHW)
            .set_net_input_format(InferFormat::RGB)
            .set_infer_input_dims(CHANNELS, WIDTH, HEIGHT);

        //.set_output_layer_names(&["feature"])?;
        log::debug!("Init params: {:?}", init_params);
        let mut infer_context = InferContext::new_with_default_logging(init_params)?;

        // Select device (creates the primary context for the runtime API)
        unsafe {
            cuda::cudaSetDevice(GPU_ID).result()?;
        } // maps cudaError -> Result

        let mut dptr: *mut c_void = ptr::null_mut();
        let mut dptr2: *mut c_void = ptr::null_mut();
        let mut row_width: usize = 0;

        let row_bytes = WIDTH * CHANNELS * std::mem::size_of::<u8>();

        // Allocate pitched 2D memory: height rows, each at least row_bytes
        unsafe {
            cuda::cudaMallocPitch(&mut dptr, &mut row_width as *mut usize, row_bytes, HEIGHT)
                .result()?;
            cuda::cudaMemset(dptr, 1, row_width * HEIGHT);

            cuda::cudaMallocPitch(&mut dptr2, &mut row_width as *mut usize, row_bytes, HEIGHT)
                .result()?;
            cuda::cudaMemset(dptr2, 3, row_width * HEIGHT);
        }

        log::info!("Pitch bytes: {}", row_width);

        // Get layer information for proper output access
        let layer_infos = infer_context.layers();
        log::info!("Number of layers: {}", layer_infos.len());

        // Log layer information
        for (i, layer_info) in layer_infos.iter().enumerate() {
            log::info!(
                "Layer {}: name={:?}, is_input={}, data_type={:?}, dims={:?}, binding_index={}",
                i,
                layer_info.layer_name(),
                layer_info.is_input(),
                layer_info.data_type(),
                layer_info.dims(),
                layer_info.binding_index(),
            );
        }

        let (mut batch_input, _rcvr) = BatchInput::new();
        batch_input.set_frames(vec![dptr, dptr2], InferFormat::RGB, row_width);
        infer_context.queue_input_batch(&batch_input)?;

        let batch_output = infer_context.dequeue_output_batch()?;
        log::info!(
            "{} frames, {} host buffers, {} device buffers",
            batch_output.num_frames(),
            batch_output.num_host_buffers(),
            batch_output.num_output_device_buffers()
        );
        log::info!("Host buffers: {:?}", batch_output.host_buffers());
        log::info!("Device buffers: {:?}", batch_output.output_device_buffers());

        let frame_outputs = batch_output.frame_outputs();
        log::info!("Frame outputs: {:#?}", frame_outputs);
        for frame_output in frame_outputs {
            for layer_output in frame_output.output_layers() {
                let tensor = unsafe {
                    std::slice::from_raw_parts(
                        layer_output.host_address as *const f32,
                        layer_output.dimensions.num_elements as usize,
                    )
                };
                log::info!(
                    "Tensor name {}, shape {:?}: {:?}",
                    layer_output.layer_name,
                    layer_output.dimensions,
                    tensor
                );
            }
        }

        Ok(())
    }
}
