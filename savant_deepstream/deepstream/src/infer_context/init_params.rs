use crate::{DeepStreamError, Result};
use deepstream_sys::{
    NvDsInferContextInitParams,
    NvDsInferContext_ResetInitParams,
    // Format constants
    NvDsInferFormat,
    NvDsInferFormat_NvDsInferFormat_BGR,
    NvDsInferFormat_NvDsInferFormat_BGRx,
    NvDsInferFormat_NvDsInferFormat_GRAY,
    NvDsInferFormat_NvDsInferFormat_RGB,
    NvDsInferFormat_NvDsInferFormat_RGBA,
    NvDsInferFormat_NvDsInferFormat_Tensor,
    NvDsInferFormat_NvDsInferFormat_Unknown,
    // Network mode constants
    NvDsInferNetworkMode,
    NvDsInferNetworkMode_NvDsInferNetworkMode_BEST,
    NvDsInferNetworkMode_NvDsInferNetworkMode_FP16,
    NvDsInferNetworkMode_NvDsInferNetworkMode_FP32,
    NvDsInferNetworkMode_NvDsInferNetworkMode_INT8,
    // Network type constants
    NvDsInferNetworkType_NvDsInferNetworkType_Other,
    // Tensor order constants
    NvDsInferTensorOrder,
    NvDsInferTensorOrder_NvDsInferTensorOrder_kNC,
    NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW,
    NvDsInferTensorOrder_NvDsInferTensorOrder_kNHWC,
};
use std::{ffi::CString, ptr};

/// Input format enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferFormat {
    /// 24-bit interleaved R-G-B format
    Rgb,
    /// 24-bit interleaved B-G-R format
    Bgr,
    /// 8-bit Luma format
    Gray,
    /// 32-bit interleaved R-G-B-A format
    Rgba,
    /// 32-bit interleaved B-G-R-x format
    Bgrx,
    /// NCHW planar tensor format
    Tensor,
    /// Unknown format
    Unknown,
}

impl From<InferFormat> for NvDsInferFormat {
    fn from(format: InferFormat) -> Self {
        match format {
            InferFormat::Rgb => NvDsInferFormat_NvDsInferFormat_RGB,
            InferFormat::Bgr => NvDsInferFormat_NvDsInferFormat_BGR,
            InferFormat::Gray => NvDsInferFormat_NvDsInferFormat_GRAY,
            InferFormat::Rgba => NvDsInferFormat_NvDsInferFormat_RGBA,
            InferFormat::Bgrx => NvDsInferFormat_NvDsInferFormat_BGRx,
            InferFormat::Tensor => NvDsInferFormat_NvDsInferFormat_Tensor,
            InferFormat::Unknown => NvDsInferFormat_NvDsInferFormat_Unknown,
        }
    }
}

impl From<NvDsInferFormat> for InferFormat {
    fn from(format: NvDsInferFormat) -> Self {
        match format {
            x if x == NvDsInferFormat_NvDsInferFormat_RGB => InferFormat::Rgb,
            x if x == NvDsInferFormat_NvDsInferFormat_BGR => InferFormat::Bgr,
            x if x == NvDsInferFormat_NvDsInferFormat_GRAY => InferFormat::Gray,
            x if x == NvDsInferFormat_NvDsInferFormat_RGBA => InferFormat::Rgba,
            x if x == NvDsInferFormat_NvDsInferFormat_BGRx => InferFormat::Bgrx,
            x if x == NvDsInferFormat_NvDsInferFormat_Tensor => InferFormat::Tensor,
            _ => InferFormat::Unknown,
        }
    }
}

/// Tensor order enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferTensorOrder {
    /// NCHW format: Number of batches, Channels, Height, Width
    Nchw,
    /// NHWC format: Number of batches, Height, Width, Channels
    Nhwc,
    /// NC format: Number of batches, Channels (for 1D data)
    Nc,
}

impl From<InferTensorOrder> for NvDsInferTensorOrder {
    fn from(order: InferTensorOrder) -> Self {
        match order {
            InferTensorOrder::Nchw => NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW,
            InferTensorOrder::Nhwc => NvDsInferTensorOrder_NvDsInferTensorOrder_kNHWC,
            InferTensorOrder::Nc => NvDsInferTensorOrder_NvDsInferTensorOrder_kNC,
        }
    }
}

impl From<NvDsInferTensorOrder> for InferTensorOrder {
    fn from(order: NvDsInferTensorOrder) -> Self {
        match order {
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW => InferTensorOrder::Nchw,
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNHWC => InferTensorOrder::Nhwc,
            x if x == NvDsInferTensorOrder_NvDsInferTensorOrder_kNC => InferTensorOrder::Nc,
            _ => InferTensorOrder::Nchw, // Default to NCHW for unknown values
        }
    }
}

/// Network mode enumeration for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferNetworkMode {
    /// 32-bit floating point precision
    Fp32,
    /// 8-bit integer precision
    Int8,
    /// 16-bit floating point precision
    Fp16,
    /// Best precision available
    Best,
}

impl From<InferNetworkMode> for NvDsInferNetworkMode {
    fn from(mode: InferNetworkMode) -> Self {
        match mode {
            InferNetworkMode::Fp32 => NvDsInferNetworkMode_NvDsInferNetworkMode_FP32,
            InferNetworkMode::Int8 => NvDsInferNetworkMode_NvDsInferNetworkMode_INT8,
            InferNetworkMode::Fp16 => NvDsInferNetworkMode_NvDsInferNetworkMode_FP16,
            InferNetworkMode::Best => NvDsInferNetworkMode_NvDsInferNetworkMode_BEST,
        }
    }
}

impl From<NvDsInferNetworkMode> for InferNetworkMode {
    fn from(mode: NvDsInferNetworkMode) -> Self {
        match mode {
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_FP32 => InferNetworkMode::Fp32,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_INT8 => InferNetworkMode::Int8,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_FP16 => InferNetworkMode::Fp16,
            x if x == NvDsInferNetworkMode_NvDsInferNetworkMode_BEST => InferNetworkMode::Best,
            _ => InferNetworkMode::Fp32, // Default to FP32 for unknown values
        }
    }
}

/// Safe wrapper for NvDsInferContextInitParams
///
/// This struct provides a safe way to configure inference context initialization
/// parameters with proper memory management for string fields.
pub struct InferContextInitParams {
    inner: NvDsInferContextInitParams,
    // Keep owned strings to prevent deallocation
    _owned_strings: Vec<CString>,
    // Keep owned output layer name pointers to prevent deallocation
    _output_layer_names: Vec<CString>,
    _output_layer_pointers: Vec<*mut std::os::raw::c_char>,
}

impl InferContextInitParams {
    /// Create a new initialization parameters structure with default values
    pub fn new() -> Self {
        let mut inner = unsafe { std::mem::zeroed() };
        unsafe {
            NvDsInferContext_ResetInitParams(&mut inner);
        }
        inner.networkType = NvDsInferNetworkType_NvDsInferNetworkType_Other;

        Self {
            inner,
            _owned_strings: Vec::new(),
            _output_layer_names: Vec::new(),
            _output_layer_pointers: Vec::new(),
        }
    }

    // int8CalibrationFilePath
    pub fn set_int8_calibration_file_path(&mut self, path: &str) -> Result<&mut Self> {
        log::info!("Setting int8 calibration file path: {}", path);
        let c_string = CString::new(path)?;
        unsafe {
            let src = c_string.as_ptr();
            let dest = self.inner.int8CalibrationFilePath.as_mut_ptr();
            let len = std::cmp::min(c_string.as_bytes().len(), 4096 - 1);
            ptr::copy_nonoverlapping(src, dest, len);
            *dest.add(len) = 0; // Null terminate
        }
        self._owned_strings.push(c_string);
        Ok(self)
    }

    /// Set the unique ID for the context
    pub fn set_unique_id(&mut self, id: u32) -> &mut Self {
        log::info!("Setting unique ID: {}", id);
        self.inner.uniqueID = id;
        self
    }

    /// Set the network mode (INT8, FP16, FP32, BEST)
    pub fn set_network_mode(&mut self, mode: InferNetworkMode) -> &mut Self {
        log::info!("Setting network mode: {:?}", mode);
        self.inner.networkMode = mode.into();
        self
    }

    /// Set the ONNX file path
    pub fn set_onnx_file_path(&mut self, path: &str) -> Result<&mut Self> {
        log::info!("Setting ONNX file path: {}", path);
        let c_string = CString::new(path)?;
        unsafe {
            let src = c_string.as_ptr();
            let dest = self.inner.onnxFilePath.as_mut_ptr();
            let len = std::cmp::min(c_string.as_bytes().len(), 4096 - 1);
            ptr::copy_nonoverlapping(src, dest, len);
            *dest.add(len) = 0; // Null terminate
        }
        self._owned_strings.push(c_string);
        Ok(self)
    }

    /// Set the engine file path
    pub fn set_engine_file_path(&mut self, path: &str) -> Result<&mut Self> {
        log::info!("Setting engine file path: {}", path);
        let c_string = CString::new(path)?;
        unsafe {
            let src = c_string.as_ptr();
            let dest = self.inner.modelEngineFilePath.as_mut_ptr();
            let len = std::cmp::min(c_string.as_bytes().len(), 4096 - 1);
            ptr::copy_nonoverlapping(src, dest, len);
            *dest.add(len) = 0; // Null terminate
        }
        self._owned_strings.push(c_string);
        Ok(self)
    }

    /// Set the maximum batch size
    pub fn set_max_batch_size(&mut self, size: u32) -> &mut Self {
        log::info!("Setting maximum batch size: {}", size);
        self.inner.maxBatchSize = size;
        self
    }

    /// Set the GPU ID
    pub fn set_gpu_id(&mut self, id: u32) -> &mut Self {
        log::info!("Setting GPU ID: {}", id);
        self.inner.gpuID = id;
        self
    }

    /// Set the network type (raw value)
    pub fn set_network_type_raw(&mut self, network_type: u32) -> &mut Self {
        log::info!("Setting network type raw: {}", network_type);
        self.inner.networkType = network_type;
        self
    }

    /// Set the network scale factor
    pub fn set_network_scale_factor(&mut self, factor: f32) -> &mut Self {
        log::info!("Setting network scale factor: {}", factor);
        self.inner.networkScaleFactor = factor;
        self
    }

    pub fn set_infer_input_dims(&mut self, c: u32, h: u32, w: u32) -> &mut Self {
        log::info!("Setting infer input dimensions: {}x{}x{}", c, h, w);
        self.inner.inferInputDims.c = c;
        self.inner.inferInputDims.h = h;
        self.inner.inferInputDims.w = w;
        self
    }

    /// Set whether to use DLA (Deep Learning Accelerator)
    pub fn set_use_dla(&mut self, use_dla: bool) -> &mut Self {
        log::info!("Setting use DLA: {}", use_dla);
        self.inner.useDLA = if use_dla { 1 } else { 0 };
        self
    }

    /// Set the DLA core to use
    pub fn set_dla_core(&mut self, core: i32) -> &mut Self {
        log::info!("Setting DLA core: {}", core);
        self.inner.dlaCore = core;
        self
    }

    /// Set the output buffer pool size
    pub fn set_output_buffer_pool_size(&mut self, size: u32) -> &mut Self {
        log::info!("Setting output buffer pool size: {}", size);
        self.inner.outputBufferPoolSize = size;
        self
    }

    /// Set the network input tensor order
    pub fn set_net_input_order(&mut self, order: InferTensorOrder) -> &mut Self {
        log::info!("Setting network input order: {:?}", order);
        self.inner.netInputOrder = order.into();
        self
    }

    pub fn set_net_input_format(&mut self, format: InferFormat) -> &mut Self {
        log::info!("Setting network input format: {:?}", format);
        self.inner.networkInputFormat = format.into();
        self
    }

    pub fn set_mean_image_file_path(&mut self, path: &str) -> Result<&mut Self> {
        log::info!("Setting mean image file path: {}", path);
        let c_string = CString::new(path)?;
        unsafe {
            let src = c_string.as_ptr();
            let dest = self.inner.meanImageFilePath.as_mut_ptr();
            let len = std::cmp::min(c_string.as_bytes().len(), 4096 - 1);
            ptr::copy_nonoverlapping(src, dest, len);
            *dest.add(len) = 0; // Null terminate
        }
        self._owned_strings.push(c_string);
        Ok(self)
    }

    pub fn set_offsets(&mut self, offsets: &[f32]) -> Result<&mut Self> {
        if offsets.len() > 4 {
            return Err(DeepStreamError::invalid_operation(
                "Offsets must be less than 4",
            ));
        }
        log::info!("Setting offsets: {:?}", offsets);
        self.inner.numOffsets = offsets.len() as u32;
        for (i, offset) in offsets.iter().enumerate() {
            self.inner.offsets[i] = *offset;
        }
        Ok(self)
    }

    pub fn set_output_layer_names(&mut self, names: &[&str]) -> Result<&mut Self> {
        log::info!("Setting output layer names: {}", names.join(", "));
        // Convert string slice to C strings and store them
        let c_strings: std::result::Result<Vec<CString>, DeepStreamError> = names
            .iter()
            .map(|s| {
                CString::new(*s).map_err(|e| {
                    DeepStreamError::invalid_operation(&format!(
                        "Invalid string in output layer names: {}",
                        e
                    ))
                })
            })
            .collect();
        let c_strings = c_strings?;

        // Store the C strings to prevent deallocation
        self._output_layer_names.extend(c_strings);

        // Clear previous output layer pointers and create new ones
        self._output_layer_pointers.clear();
        self._output_layer_pointers = self
            ._output_layer_names
            .iter()
            .map(|s| s.as_ptr() as *mut std::os::raw::c_char)
            .collect();

        // Set the fields in the inner structure
        self.inner.numOutputLayers = self._output_layer_names.len() as u32;
        if !self._output_layer_pointers.is_empty() {
            self.inner.outputLayerNames = self._output_layer_pointers.as_mut_ptr();
        } else {
            self.inner.outputLayerNames = ptr::null_mut();
        }

        Ok(self)
    }

    pub fn set_workspace_size(&mut self, size: u32) -> &mut Self {
        log::info!("Setting workspace size: {}", size);
        self.inner.workspaceSize = size;
        self
    }

    pub fn set_output_host_copy(&mut self, host_copy: bool) -> &mut Self {
        log::info!("Setting output device-to-host copy: {}", host_copy);
        self.inner.disableOutputHostCopy = if !host_copy { 1 } else { 0 };
        self
    }

    pub fn set_auto_inc_mem(&mut self, auto_inc_mem: bool) -> &mut Self {
        log::info!(
            "Setting auto buffer pool increment memory: {}",
            auto_inc_mem
        );
        self.inner.autoIncMem = if auto_inc_mem { 1 } else { 0 };
        self
    }

    pub fn set_max_gpu_mem_percentage(&mut self, max_gpu_mem_per: f64) -> &mut Self {
        log::info!(
            "Setting max GPU memory percentage when expanding buffer pool: {}",
            max_gpu_mem_per
        );
        self.inner.maxGPUMemPer = max_gpu_mem_per;
        self
    }

    pub fn set_dump_ip_tensor(&mut self, dump_ip_tensor: bool) -> &mut Self {
        log::info!("Setting dump input tensor: {}", dump_ip_tensor);
        self.inner.dumpIpTensor = if dump_ip_tensor { 1 } else { 0 };
        self
    }

    pub fn set_dump_op_tensor(&mut self, dump_op_tensor: bool) -> &mut Self {
        log::info!("Setting dump output tensor: {}", dump_op_tensor);
        self.inner.dumpOpTensor = if dump_op_tensor { 1 } else { 0 };
        self
    }

    /// Get access to the inner raw structure
    ///
    /// # Safety
    /// This provides access to the raw C structure. Use with caution.
    pub fn as_raw(&self) -> &NvDsInferContextInitParams {
        &self.inner
    }
}

impl std::fmt::Debug for InferContextInitParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Helper function to safely convert C string array to String
        let c_str_array_to_string = |arr: &[std::os::raw::c_char]| -> String {
            unsafe {
                std::ffi::CStr::from_ptr(arr.as_ptr())
                    .to_string_lossy()
                    .to_string()
            }
        };

        f.debug_struct("InferContextInitParams")
            .field("unique_id", &self.inner.uniqueID)
            .field("gpu_id", &self.inner.gpuID)
            .field("max_batch_size", &self.inner.maxBatchSize)
            .field(
                "network_mode",
                &InferNetworkMode::from(self.inner.networkMode),
            )
            .field("network_type", &self.inner.networkType)
            .field("network_scale_factor", &self.inner.networkScaleFactor)
            .field("network_input_order", &self.inner.netInputOrder)
            .field("network_input_format", &self.inner.networkInputFormat)
            .field(
                "infer_input_dims",
                &format!(
                    "{}x{}x{}",
                    self.inner.inferInputDims.c,
                    self.inner.inferInputDims.h,
                    self.inner.inferInputDims.w
                ),
            )
            .field("auto_host_copy", &(self.inner.disableOutputHostCopy != 0))
            .field("use_dla", &(self.inner.useDLA != 0))
            .field("dla_core", &self.inner.dlaCore)
            .field("output_buffer_pool_size", &self.inner.outputBufferPoolSize)
            .field("workspace_size", &self.inner.workspaceSize)
            .field("auto_inc_mem", &(self.inner.autoIncMem != 0))
            .field("max_gpu_mem_percentage", &self.inner.maxGPUMemPer)
            .field("dump_ip_tensor", &(self.inner.dumpIpTensor != 0))
            .field("dump_op_tensor", &(self.inner.dumpOpTensor != 0))
            .field(
                "model_file_path",
                &c_str_array_to_string(&self.inner.modelFilePath),
            )
            .field(
                "onnx_file_path",
                &c_str_array_to_string(&self.inner.onnxFilePath),
            )
            .field(
                "int8_calibration_file_path",
                &c_str_array_to_string(&self.inner.int8CalibrationFilePath),
            )
            .field(
                "mean_image_file_path",
                &c_str_array_to_string(&self.inner.meanImageFilePath),
            )
            .field("num_output_layers", &self.inner.numOutputLayers)
            .field("offsets", &format!("{:?}", self.inner.offsets))
            .field("num_offsets", &self.inner.numOffsets)
            .finish()
    }
}

impl Default for InferContextInitParams {
    fn default() -> Self {
        Self::new()
    }
}
