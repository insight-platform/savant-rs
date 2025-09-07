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

    /// Set the network mode (INT8, FP16, FP32)
    pub fn set_network_mode(&mut self, mode: u32) -> &mut Self {
        log::info!("Setting network mode: {}", mode);
        self.inner.networkMode = mode;
        self
    }

    /// Set the model file path
    pub fn set_model_file_path(&mut self, path: &str) -> Result<&mut Self> {
        log::info!("Setting model file path: {}", path);
        let c_string = CString::new(path)?;
        unsafe {
            let src = c_string.as_ptr();
            let dest = self.inner.modelFilePath.as_mut_ptr();
            let len = std::cmp::min(c_string.as_bytes().len(), 4096 - 1);
            ptr::copy_nonoverlapping(src, dest, len);
            *dest.add(len) = 0; // Null terminate
        }
        self._owned_strings.push(c_string);
        Ok(self)
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

    pub fn set_offsets(&mut self, offsets: [f32; 4], num_offsets: u32) -> &mut Self {
        log::info!("Setting offsets: {:?}", offsets);
        self.inner.numOffsets = num_offsets;
        self.inner.offsets = offsets;
        self
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

impl Default for InferContextInitParams {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_infer_context_init_params() -> Result<()> {
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

        // Test that the tensor orders were set correctly
        assert_eq!(
            init_params.as_raw().netInputOrder,
            NvDsInferTensorOrder_NvDsInferTensorOrder_kNCHW
        );

        // Test output layer names
        let raw_params = init_params.as_raw();
        assert_eq!(raw_params.numOutputLayers, 1);
        assert!(!raw_params.outputLayerNames.is_null());

        Ok(())
    }
}
