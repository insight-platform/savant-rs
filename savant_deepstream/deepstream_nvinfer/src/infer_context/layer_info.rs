use super::DataType;
use crate::infer_tensor_meta::InferDims;
use deepstream_sys::NvDsInferLayerInfo;
use std::ffi::CStr;

/// Safe wrapper for NvDsInferLayerInfo
#[derive(Clone)]
pub struct LayerInfo(NvDsInferLayerInfo);

impl LayerInfo {
    /// Create a new layer info structure
    pub fn new() -> Self {
        Self(unsafe { std::mem::zeroed() })
    }

    pub fn dims(&self) -> InferDims {
        unsafe { InferDims::from(&self.0.__bindgen_anon_1.dims) }
    }

    /// Get the data type
    pub fn data_type(&self) -> DataType {
        DataType::from(self.0.dataType)
    }

    pub fn byte_length(&self) -> usize {
        // multiply all dimensions
        let elements = self.dims().num_elements as usize;
        let data_type = self.data_type();
        match data_type {
            DataType::Float => elements * 4,
            DataType::Half => elements * 2,
            DataType::Int8 => elements,
            DataType::Int32 => elements * 4,
        }
    }

    /// Get the data type as raw value
    pub fn data_type_raw(&self) -> u32 {
        self.0.dataType
    }

    /// Get the binding index
    pub fn binding_index(&self) -> i32 {
        self.0.bindingIndex
    }

    /// Get the layer name
    pub fn layer_name(&self) -> Option<String> {
        unsafe {
            if self.0.layerName.is_null() {
                None
            } else {
                CStr::from_ptr(self.0.layerName)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }

    /// Check if this is an input layer
    pub fn is_input(&self) -> bool {
        self.0.isInput != 0
    }

    /// Get access to the raw structure
    ///
    /// # Safety
    /// This provides access to the raw C structure. Use with caution.
    pub fn as_raw(&self) -> &NvDsInferLayerInfo {
        &self.0
    }
}

impl Default for LayerInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LayerInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerInfo")
            .field("data_type", &self.data_type())
            .field("binding_index", &self.binding_index())
            .field("layer_name", &self.layer_name())
            .field("is_input", &self.is_input())
            .field("dims", &self.dims())
            .finish()
    }
}
