use crate::{infer_tensor_meta::InferDims, DataType};

#[derive(Debug)]
pub struct OutputLayer {
    pub layer_name: String,
    pub dimensions: InferDims,
    pub data_type: DataType,
    pub byte_length: usize,
    pub device_address: *mut std::ffi::c_void,
    pub host_address: *mut std::ffi::c_void,
}

#[derive(Debug)]
pub struct FrameOutput {
    output_layers: Vec<OutputLayer>,
}

impl FrameOutput {
    pub fn new() -> Self {
        Self {
            output_layers: Vec::new(),
        }
    }

    pub fn add_output_layer(&mut self, output_layer: OutputLayer) {
        self.output_layers.push(output_layer);
    }

    pub fn output_layers(&self) -> &Vec<OutputLayer> {
        &self.output_layers
    }
}
