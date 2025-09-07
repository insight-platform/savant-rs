use crate::infer_context::{
    output::{FrameOutput, OutputLayer},
    LayerInfo,
};

use super::InferFormat;
use deepstream_sys::{
    NvDsInferContextBatchInput, NvDsInferContextBatchOutput, NvDsInferContextHandle,
    NvDsInferContext_ReleaseBatchOutput,
};

/// Network type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkType {
    Other,
}

impl From<u32> for NetworkType {
    fn from(value: u32) -> Self {
        match value {
            100 => NetworkType::Other,
            _ => unimplemented!("Unknown network type: {}", value),
        }
    }
}

impl From<NetworkType> for u32 {
    fn from(network_type: NetworkType) -> Self {
        match network_type {
            NetworkType::Other => 100,
        }
    }
}

/// Safe wrapper for NvDsInferContextBatchInput
pub struct BatchInput {
    pub(crate) inner: NvDsInferContextBatchInput,
    _frame_buffers: Vec<*mut std::ffi::c_void>,
}

impl BatchInput {
    /// Create a new batch input
    ///
    /// # Returns
    /// A tuple containing the BatchInput and a receiver for buffer return notifications
    pub fn new() -> Self {
        let mut batch_input = Self {
            inner: unsafe { std::mem::zeroed() },
            _frame_buffers: Vec::new(),
        };

        // Set up the internal callback
        batch_input.setup_internal_callback();

        batch_input
    }

    // /// Set up the internal callback for buffer return
    fn setup_internal_callback(&mut self) {
        //     // Create a boxed sender to pass to the callback
        //     //let sender_box = Box::new(self._sender.clone());
        //     //let sender_ptr = Box::into_raw(sender_box);

        //     //self.inner.returnInputFunc = Some(buffer_return_callback_wrapper);
        //     //self.inner.returnFuncData = sender_ptr as *mut std::ffi::c_void;
    }

    /// Set the input frames
    ///
    /// # Arguments
    /// * `frames` - Vector of frame buffer pointers
    /// * `format` - Input format
    /// * `pitch` - Input pitch in bytes
    pub fn set_frames(
        &mut self,
        frames: Vec<*mut std::ffi::c_void>,
        format: InferFormat,
        memory_row_width: usize,
    ) -> &mut Self {
        self._frame_buffers = frames;
        self.inner.inputFrames = self._frame_buffers.as_mut_ptr();
        self.inner.numInputFrames = self._frame_buffers.len() as u32;
        self.inner.inputFormat = format.into();
        self.inner.inputPitch = memory_row_width as u32;
        self
    }
}

impl Default for BatchInput {
    fn default() -> Self {
        Self::new()
    }
}

/// Safe wrapper for NvDsInferContextBatchOutput
pub struct BatchOutput {
    inner: NvDsInferContextBatchOutput,
    output_layers: Vec<LayerInfo>,
    pub(crate) context_handle: Option<NvDsInferContextHandle>,
}

impl BatchOutput {
    /// Create a new batch output
    pub(crate) fn new(output_layers: Vec<LayerInfo>) -> Self {
        Self {
            inner: unsafe { std::mem::zeroed() },
            output_layers,
            context_handle: None,
        }
    }

    /// Get the number of frames in this batch
    pub fn num_frames(&self) -> usize {
        self.inner.numFrames as usize
    }

    /// Get the number of output device buffers
    pub fn num_output_device_buffers(&self) -> usize {
        self.inner.numOutputDeviceBuffers as usize
    }

    /// Get output device buffers
    pub fn output_device_buffers(&self) -> Vec<*mut std::ffi::c_void> {
        if self.inner.outputDeviceBuffers.is_null() {
            return Vec::new();
        }

        unsafe {
            std::slice::from_raw_parts(
                self.inner.outputDeviceBuffers,
                self.inner.numOutputDeviceBuffers as usize,
            )
            .to_vec()
        }
    }

    /// Get the number of host buffers
    pub fn num_host_buffers(&self) -> usize {
        self.inner.numHostBuffers as usize
    }

    /// Get host buffers
    pub fn host_buffers(&self) -> Vec<*mut std::ffi::c_void> {
        if self.inner.hostBuffers.is_null() {
            return Vec::new();
        }

        unsafe {
            std::slice::from_raw_parts(self.inner.hostBuffers, self.inner.numHostBuffers as usize)
                .to_vec()
        }
    }

    /// Get access to the raw output structure
    ///
    /// # Safety
    /// This provides access to the raw C structure. Use with caution.
    pub fn as_raw(&self) -> &NvDsInferContextBatchOutput {
        &self.inner
    }

    /// Get access to the raw output structure mutably
    ///
    /// # Safety
    /// This provides mutable access to the raw C structure. Use with caution.
    pub fn as_raw_mut(&mut self) -> &mut NvDsInferContextBatchOutput {
        &mut self.inner
    }

    pub fn frame_outputs(&self) -> Vec<FrameOutput> {
        let mut frame_outputs = Vec::new();
        for f in 0..self.inner.numFrames as usize {
            let mut frame_output = FrameOutput::new();
            for (l, li) in self.output_layers.iter().enumerate() {
                let host_buffer_indx = l + 1;
                let device_buffer_indx = l;
                let host_buffer = self.host_buffers()[host_buffer_indx];
                let device_buffer = self.output_device_buffers()[device_buffer_indx];
                let data_length = li.byte_length();
                let host_data_ptr = unsafe { host_buffer.add(f * data_length) };
                let device_data_ptr = unsafe { device_buffer.add(f * data_length) };
                let output_layer = OutputLayer {
                    layer_name: li.layer_name().unwrap(),
                    dimensions: li.dims(),
                    data_type: li.data_type(),
                    byte_length: data_length,
                    device_address: device_data_ptr,
                    host_address: host_data_ptr,
                };
                frame_output.add_output_layer(output_layer);
            }
            frame_outputs.push(frame_output);
        }
        frame_outputs
    }
}

impl Drop for BatchOutput {
    fn drop(&mut self) {
        if let Some(handle) = self.context_handle {
            if !handle.is_null() {
                unsafe {
                    NvDsInferContext_ReleaseBatchOutput(handle, &mut self.inner);
                }
            }
        }
    }
}

// /// Internal callback for buffer return that sends to mpsc channel
// unsafe extern "C" fn buffer_return_callback_wrapper(data: *mut std::ffi::c_void) {
//     if data.is_null() {
//         return;
//     }

//     // Cast the data pointer back to the boxed sender
//     let sender_box = data as *mut Box<channel::Sender<()>>;
//     if let Some(sender_ref) = std::ptr::NonNull::new(sender_box) {
//         let sender_box = Box::from_raw(sender_ref.as_ptr());
//         // Send a signal to unblock the receiver
//         let _ = sender_box.send(());
//         // Don't drop the sender_box here as it's managed by BatchInput
//         std::mem::forget(sender_box);
//     }
// }
