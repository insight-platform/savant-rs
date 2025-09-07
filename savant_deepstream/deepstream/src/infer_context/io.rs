use super::InferFormat;
use crossbeam::channel;
use deepstream_sys::{
    NvDsInferContextBatchInput, NvDsInferContextBatchOutput, NvDsInferContextHandle,
    NvDsInferContext_ReleaseBatchOutput,
};

/// Safe wrapper for NvDsInferContextBatchInput
pub struct BatchInput {
    pub(crate) inner: NvDsInferContextBatchInput,
    _frame_buffers: Vec<*mut std::ffi::c_void>,
    _sender: channel::Sender<()>,
}

impl BatchInput {
    /// Create a new batch input
    ///
    /// # Returns
    /// A tuple containing the BatchInput and a receiver for buffer return notifications
    pub fn new() -> (Self, channel::Receiver<()>) {
        let (sender, receiver) = channel::unbounded();

        let mut batch_input = Self {
            inner: unsafe { std::mem::zeroed() },
            _frame_buffers: Vec::new(),
            _sender: sender,
        };

        // Set up the internal callback
        batch_input.setup_internal_callback();

        (batch_input, receiver)
    }

    /// Set up the internal callback for buffer return
    fn setup_internal_callback(&mut self) {
        // Create a boxed sender to pass to the callback
        let sender_box = Box::new(self._sender.clone());
        let sender_ptr = Box::into_raw(sender_box);

        self.inner.returnInputFunc = Some(buffer_return_callback_wrapper);
        self.inner.returnFuncData = sender_ptr as *mut std::ffi::c_void;
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
        pitch: u32,
    ) -> &mut Self {
        self._frame_buffers = frames;
        self.inner.inputFrames = self._frame_buffers.as_mut_ptr();
        self.inner.numInputFrames = self._frame_buffers.len() as u32;
        self.inner.inputFormat = format.into();
        self.inner.inputPitch = pitch;
        self
    }
}

impl Default for BatchInput {
    fn default() -> Self {
        Self::new().0
    }
}

/// Safe wrapper for NvDsInferContextBatchOutput
pub struct BatchOutput {
    inner: NvDsInferContextBatchOutput,
    pub(crate) context_handle: Option<NvDsInferContextHandle>,
}

impl BatchOutput {
    /// Create a new batch output
    pub fn new() -> Self {
        Self {
            inner: unsafe { std::mem::zeroed() },
            context_handle: None,
        }
    }

    /// Get the number of output device buffers
    pub fn num_output_device_buffers(&self) -> u32 {
        self.inner.numOutputDeviceBuffers
    }

    pub fn output_device_buffers(&self) -> Vec<*mut std::ffi::c_void> {
        unsafe {
            std::slice::from_raw_parts(
                self.inner.outputDeviceBuffers,
                self.inner.numOutputDeviceBuffers as usize,
            )
            .to_vec()
        }
    }

    /// Get the number of host buffers
    pub fn num_host_buffers(&self) -> u32 {
        self.inner.numHostBuffers
    }

    pub fn host_buffers(&self) -> Vec<*mut std::ffi::c_void> {
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

/// Internal callback for buffer return that sends to mpsc channel
unsafe extern "C" fn buffer_return_callback_wrapper(data: *mut std::ffi::c_void) {
    if data.is_null() {
        return;
    }

    // Cast the data pointer back to the boxed sender
    let sender_box = data as *mut Box<channel::Sender<()>>;
    if let Some(sender_ref) = std::ptr::NonNull::new(sender_box) {
        let sender_box = Box::from_raw(sender_ref.as_ptr());
        // Send a signal to unblock the receiver
        let _ = sender_box.send(());
        // Don't drop the sender_box here as it's managed by BatchInput
        std::mem::forget(sender_box);
    }
}
