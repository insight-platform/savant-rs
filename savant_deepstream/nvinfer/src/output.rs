//! Output types for batch inference results.

use crate::batch_meta_builder::clear_all_frame_objects;
use crate::nvinfer_types::DataType;
use deepstream::InferDims;
use deepstream_sys::GstBuffer;
use std::ffi::c_void;

/// Zero-copy view into a single output tensor.
/// Valid while the parent [`BatchInferenceOutput`] is alive.
///
/// # Safety
/// Safe to send between threads when transferred with the owning `BatchInferenceOutput`;
/// the pointers remain valid until the output is dropped.
///
/// **Important**: if the owning `BatchInferenceOutput` was created with
/// [`MetaClearPolicy::After`] or [`MetaClearPolicy::Both`], dropping the
/// `BatchInferenceOutput` will release object metadata and invalidate these
/// pointers.  Consume all `TensorView`s before dropping the owning output.
#[derive(Debug)]
pub struct TensorView {
    /// Output layer name.
    pub name: String,
    /// Tensor dimensions.
    pub dims: InferDims,
    /// Data type.
    pub data_type: DataType,
    /// Host buffer pointer (may be null if host copy disabled).
    pub host_ptr: *const c_void,
    /// Device buffer pointer.
    pub device_ptr: *const c_void,
    /// Byte length of the tensor.
    pub byte_length: usize,
}

impl TensorView {
    /// Interpret host data as a typed slice (zero-copy).
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and properly aligned for `T`.
    /// The slice length is derived from `byte_length / size_of::<T>()`.
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        if self.host_ptr.is_null() || self.byte_length == 0 {
            return &[];
        }
        let len = self.byte_length / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.host_ptr as *const T, len)
    }
}

/// Per-element inference output for one ROI in one frame.
#[derive(Debug)]
pub struct ElementOutput {
    /// User-provided frame ID from [`SavantIdMeta`] (if present).
    pub frame_id: Option<i64>,
    /// ROI identifier from [`crate::roi::Roi::id`].
    ///
    /// `None` when no explicit ROIs were supplied and the full frame was used.
    pub roi_id: Option<i64>,
    /// Output tensors by layer name.
    pub tensors: Vec<TensorView>,
}

/// Owns the output gst::Sample (and thus the buffer); tensor views borrow from it.
///
/// When constructed with [`MetaClearPolicy::After`] or
/// [`MetaClearPolicy::Both`], dropping this value calls
/// `nvds_clear_obj_meta_list` on every frame, returning all
/// `NvDsObjectMeta` entries to the DeepStream pool.  This invalidates the
/// raw pointers inside any [`TensorView`]s still alive, so all `TensorView`s
/// should be consumed before dropping `BatchInferenceOutput`.
pub struct BatchInferenceOutput {
    batch_id: u64,
    elements: Vec<ElementOutput>,
    /// When `true`, clear all frame object metas when this value is dropped.
    clear_on_drop: bool,
    /// Holds the GStreamer sample (and thus the buffer) alive.
    /// **Must be declared last** so that it is still valid when `Drop::drop`
    /// runs and clears the object metas.
    _sample: gstreamer::Sample,
}

// Safe to send: ownership transfer; pointers valid until BatchInferenceOutput is dropped.
unsafe impl Send for TensorView {}
unsafe impl Send for ElementOutput {}
unsafe impl Send for BatchInferenceOutput {}

impl BatchInferenceOutput {
    /// Create from sample and extracted elements.
    pub(crate) fn new(
        batch_id: u64,
        sample: gstreamer::Sample,
        elements: Vec<ElementOutput>,
        clear_on_drop: bool,
    ) -> Self {
        Self {
            batch_id,
            elements,
            clear_on_drop,
            _sample: sample,
        }
    }

    /// Get the user-provided batch ID.
    pub fn batch_id(&self) -> u64 {
        self.batch_id
    }

    /// Get per-element outputs.
    pub fn elements(&self) -> &[ElementOutput] {
        &self.elements
    }

    /// Number of elements in the batch.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }
}

impl Drop for BatchInferenceOutput {
    fn drop(&mut self) {
        if self.clear_on_drop {
            if let Some(buffer) = self._sample.buffer() {
                // _sample is still alive here (fields drop after this fn returns,
                // in declaration order; _sample is last).
                unsafe {
                    clear_all_frame_objects(buffer.as_ptr() as *mut GstBuffer);
                }
            }
        }
    }
}

impl std::fmt::Debug for BatchInferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchInferenceOutput")
            .field("batch_id", &self.batch_id)
            .field("num_elements", &self.elements.len())
            .field("clear_on_drop", &self.clear_on_drop)
            .finish()
    }
}
