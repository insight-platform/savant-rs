//! Output types for batch inference results.

use crate::nvinfer_types::DataType;
use deepstream::InferDims;
use std::ffi::c_void;

/// Zero-copy view into a single output tensor.
/// Valid while the parent [`BatchInferenceOutput`] is alive.
///
/// # Safety
/// Safe to send between threads when transferred with the owning `BatchInferenceOutput`;
/// the pointers remain valid until the output is dropped.
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

/// Per-element inference output.
#[derive(Debug)]
pub struct ElementOutput {
    /// User-provided ID from SavantIdMeta (if present).
    pub id: Option<i64>,
    /// Output tensors by layer name.
    pub tensors: Vec<TensorView>,
}

/// Owns the output gst::Sample (and thus the buffer); tensor views borrow from it.
#[derive(Debug)]
pub struct BatchInferenceOutput {
    batch_id: u64,
    _sample: gstreamer::Sample,
    elements: Vec<ElementOutput>,
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
    ) -> Self {
        Self {
            batch_id,
            _sample: sample,
            elements,
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
