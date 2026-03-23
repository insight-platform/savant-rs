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
    /// `true` when the host buffer contains valid data (D2H copy was performed).
    /// When `false`, only [`device_ptr`] is usable.
    pub host_copy_enabled: bool,
}

impl TensorView {
    /// Interpret host data as a typed slice (zero-copy).
    ///
    /// Returns an empty slice when host copy is disabled, the pointer is null,
    /// or the byte length is zero.
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and properly aligned for `T`.
    /// The slice length is derived from `byte_length / size_of::<T>()`.
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        if !self.host_copy_enabled || self.host_ptr.is_null() || self.byte_length == 0 {
            return &[];
        }
        let len = self.byte_length / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.host_ptr as *const T, len)
    }
}

/// Per-element inference output for one ROI in one frame.
#[derive(Debug)]
pub struct ElementOutput {
    /// ROI identifier from [`crate::roi::Roi::id`].
    ///
    /// `None` when no explicit ROIs were supplied and the full frame was used.
    pub roi_id: Option<i64>,
    /// DeepStream surface slot index (`NvDsFrameMeta.batch_id`): index into
    /// `NvBufSurface.surfaceList` for this frame.
    ///
    /// User frame ids live on the output [`BatchInferenceOutput::buffer`] via
    /// [`SharedBuffer::savant_ids`](deepstream_buffers::SharedBuffer::savant_ids)
    /// (same order as surface slots).
    pub slot_number: u32,
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
    elements: Vec<ElementOutput>,
    /// When `true`, clear all frame object metas when this value is dropped.
    clear_on_drop: bool,
    /// `true` when host (CPU) tensor buffers contain valid data.
    host_copy_enabled: bool,
    /// Ref-counted handle to the output GstBuffer (obtained via
    /// `gst_mini_object_ref`, NOT `_copy`).
    ///
    /// **Must be declared last** so that during field destruction (after
    /// `Drop::drop` returns), `elements` (which contain `TensorView`s with
    /// raw pointers into this buffer's metadata) are dropped first.
    buffer: deepstream_buffers::SharedBuffer,
}

// Safe to send: ownership transfer; pointers valid until BatchInferenceOutput is dropped.
unsafe impl Send for TensorView {}
unsafe impl Send for ElementOutput {}
unsafe impl Send for BatchInferenceOutput {}

impl BatchInferenceOutput {
    pub(crate) fn new(
        buffer: deepstream_buffers::SharedBuffer,
        elements: Vec<ElementOutput>,
        clear_on_drop: bool,
        host_copy_enabled: bool,
    ) -> Self {
        Self {
            elements,
            clear_on_drop,
            host_copy_enabled,
            buffer,
        }
    }

    /// Get per-element outputs.
    pub fn elements(&self) -> &[ElementOutput] {
        &self.elements
    }

    /// Number of elements in the batch.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Whether host (CPU) tensor buffers contain valid data.
    ///
    /// Returns `false` when `disable_output_host_copy` was set in the config,
    /// meaning only device (GPU) pointers in [`TensorView`] are usable.
    pub fn host_copy_enabled(&self) -> bool {
        self.host_copy_enabled
    }

    /// Get the output buffer as a [`SharedBuffer`](deepstream_buffers::SharedBuffer).
    ///
    /// The returned handle is a ref-counted clone pointing to the same
    /// underlying `GstBuffer`.  `SavantIdMeta` attached to the input buffer
    /// is preserved and can be read via
    /// [`SharedBuffer::savant_ids()`](deepstream_buffers::SharedBuffer::savant_ids).
    pub fn buffer(&self) -> deepstream_buffers::SharedBuffer {
        self.buffer.clone()
    }
}

impl Drop for BatchInferenceOutput {
    fn drop(&mut self) {
        if self.clear_on_drop {
            let guard = self.buffer.lock();
            unsafe {
                clear_all_frame_objects(guard.as_ref().as_ptr() as *mut GstBuffer);
            }
        }
    }
}

impl std::fmt::Debug for BatchInferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchInferenceOutput")
            .field("num_elements", &self.elements.len())
            .field("clear_on_drop", &self.clear_on_drop)
            .field("host_copy_enabled", &self.host_copy_enabled)
            .finish()
    }
}
