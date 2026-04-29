//! Output types for batch inference results.

use crate::nvinfer_types::DataType;
use deepstream::clear_all_frame_objects;
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
/// `MetaClearPolicy::After` or `MetaClearPolicy::Both`, dropping the
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
    /// When `false`, only `device_ptr` is usable.
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

    /// Access host tensor data as an `f32` slice.
    ///
    /// Returns an error if the data type is not [`DataType::Float`],
    /// host copy is disabled, or the host pointer is null.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use deepstream_nvinfer::output::TensorView;
    /// # fn example(tv: &TensorView) {
    /// let floats = tv.as_f32s().expect("float tensor with host data");
    /// println!("first value: {}", floats[0]);
    /// # }
    /// ```
    pub fn as_f32s(&self) -> crate::Result<&[f32]> {
        self.checked_host_slice::<f32>(DataType::Float, "float32")
    }

    /// Access host tensor data as an `i32` slice.
    ///
    /// Returns an error if the data type is not [`DataType::Int32`],
    /// host copy is disabled, or the host pointer is null.
    pub fn as_i32s(&self) -> crate::Result<&[i32]> {
        self.checked_host_slice::<i32>(DataType::Int32, "int32")
    }

    /// Access host tensor data as an `i8` slice.
    ///
    /// Returns an error if the data type is not [`DataType::Int8`],
    /// host copy is disabled, or the host pointer is null.
    pub fn as_i8s(&self) -> crate::Result<&[i8]> {
        self.checked_host_slice::<i8>(DataType::Int8, "int8")
    }

    /// Copy host tensor data into a `Vec<f32>`, transparently converting
    /// from fp16 when needed.
    ///
    /// Supports [`DataType::Float`] (zero conversion, cloned into the
    /// returned `Vec`) and [`DataType::Half`] (each `f16` element is
    /// widened to `f32`).  Returns `NvInferError::TensorTypeMismatch`
    /// for any other dtype and `NvInferError::HostDataUnavailable` when
    /// the host copy is disabled or the pointer is null.
    ///
    /// This is the canonical helper for downstream decoders such as YOLO
    /// post-processing, which otherwise re-implement the same match.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use deepstream_nvinfer::output::TensorView;
    /// # fn example(tv: &TensorView) {
    /// let floats: Vec<f32> = tv.to_f32_vec().expect("fp16/fp32 host tensor");
    /// assert!(!floats.is_empty());
    /// # }
    /// ```
    pub fn to_f32_vec(&self) -> crate::Result<Vec<f32>> {
        if !self.host_copy_enabled || self.host_ptr.is_null() || self.byte_length == 0 {
            return Err(crate::NvInferError::HostDataUnavailable);
        }
        match self.data_type {
            DataType::Float => {
                // SAFETY: dtype check above guarantees elements are f32;
                // host_copy_enabled/host_ptr/byte_length were just validated.
                let raw: &[f32] = unsafe { self.as_slice() };
                Ok(raw.to_vec())
            }
            DataType::Half => {
                // SAFETY: dtype check above guarantees elements are fp16.
                let raw: &[half::f16] = unsafe { self.as_slice() };
                Ok(raw.iter().map(|v| v.to_f32()).collect())
            }
            other => Err(crate::NvInferError::TensorTypeMismatch {
                expected: "float32 or float16",
                actual: other.name(),
            }),
        }
    }

    fn checked_host_slice<T>(
        &self,
        expected_type: DataType,
        type_name: &'static str,
    ) -> crate::Result<&[T]> {
        if !self.host_copy_enabled || self.host_ptr.is_null() || self.byte_length == 0 {
            return Err(crate::NvInferError::HostDataUnavailable);
        }
        if self.data_type != expected_type {
            return Err(crate::NvInferError::TensorTypeMismatch {
                expected: type_name,
                actual: self.data_type.name(),
            });
        }
        let len = self.byte_length / std::mem::size_of::<T>();
        // SAFETY: host_copy_enabled is true, pointer is non-null, and the data
        // type matches `T`. CUDA allocators guarantee alignment for all primitive types.
        Ok(unsafe { std::slice::from_raw_parts(self.host_ptr as *const T, len) })
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
    pub slot_number: i64,
    /// Output tensors by layer name.
    pub tensors: Vec<TensorView>,
}

/// Owns the output gst::Sample (and thus the buffer); tensor views borrow from it.
///
/// When constructed with `MetaClearPolicy::After` or
/// `MetaClearPolicy::Both`, dropping this value calls
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

    /// Consume this output, returning its constituent parts.
    ///
    /// Disables `clear_on_drop` so the caller takes ownership of cleanup
    /// responsibility.  Returns `(buffer, elements, clear_on_drop_was,
    /// host_copy_enabled)`.
    pub(crate) fn into_parts(
        mut self,
    ) -> (
        deepstream_buffers::SharedBuffer,
        Vec<ElementOutput>,
        bool,
        bool,
    ) {
        let clear = self.clear_on_drop;
        let host = self.host_copy_enabled;
        self.clear_on_drop = false;
        let elements = std::mem::take(&mut self.elements);
        let buffer = self.buffer.clone();
        (buffer, elements, clear, host)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvinfer_types::DataType;
    use std::ffi::c_void;

    fn make_tensor<T>(data: &[T], data_type: DataType, host_copy_enabled: bool) -> TensorView {
        let num = data.len() as u32;
        TensorView {
            name: "test".into(),
            dims: InferDims {
                dimensions: vec![num],
                num_elements: num,
            },
            data_type,
            host_ptr: if host_copy_enabled {
                data.as_ptr() as *const c_void
            } else {
                std::ptr::null()
            },
            device_ptr: std::ptr::null(),
            byte_length: std::mem::size_of_val(data),
            host_copy_enabled,
        }
    }

    #[test]
    fn as_f32s_succeeds_for_float_tensor() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tv = make_tensor(&data, DataType::Float, true);
        let slice = tv.as_f32s().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_i32s_succeeds_for_int32_tensor() {
        let data: Vec<i32> = vec![10, 20, 30];
        let tv = make_tensor(&data, DataType::Int32, true);
        let slice = tv.as_i32s().unwrap();
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn as_i8s_succeeds_for_int8_tensor() {
        let data: Vec<i8> = vec![-1, 0, 1, 127];
        let tv = make_tensor(&data, DataType::Int8, true);
        let slice = tv.as_i8s().unwrap();
        assert_eq!(slice, &[-1, 0, 1, 127]);
    }

    #[test]
    fn as_f32s_fails_for_wrong_type() {
        let data: Vec<i32> = vec![1, 2, 3, 4];
        let tv = make_tensor(&data, DataType::Int32, true);
        let err = tv.as_f32s().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expected float32"), "unexpected error: {msg}");
        assert!(msg.contains("got int32"), "unexpected error: {msg}");
    }

    #[test]
    fn as_f32s_fails_when_host_copy_disabled() {
        let data: Vec<f32> = vec![1.0];
        let tv = make_tensor(&data, DataType::Float, false);
        let err = tv.as_f32s().unwrap_err();
        assert!(
            err.to_string().contains("unavailable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn to_f32_vec_copies_float_tensor() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let tv = make_tensor(&data, DataType::Float, true);
        assert_eq!(tv.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_f32_vec_widens_half_tensor() {
        let data: Vec<half::f16> = vec![
            half::f16::from_f32(0.25),
            half::f16::from_f32(-1.5),
            half::f16::from_f32(4.0),
        ];
        let tv = make_tensor(&data, DataType::Half, true);
        let out = tv.to_f32_vec().unwrap();
        assert_eq!(out, vec![0.25, -1.5, 4.0]);
    }

    #[test]
    fn to_f32_vec_rejects_unsupported_dtype() {
        let data: Vec<i32> = vec![1, 2, 3];
        let tv = make_tensor(&data, DataType::Int32, true);
        let err = tv.to_f32_vec().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected float32 or float16"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("got int32"), "unexpected error: {msg}");
    }

    #[test]
    fn to_f32_vec_fails_when_host_copy_disabled() {
        let data: Vec<f32> = vec![1.0];
        let tv = make_tensor(&data, DataType::Float, false);
        assert!(matches!(
            tv.to_f32_vec().unwrap_err(),
            crate::NvInferError::HostDataUnavailable
        ));
    }

    #[test]
    fn as_f32s_fails_when_byte_length_zero() {
        let tv = TensorView {
            name: "empty".into(),
            dims: InferDims {
                dimensions: vec![0],
                num_elements: 0,
            },
            data_type: DataType::Float,
            host_ptr: 0x1 as *const c_void, // non-null but zero length
            device_ptr: std::ptr::null(),
            byte_length: 0,
            host_copy_enabled: true,
        };
        assert!(tv.as_f32s().is_err());
    }
}
