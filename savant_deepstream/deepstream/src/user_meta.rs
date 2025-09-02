use crate::{BatchMeta, DeepStreamError, Result};
use deepstream_sys::NvDsUserMeta;

/// Safe wrapper for DeepStream user metadata
///
/// This struct provides safe access to user metadata while managing
/// the underlying C memory properly. It implements RAII patterns to
/// ensure proper cleanup.
pub struct UserMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsUserMeta,
    _batch_meta: BatchMeta,
}

impl UserMeta {
    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    /// This is typically used internally or when working with existing
    /// user metadata.
    pub unsafe fn from_raw(raw: *mut NvDsUserMeta, batch_meta: &BatchMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("UserMeta::from_raw"));
        }

        Ok(Self {
            raw,
            _batch_meta: batch_meta.clone(),
        })
    }

    /// Get the raw pointer
    ///
    /// # Safety
    /// This returns the raw C pointer. Use with caution.
    pub fn as_raw(&self) -> *mut NvDsUserMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsUserMeta {
        &*self.raw
    }

    /// Get the raw pointer as a mutable reference
    ///
    /// # Safety
    /// This returns a mutable reference to the raw C structure. Use with caution.
    pub unsafe fn as_mut(&mut self) -> &mut NvDsUserMeta {
        &mut *self.raw
    }

    /// Get the batch metadata that this user metadata belongs to
    ///
    /// # Returns
    /// * `BatchMeta` - The batch metadata instance
    pub fn batch_meta(&self) -> BatchMeta {
        self._batch_meta.clone()
    }

    /// Get the metadata type from NvDsBaseMeta
    ///
    /// This returns the metadata type identifier that categorizes
    /// this user metadata.
    ///
    /// # Returns
    /// * `i32` - The metadata type value
    pub fn meta_type(&self) -> i32 {
        unsafe { (*self.raw).base_meta.meta_type }
    }

    /// Get the user metadata data pointer
    ///
    /// This returns the raw pointer to the user-defined data.
    /// The actual type and interpretation depends on the user.
    ///
    /// # Returns
    /// * `*mut c_void` - Raw pointer to user data
    pub fn user_meta_data(&self) -> *mut std::ffi::c_void {
        unsafe { (*self.raw).user_meta_data }
    }

    /// Get the user metadata data as a specific type
    ///
    /// # Safety
    /// The caller must ensure the type T matches the actual data type
    /// stored in the user metadata.
    ///
    /// # Type Parameters
    /// * `T` - The type to cast the user data to
    ///
    /// # Returns
    /// * `Option<&T>` - Reference to the data if not null, None otherwise
    pub unsafe fn user_meta_data_as<T>(&self) -> Option<&T> {
        let data_ptr = self.user_meta_data();
        if data_ptr.is_null() {
            None
        } else {
            Some(&*(data_ptr as *const T))
        }
    }

    /// Get the user metadata data as a mutable reference to a specific type
    ///
    /// # Safety
    /// The caller must ensure the type T matches the actual data type
    /// stored in the user metadata.
    ///
    /// # Type Parameters
    /// * `T` - The type to cast the user data to
    ///
    /// # Returns
    /// * `Option<&mut T>` - Mutable reference to the data if not null, None otherwise
    pub unsafe fn user_meta_data_as_mut<T>(&mut self) -> Option<&mut T> {
        let data_ptr = self.user_meta_data();
        if data_ptr.is_null() {
            None
        } else {
            Some(&mut *(data_ptr as *mut T))
        }
    }

    /// Check if this user metadata has user data
    pub fn has_user_data(&self) -> bool {
        !self.user_meta_data().is_null()
    }

    /// Get inference tensor metadata if this user metadata contains it
    ///
    /// This method checks if the metadata type is `NVDSINFER_TENSOR_OUTPUT_META`
    /// and if so, returns the data as `InferTensorMeta`.
    ///
    /// # Returns
    /// * `Option<InferTensorMeta>` - The inference tensor metadata if available, None otherwise
    pub fn as_infer_tensor_meta(&self) -> Option<crate::InferTensorMeta> {
        if self.meta_type() == deepstream_sys::NvDsMetaType_NVDSINFER_TENSOR_OUTPUT_META {
            let data_ptr = self.user_meta_data();
            if !data_ptr.is_null() {
                // Safety: We've verified the type and data pointer is not null
                unsafe {
                    crate::InferTensorMeta::from_raw(
                        data_ptr as *mut deepstream_sys::NvDsInferTensorMeta,
                        &self._batch_meta,
                    )
                    .ok()
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Clone for UserMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            _batch_meta: self._batch_meta.clone(),
        }
    }
}

impl std::fmt::Debug for UserMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserMeta")
            .field("meta_type", &self.meta_type())
            .field("has_user_data", &self.has_user_data())
            .field("user_meta_data_ptr", &self.user_meta_data())
            .finish()
    }
}
