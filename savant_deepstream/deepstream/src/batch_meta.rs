use crate::{DeepStreamError, FrameMeta, Result};
use deepstream_sys::{GstBuffer, NvDsBatchMeta};
use parking_lot::Mutex;
use std::sync::Arc;

struct BatchMetaLock(*mut NvDsBatchMeta);

unsafe impl Send for BatchMetaLock {}

impl BatchMetaLock {
    fn new(raw: *mut NvDsBatchMeta) -> Self {
        unsafe {
            deepstream_sys::nvds_acquire_meta_lock(raw);
        }
        Self(raw)
    }
}

impl Drop for BatchMetaLock {
    fn drop(&mut self) {
        unsafe {
            deepstream_sys::nvds_release_meta_lock(self.0);
        }
    }
}

/// Safe wrapper for DeepStream batch metadata
///
/// This struct provides safe access to batch metadata while managing
/// the underlying C memory properly.
pub struct BatchMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsBatchMeta,
    lock: Arc<Mutex<BatchMetaLock>>,
}

impl BatchMeta {
    /// Get batch metadata from a GstBuffer
    ///
    /// # Safety
    /// Works with raw pointers.
    ///
    /// This is the primary way to obtain batch metadata from a GStreamer buffer
    /// that contains DeepStream metadata.
    pub unsafe fn from_gst_buffer(buffer: *mut GstBuffer) -> Result<Self> {
        unsafe {
            let raw = deepstream_sys::gst_buffer_get_nvds_batch_meta(buffer);
            if raw.is_null() {
                return Err(DeepStreamError::null_pointer(
                    "No batch metadata found in buffer",
                ));
            }
            Ok(Self {
                raw,
                lock: Arc::new(Mutex::new(BatchMetaLock::new(raw))),
            })
        }
    }

    /// Get the raw pointer
    ///
    /// # Safety
    /// This returns the raw C pointer. Use with caution.
    pub fn as_raw(&self) -> *mut NvDsBatchMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsBatchMeta {
        &*self.raw
    }

    /// Get the number of frames in the batch
    pub fn num_frames(&self) -> u32 {
        unsafe { (*self.raw).num_frames_in_batch }
    }

    /// Get the maximum frames in batch
    pub fn max_frames_in_batch(&self) -> u32 {
        unsafe { (*self.raw).max_frames_in_batch }
    }

    /// Get all frame metadata
    pub fn frames(&self) -> Vec<FrameMeta> {
        let mut frames = Vec::new();
        let mut current = unsafe { (*self.raw).frame_meta_list };

        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(frame) =
                    unsafe { FrameMeta::from_raw(data as *mut deepstream_sys::NvDsFrameMeta, self) }
                {
                    frames.push(frame);
                }
            }
            current = unsafe { (*current).next };
        }

        frames
    }

    /// Find frames by source ID
    pub fn find_frames_by_source(&self, source_id: u32) -> Vec<FrameMeta> {
        self.frames()
            .into_iter()
            .filter(|frame| frame.source_id() == source_id)
            .collect()
    }

    /// Check if the batch is empty (has no frames)
    pub fn is_empty(&self) -> bool {
        self.num_frames() == 0
    }

    // Misc batch info and reserved fields

    /// Get the miscellaneous batch info array
    ///
    /// This field contains additional batch-specific information
    /// that can be used for custom processing or extensions.
    pub fn misc_batch_info(&self) -> [i64; 4] {
        unsafe { (*self.raw).misc_batch_info }
    }

    /// Set the miscellaneous batch info array
    ///
    /// This field can be used to store additional batch-specific information
    /// for custom processing or extensions.
    pub fn set_misc_batch_info(&mut self, info: [i64; 4]) {
        unsafe { (*self.raw).misc_batch_info = info }
    }

    /// Get a specific element from the misc_batch_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve (0-3)
    ///
    /// # Returns
    /// * `Some(value)` if the index is valid, `None` otherwise
    pub fn get_misc_batch_info_at(&self, index: usize) -> Option<i64> {
        if index < 4 {
            Some(unsafe { (*self.raw).misc_batch_info[index] })
        } else {
            None
        }
    }

    /// Set a specific element in the misc_batch_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to set (0-3)
    /// * `value` - The value to set
    ///
    /// # Returns
    /// * `true` if the index is valid and value was set, `false` otherwise
    pub fn set_misc_batch_info_at(&mut self, index: usize, value: i64) -> bool {
        if index < 4 {
            unsafe { (*self.raw).misc_batch_info[index] = value }
            true
        } else {
            false
        }
    }

    /// Get the reserved field array
    ///
    /// This field is reserved for future use and should not be modified
    /// unless you know exactly what you're doing.
    pub fn reserved(&self) -> [i64; 4] {
        unsafe { (*self.raw).reserved }
    }

    /// Set the reserved field array
    ///
    /// # Warning
    /// This field is reserved for future use. Only modify it if you
    /// have a specific reason and understand the implications.
    pub fn set_reserved(&mut self, reserved: [i64; 4]) {
        unsafe { (*self.raw).reserved = reserved }
    }

    /// Get a specific element from the reserved array
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve (0-3)
    ///
    /// # Returns
    /// * `Some(value)` if the index is valid, `None` otherwise
    pub fn get_reserved_at(&self, index: usize) -> Option<i64> {
        if index < 4 {
            Some(unsafe { (*self.raw).reserved[index] })
        } else {
            None
        }
    }

    /// Set a specific element in the reserved array
    ///
    /// # Arguments
    /// * `index` - The index of the element to set (0-3)
    /// * `value` - The value to set
    ///
    /// # Returns
    /// * `true` if the index is valid and value was set, `false` otherwise
    ///
    /// # Warning
    /// This field is reserved for future use. Only modify it if you
    /// have a specific reason and understand the implications.
    pub fn set_reserved_at(&mut self, index: usize, value: i64) -> bool {
        if index < 4 {
            unsafe { (*self.raw).reserved[index] = value }
            true
        } else {
            false
        }
    }

    // User metadata access

    /// Get all user metadata associated with this batch
    ///
    /// # Returns
    /// * `Vec<UserMeta>` - Vector of user metadata instances
    pub fn user_meta(&self) -> Vec<crate::UserMeta> {
        let mut user_meta_list = Vec::new();
        let mut current = unsafe { (*self.raw).batch_user_meta_list };

        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(user_meta) = unsafe {
                    crate::UserMeta::from_raw(data as *mut deepstream_sys::NvDsUserMeta, self)
                } {
                    user_meta_list.push(user_meta);
                }
            }
            current = unsafe { (*current).next };
        }

        user_meta_list
    }

    /// Check if the batch has any user metadata
    pub fn has_user_meta(&self) -> bool {
        !unsafe { (*self.raw).batch_user_meta_list.is_null() }
    }
}

impl Clone for BatchMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            lock: self.lock.clone(),
        }
    }
}

impl std::fmt::Debug for BatchMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchMeta")
            .field("num_frames", &self.num_frames())
            .field("max_frames_in_batch", &self.max_frames_in_batch())
            .field("is_empty", &self.is_empty())
            .finish()
    }
}
