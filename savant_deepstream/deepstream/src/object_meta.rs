use crate::rect_params::RectParams;
use crate::{BatchMeta, DeepStreamError, Result};
use deepstream_sys::NvDsObjectMeta;
use std::{
    ffi::{CStr, CString},
    ptr,
};

/// Safe wrapper for DeepStream object metadata
///
/// This struct provides safe access to object metadata while managing
/// the underlying C memory properly. It implements RAII patterns to
/// ensure proper cleanup.
pub struct ObjectMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsObjectMeta,
    /// Reference to batch meta for lifetime management
    _batch_meta: BatchMeta,
}

impl ObjectMeta {
    /// Create from a batch metadata
    ///
    /// This acquires object metadata from the batch's memory pool.
    pub fn from_batch(batch_meta: &BatchMeta) -> Result<Self> {
        let raw = unsafe { deepstream_sys::nvds_acquire_obj_meta_from_pool(batch_meta.as_raw()) };

        if raw.is_null() {
            return Err(DeepStreamError::allocation_failed(
                "Failed to acquire object metadata from pool",
            ));
        }

        Ok(Self {
            raw,
            _batch_meta: batch_meta.clone(),
        })
    }

    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    /// This is typically used internally or when working with existing
    /// object metadata.
    pub unsafe fn from_raw(raw: *mut NvDsObjectMeta, batch_meta: &BatchMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("ObjectMeta::from_raw"));
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
    pub fn as_raw(&self) -> *mut NvDsObjectMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsObjectMeta {
        &*self.raw
    }

    /// Get the raw pointer as a mutable reference
    ///
    /// # Safety
    /// This returns a mutable reference to the raw C structure. Use with caution.
    pub unsafe fn as_mut(&mut self) -> &mut NvDsObjectMeta {
        &mut *self.raw
    }

    // Basic properties

    /// Get the unique component ID
    pub fn unique_component_id(&self) -> i32 {
        unsafe { (*self.raw).unique_component_id }
    }

    /// Set the unique component ID
    pub fn set_unique_component_id(&mut self, id: i32) {
        unsafe { (*self.raw).unique_component_id = id }
    }

    /// Get the class ID
    pub fn class_id(&self) -> i32 {
        unsafe { (*self.raw).class_id }
    }

    /// Set the class ID
    pub fn set_class_id(&mut self, id: i32) {
        unsafe { (*self.raw).class_id = id }
    }

    /// Get the object ID
    pub fn object_id(&self) -> u64 {
        unsafe { (*self.raw).object_id }
    }

    /// Set the object ID
    pub fn set_object_id(&mut self, id: u64) {
        unsafe { (*self.raw).object_id = id }
    }

    /// Get the confidence score
    pub fn confidence(&self) -> f32 {
        unsafe { (*self.raw).confidence }
    }

    /// Set the confidence score
    pub fn set_confidence(&mut self, confidence: f32) {
        unsafe { (*self.raw).confidence = confidence }
    }

    /// Get the tracker confidence score
    pub fn tracker_confidence(&self) -> f32 {
        unsafe { (*self.raw).tracker_confidence }
    }

    /// Set the tracker confidence score
    pub fn set_tracker_confidence(&mut self, confidence: f32) {
        unsafe { (*self.raw).tracker_confidence = confidence }
    }

    // Bounding box

    /// Get the rectangle parameters
    pub fn rect_params(&self) -> Result<RectParams> {
        unsafe { Ok(RectParams::from_ref(&(*self.raw).rect_params)) }
    }

    /// Set the rectangle parameters
    pub fn set_rect_params(&mut self, rect: RectParams) {
        unsafe {
            (*self.raw).rect_params = rect.to_raw();
        }
    }

    /// Set the bounding box
    pub fn set_bbox(&mut self, left: f32, top: f32, width: f32, height: f32) {
        let rect = RectParams::new(left, top, width, height);
        self.set_rect_params(rect);
    }

    /// Get the bounding box as a tuple
    pub fn bbox(&self) -> Result<(f32, f32, f32, f32)> {
        let rect = self.rect_params()?;
        Ok((rect.left(), rect.top(), rect.width(), rect.height()))
    }

    // Label information

    /// Get the object label
    pub fn label(&self) -> Result<Option<String>> {
        unsafe {
            let label_array = &(*self.raw).obj_label;
            // Find the first null byte to determine string length
            let mut len = 0;
            for (i, &byte) in label_array.iter().enumerate() {
                if byte == 0 {
                    len = i;
                    break;
                }
            }

            if len == 0 {
                return Ok(None);
            }

            let c_str = CStr::from_bytes_with_nul(std::mem::transmute::<&[i8], &[u8]>(
                &label_array[..=len],
            ))
            .map_err(|e| DeepStreamError::ConversionError(format!("Invalid C string: {}", e)))?;
            let label = c_str.to_str()?.to_string();
            Ok(Some(label))
        }
    }

    /// Set the object label
    pub fn set_label(&mut self, label: &str) -> Result<()> {
        let c_string = CString::new(label)?;
        let bytes = c_string.as_bytes_with_nul();

        unsafe {
            let label_array = &mut (*self.raw).obj_label;
            let copy_len = bytes.len().min(label_array.len());
            let target_slice =
                std::mem::transmute::<&mut [i8], &mut [u8]>(&mut label_array[..copy_len]);
            target_slice.copy_from_slice(&bytes[..copy_len]);

            // Fill remaining bytes with null
            for i in copy_len..label_array.len() {
                label_array[i] = 0;
            }
        }
        Ok(())
    }

    // Parent relationship

    /// Get the parent object metadata
    pub fn parent(&self) -> Option<ObjectMeta> {
        unsafe {
            let parent_ptr = (*self.raw).parent;
            if parent_ptr.is_null() {
                None
            } else {
                Some(ObjectMeta {
                    raw: parent_ptr,
                    _batch_meta: self._batch_meta.clone(),
                })
            }
        }
    }

    /// Set the parent object metadata
    ///
    /// # Arguments
    /// * `parent` - The parent object metadata
    ///
    /// # Returns
    /// * `Ok(())` if the parent was set successfully
    /// * `Err(DeepStreamError)` if the objects don't belong to the same batch
    ///
    /// # Safety
    /// This method ensures that both objects belong to the same batch
    /// to prevent invalid parent-child relationships.
    pub fn set_parent(&mut self, parent: &ObjectMeta) -> Result<()> {
        // Check if both objects belong to the same batch by comparing their batch references
        if std::ptr::eq(&parent._batch_meta.as_raw(), &self._batch_meta.as_raw()) {
            unsafe {
                (*self.raw).parent = parent.raw;
            }
            Ok(())
        } else {
            Err(DeepStreamError::invalid_parameter(
                "Parent object must be in the same batchmeta",
            ))
        }
    }

    /// Clear the parent reference
    pub fn clear_parent(&mut self) {
        unsafe {
            (*self.raw).parent = ptr::null_mut();
        }
    }

    // Detector and tracker information

    /// Get detector bounding box info
    pub fn detector_bbox_info(&self) -> Option<RectParams> {
        // For now, return None as this requires understanding the NvDsComp_BboxInfo structure
        None
    }

    /// Set detector bounding box info
    pub fn set_detector_bbox_info(&mut self, _bbox: RectParams) {
        // This would need proper memory management for the bbox_info
        // For now, we'll skip this as it requires more complex memory handling
        log::warn!("set_detector_bbox_info not yet implemented");
    }

    /// Get tracker bounding box info
    pub fn tracker_bbox_info(&self) -> Option<RectParams> {
        // For now, return None as this requires understanding the NvDsComp_BboxInfo structure
        None
    }

    /// Set tracker bounding box info
    pub fn set_tracker_bbox_info(&mut self, _bbox: RectParams) {
        // This would need proper memory management for the bbox_info
        // For now, we'll skip this as it requires more complex memory handling
        log::warn!("set_tracker_bbox_info not yet implemented");
    }

    // Utility methods

    /// Check if this object has a parent
    pub fn has_parent(&self) -> bool {
        unsafe { !(*self.raw).parent.is_null() }
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> Result<f32> {
        let rect = self.rect_params()?;
        Ok(rect.width() * rect.height())
    }

    /// Check if a point is inside the object's bounding box
    pub fn contains_point(&self, x: f32, y: f32) -> Result<bool> {
        let rect = self.rect_params()?;
        Ok(rect.contains(x, y))
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Result<(f32, f32)> {
        let rect = self.rect_params()?;
        Ok(rect.center())
    }

    // Misc object info and reserved fields

    /// Get the miscellaneous object info array
    ///
    /// This field contains additional object-specific information
    /// that can be used for custom processing or extensions.
    pub fn misc_obj_info(&self) -> [i64; 4] {
        unsafe { (*self.raw).misc_obj_info }
    }

    /// Set the miscellaneous object info array
    ///
    /// This field can be used to store additional object-specific information
    /// for custom processing or extensions.
    pub fn set_misc_obj_info(&mut self, info: [i64; 4]) {
        unsafe { (*self.raw).misc_obj_info = info }
    }

    /// Get a specific element from the misc_obj_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve (0-3)
    ///
    /// # Returns
    /// * `Some(value)` if the index is valid, `None` otherwise
    pub fn get_misc_obj_info_at(&self, index: usize) -> Option<i64> {
        if index < 4 {
            Some(unsafe { (*self.raw).misc_obj_info[index] })
        } else {
            None
        }
    }

    /// Set a specific element in the misc_obj_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to set (0-3)
    /// * `value` - The value to set
    ///
    /// # Returns
    /// * `true` if the index is valid and value was set, `false` otherwise
    pub fn set_misc_obj_info_at(&mut self, index: usize, value: i64) -> bool {
        if index < 4 {
            unsafe { (*self.raw).misc_obj_info[index] = value }
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

    /// Get all user metadata associated with this object
    ///
    /// # Returns
    /// * `Vec<UserMeta>` - Vector of user metadata instances
    pub fn user_meta(&self) -> Vec<crate::UserMeta> {
        let mut user_meta_list = Vec::new();
        let mut current = unsafe { (*self.raw).obj_user_meta_list };

        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(user_meta) = unsafe {
                    crate::UserMeta::from_raw(
                        data as *mut deepstream_sys::NvDsUserMeta,
                        &self._batch_meta,
                    )
                } {
                    user_meta_list.push(user_meta);
                }
            }
            current = unsafe { (*current).next };
        }

        user_meta_list
    }

    /// Check if the object has any user metadata
    pub fn has_user_meta(&self) -> bool {
        !unsafe { (*self.raw).obj_user_meta_list.is_null() }
    }
}

impl Clone for ObjectMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            _batch_meta: self._batch_meta.clone(),
        }
    }
}

impl std::fmt::Debug for ObjectMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectMeta")
            .field("class_id", &self.class_id())
            .field("object_id", &self.object_id())
            .field("confidence", &self.confidence())
            .field("bbox", &self.bbox().unwrap_or_default())
            .field("label", &self.label().unwrap_or_default())
            .field("has_parent", &self.has_parent())
            .finish()
    }
}
