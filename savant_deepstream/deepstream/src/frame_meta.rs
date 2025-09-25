use crate::{BatchMeta, DeepStreamError, ObjectMeta, Result};
use deepstream_sys::NvDsFrameMeta;

/// Safe wrapper for DeepStream frame metadata
///
/// This struct provides safe access to frame metadata while managing
/// the underlying C memory properly.
pub struct FrameMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsFrameMeta,
    /// Reference to batch meta for lifetime management
    _batch_meta: BatchMeta,
}

impl FrameMeta {
    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    /// This is typically used internally or when working with existing
    /// frame metadata.
    pub unsafe fn from_raw(raw: *mut NvDsFrameMeta, batch_meta: &BatchMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("FrameMeta::from_raw"));
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
    pub fn as_raw(&self) -> *mut NvDsFrameMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsFrameMeta {
        &*self.raw
    }

    /// Get the raw pointer as a mutable reference
    ///
    /// # Safety
    /// This returns a mutable reference to the raw C structure. Use with caution.
    pub unsafe fn as_mut(&mut self) -> &mut NvDsFrameMeta {
        &mut *self.raw
    }

    /// Get the frame number
    pub fn frame_num(&self) -> i32 {
        unsafe { (*self.raw).frame_num }
    }

    /// Get the buffer PTS (Presentation Time Stamp)
    pub fn buf_pts(&self) -> u64 {
        unsafe { (*self.raw).buf_pts }
    }

    /// Set the buffer PTS
    pub fn set_buf_pts(&mut self, pts: u64) {
        unsafe { (*self.raw).buf_pts = pts }
    }

    /// Get the source ID
    pub fn source_id(&self) -> u32 {
        unsafe { (*self.raw).source_id }
    }

    /// Get the NTP timestamp
    pub fn ntp_timestamp(&self) -> u64 {
        unsafe { (*self.raw).ntp_timestamp }
    }

    /// Get the number of surfaces per frame
    pub fn num_surfaces_per_frame(&self) -> i32 {
        unsafe { (*self.raw).num_surfaces_per_frame }
    }

    /// Get the surface type
    pub fn surface_type(&self) -> u32 {
        unsafe { (*self.raw).surface_type }
    }

    /// Get the surface index
    pub fn surface_index(&self) -> u32 {
        unsafe { (*self.raw).surface_index }
    }

    /// Get the batch ID
    pub fn batch_id(&self) -> u32 {
        unsafe { (*self.raw).batch_id }
    }

    /// Get the pad index
    pub fn pad_index(&self) -> u32 {
        unsafe { (*self.raw).pad_index }
    }

    /// Get the number of objects in the frame
    pub fn num_objects(&self) -> usize {
        // Count objects by traversing the list
        self.objects().len()
    }

    /// Get the frame width
    pub fn width(&self) -> u32 {
        unsafe { (*self.raw).source_frame_width }
    }

    /// Get the frame height
    pub fn height(&self) -> u32 {
        unsafe { (*self.raw).source_frame_height }
    }

    /// Get all object metadata
    pub fn objects(&self) -> Vec<ObjectMeta> {
        let mut objects = Vec::new();
        let mut current = unsafe { (*self.raw).obj_meta_list };

        while !current.is_null() {
            // Get the data pointer from the GList node
            let data = unsafe { (*current).data };
            if !data.is_null() {
                // Convert the data pointer to ObjectMeta
                if let Ok(obj) = unsafe {
                    ObjectMeta::from_raw(
                        data as *mut deepstream_sys::NvDsObjectMeta,
                        &self._batch_meta,
                    )
                } {
                    objects.push(obj);
                }
            }

            // Move to the next GList node
            current = unsafe { (*current).next };
        }

        objects
    }

    /// Add an object to the frame
    pub fn add_object(
        &mut self,
        obj_meta: &mut ObjectMeta,
        parent: Option<&ObjectMeta>,
    ) -> Result<()> {
        unsafe {
            let parent_ptr = parent.map(|p| p.as_raw()).unwrap_or(std::ptr::null_mut());
            deepstream_sys::nvds_add_obj_meta_to_frame(self.raw, obj_meta.as_raw(), parent_ptr);
        }
        Ok(())
    }

    /// Remove an object from the frame
    pub fn remove_object(&mut self, obj_meta: &ObjectMeta) -> Result<()> {
        unsafe {
            deepstream_sys::nvds_remove_obj_meta_from_frame(self.raw, obj_meta.as_raw());
        }
        Ok(())
    }

    /// Clear all objects from the frame
    pub fn clear_objects(&mut self) -> Result<()> {
        // Get all objects first
        let objects = self.objects();

        // Remove each object
        for obj in objects {
            self.remove_object(&obj)?;
        }

        Ok(())
    }

    /// Check if the frame is empty (has no objects)
    pub fn is_empty(&self) -> bool {
        self.num_objects() == 0
    }

    /// Get the frame area
    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }

    /// Get the center point of the frame
    pub fn center(&self) -> (f32, f32) {
        (self.width() as f32 / 2.0, self.height() as f32 / 2.0)
    }

    /// Get the aspect ratio of the frame
    pub fn aspect_ratio(&self) -> f32 {
        if self.height() > 0 {
            self.width() as f32 / self.height() as f32
        } else {
            0.0
        }
    }

    /// Check if the frame is in landscape orientation
    pub fn is_landscape(&self) -> bool {
        self.width() > self.height()
    }

    /// Check if the frame is in portrait orientation
    pub fn is_portrait(&self) -> bool {
        self.height() > self.width()
    }

    /// Check if the frame is square
    pub fn is_square(&self) -> bool {
        self.width() == self.height()
    }

    // Misc frame info and pipeline dimensions

    /// Get the miscellaneous frame info array
    ///
    /// This field contains additional frame-specific information
    /// that can be used for custom processing or extensions.
    pub fn misc_frame_info(&self) -> [i64; 4] {
        unsafe { (*self.raw).misc_frame_info }
    }

    /// Set the miscellaneous frame info array
    ///
    /// This field can be used to store additional frame-specific information
    /// for custom processing or extensions.
    pub fn set_misc_frame_info(&mut self, info: [i64; 4]) {
        unsafe { (*self.raw).misc_frame_info = info }
    }

    /// Get a specific element from the misc_frame_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve (0-3)
    ///
    /// # Returns
    /// * `Some(value)` if the index is valid, `None` otherwise
    pub fn get_misc_frame_info_at(&self, index: usize) -> Option<i64> {
        if index < 4 {
            Some(unsafe { (*self.raw).misc_frame_info[index] })
        } else {
            None
        }
    }

    /// Set a specific element in the misc_frame_info array
    ///
    /// # Arguments
    /// * `index` - The index of the element to set (0-3)
    /// * `value` - The value to set
    ///
    /// # Returns
    /// * `true` if the index is valid and value was set, `false` otherwise
    pub fn set_misc_frame_info_at(&mut self, index: usize, value: i64) -> bool {
        if index < 4 {
            unsafe { (*self.raw).misc_frame_info[index] = value }
            true
        } else {
            false
        }
    }

    /// Get the pipeline width
    ///
    /// This represents the width of the pipeline processing this frame.
    pub fn pipeline_width(&self) -> u32 {
        unsafe { (*self.raw).pipeline_width }
    }

    /// Get the pipeline height
    ///
    /// This represents the height of the pipeline processing this frame.
    pub fn pipeline_height(&self) -> u32 {
        unsafe { (*self.raw).pipeline_height }
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

    /// Get all user metadata associated with this frame
    ///
    /// # Returns
    /// * `Vec<UserMeta>` - Vector of user metadata instances
    pub fn user_meta(&self) -> Vec<crate::UserMeta> {
        let mut user_meta_list = Vec::new();
        let mut current = unsafe { (*self.raw).frame_user_meta_list };

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

    /// Check if the frame has any user metadata
    pub fn has_user_meta(&self) -> bool {
        !unsafe { (*self.raw).frame_user_meta_list.is_null() }
    }
}

impl Clone for FrameMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            _batch_meta: self._batch_meta.clone(),
        }
    }
}

impl std::fmt::Debug for FrameMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameMeta")
            .field("frame_num", &self.frame_num())
            .field("buf_pts", &self.buf_pts())
            .field("source_id", &self.source_id())
            .field("ntp_timestamp", &self.ntp_timestamp())
            .field("num_surfaces_per_frame", &self.num_surfaces_per_frame())
            .field("surface_type", &self.surface_type())
            .field("surface_index", &self.surface_index())
            .field("batch_id", &self.batch_id())
            .field("pad_index", &self.pad_index())
            .field("pipeline_width", &self.pipeline_width())
            .field("pipeline_height", &self.pipeline_height())
            .field("misc_frame_info", &self.misc_frame_info())
            .field("reserved", &self.reserved())
            .field("num_objects", &self.num_objects())
            .field("width", &self.width())
            .field("height", &self.height())
            .field("is_empty", &self.is_empty())
            .finish()
    }
}
