use crate::{BatchMeta, DeepStreamError, Result};
use deepstream_sys::NvDsObjectMeta;
use std::ffi::CString;

/// Safe wrapper for DeepStream object metadata.
///
/// Provides access to object-level metadata fields such as class ID,
/// object ID, confidence, label, and user metadata lists. The underlying
/// memory is owned by DeepStream and remains valid as long as the parent
/// [`BatchMeta`] is alive.
pub struct ObjectMeta {
    raw: *mut NvDsObjectMeta,
    _batch_meta: BatchMeta,
}

impl ObjectMeta {
    /// Acquire a new object meta from the batch pool.
    pub fn from_batch(batch_meta: &BatchMeta) -> Result<Self> {
        // SAFETY: `batch_meta.as_raw()` is guaranteed non-null by the `BatchMeta`
        // constructor invariant. The returned pointer may be null if the pool is
        // exhausted, which is checked below.
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

    /// Wrap an existing raw pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer to a `NvDsObjectMeta` that
    /// belongs to `batch_meta` and will remain valid for the lifetime of the
    /// returned value.
    pub unsafe fn from_raw(raw: *mut NvDsObjectMeta, batch_meta: &BatchMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("ObjectMeta::from_raw"));
        }
        Ok(Self {
            raw,
            _batch_meta: batch_meta.clone(),
        })
    }

    /// Raw pointer to the underlying C struct.
    pub fn as_raw(&self) -> *mut NvDsObjectMeta {
        self.raw
    }

    // ── Basic properties ──────────────────────────────────────────────

    /// Unique component ID assigned by the element that created this meta.
    pub fn unique_component_id(&self) -> i32 {
        unsafe { (*self.raw).unique_component_id }
    }

    /// Set the unique component ID.
    pub fn set_unique_component_id(&mut self, id: i32) {
        unsafe { (*self.raw).unique_component_id = id }
    }

    /// Class ID of the detected object.
    pub fn class_id(&self) -> i32 {
        unsafe { (*self.raw).class_id }
    }

    /// Set the class ID.
    pub fn set_class_id(&mut self, id: i32) {
        unsafe { (*self.raw).class_id = id }
    }

    /// Unique object ID (typically assigned by a tracker).
    pub fn object_id(&self) -> u64 {
        unsafe { (*self.raw).object_id }
    }

    /// Set the object ID.
    pub fn set_object_id(&mut self, id: u64) {
        unsafe { (*self.raw).object_id = id }
    }

    /// Detection confidence score.
    pub fn confidence(&self) -> f32 {
        unsafe { (*self.raw).confidence }
    }

    /// Set the confidence score.
    pub fn set_confidence(&mut self, confidence: f32) {
        unsafe { (*self.raw).confidence = confidence }
    }

    /// Tracker confidence score.
    pub fn tracker_confidence(&self) -> f32 {
        unsafe { (*self.raw).tracker_confidence }
    }

    /// Set the tracker confidence score.
    pub fn set_tracker_confidence(&mut self, confidence: f32) {
        unsafe { (*self.raw).tracker_confidence = confidence }
    }

    // ── Bounding boxes (rect_params, detector, tracker) ───────────────

    /// Left coordinate of [`NvDsObjectMeta::rect_params`].
    pub fn rect_left(&self) -> f32 {
        unsafe { (*self.raw).rect_params.left }
    }

    /// Top coordinate of [`NvDsObjectMeta::rect_params`].
    pub fn rect_top(&self) -> f32 {
        unsafe { (*self.raw).rect_params.top }
    }

    /// Width of [`NvDsObjectMeta::rect_params`].
    pub fn rect_width(&self) -> f32 {
        unsafe { (*self.raw).rect_params.width }
    }

    /// Height of [`NvDsObjectMeta::rect_params`].
    pub fn rect_height(&self) -> f32 {
        unsafe { (*self.raw).rect_params.height }
    }

    /// Set axis-aligned rectangle in [`NvDsObjectMeta::rect_params`].
    pub fn set_rect_params(&mut self, left: f32, top: f32, width: f32, height: f32) {
        unsafe {
            (*self.raw).rect_params.left = left;
            (*self.raw).rect_params.top = top;
            (*self.raw).rect_params.width = width;
            (*self.raw).rect_params.height = height;
        }
    }

    /// Copy `rect_params` dimensions into `detector_bbox_info.org_bbox_coords`.
    pub fn sync_detector_bbox_from_rect(&mut self) {
        unsafe {
            let r = &(*self.raw).rect_params;
            (*self.raw).detector_bbox_info.org_bbox_coords.left = r.left;
            (*self.raw).detector_bbox_info.org_bbox_coords.top = r.top;
            (*self.raw).detector_bbox_info.org_bbox_coords.width = r.width;
            (*self.raw).detector_bbox_info.org_bbox_coords.height = r.height;
        }
    }

    /// Copy `rect_params` dimensions into `tracker_bbox_info.org_bbox_coords`.
    pub fn sync_tracker_bbox_from_rect(&mut self) {
        unsafe {
            let r = &(*self.raw).rect_params;
            (*self.raw).tracker_bbox_info.org_bbox_coords.left = r.left;
            (*self.raw).tracker_bbox_info.org_bbox_coords.top = r.top;
            (*self.raw).tracker_bbox_info.org_bbox_coords.width = r.width;
            (*self.raw).tracker_bbox_info.org_bbox_coords.height = r.height;
        }
    }

    // ── Label ─────────────────────────────────────────────────────────

    /// Read the object label from the fixed-size `obj_label` buffer.
    ///
    /// Returns `Ok(None)` only when the first byte is NUL (i.e. the label
    /// is truly empty). A buffer filled with non-NUL bytes is returned as
    /// a string of the full buffer length.
    pub fn label(&self) -> Result<Option<String>> {
        unsafe {
            let label_array = &(*self.raw).obj_label;

            // Find the first NUL byte, or use the full buffer length.
            let len = label_array
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(label_array.len());

            if len == 0 {
                return Ok(None);
            }

            // Cast to u8 — c_char is i8 on x86_64, u8 on aarch64.
            let bytes = std::slice::from_raw_parts(label_array.as_ptr().cast::<u8>(), len);
            let s = std::str::from_utf8(bytes).map_err(|e| {
                DeepStreamError::ConversionError(format!("Invalid UTF-8 in obj_label: {e}"))
            })?;
            Ok(Some(s.to_owned()))
        }
    }

    /// Write a label into the fixed-size `obj_label` buffer.
    ///
    /// The label is truncated (with a NUL terminator) if it exceeds the
    /// buffer capacity.
    pub fn set_label(&mut self, label: &str) -> Result<()> {
        let c_string = CString::new(label)?;
        let bytes = c_string.as_bytes_with_nul();

        unsafe {
            let label_array = &mut (*self.raw).obj_label;
            let copy_len = bytes.len().min(label_array.len());
            // Cast to u8 — c_char is i8 on x86_64, u8 on aarch64.
            let dst = std::slice::from_raw_parts_mut(
                label_array.as_mut_ptr().cast::<u8>(),
                label_array.len(),
            );
            dst[..copy_len].copy_from_slice(&bytes[..copy_len]);
            // Ensure NUL termination when truncated.
            if copy_len == label_array.len() {
                dst[label_array.len() - 1] = 0;
            }
            // Zero-fill the remainder.
            for b in dst[copy_len..].iter_mut() {
                *b = 0;
            }
        }
        Ok(())
    }

    // ── Parent relationship ───────────────────────────────────────────

    /// Get the parent object metadata, if any.
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

    /// Set the parent object.
    ///
    /// Both objects must belong to the same batch.
    pub fn set_parent(&mut self, parent: &ObjectMeta) -> Result<()> {
        if parent._batch_meta.as_raw() == self._batch_meta.as_raw() {
            unsafe {
                (*self.raw).parent = parent.raw;
            }
            Ok(())
        } else {
            Err(DeepStreamError::invalid_parameter(
                "Parent object must be in the same batch",
            ))
        }
    }

    /// Clear the parent reference.
    pub fn clear_parent(&mut self) {
        unsafe {
            (*self.raw).parent = std::ptr::null_mut();
        }
    }

    /// Whether this object has a parent.
    pub fn has_parent(&self) -> bool {
        unsafe { !(*self.raw).parent.is_null() }
    }

    // ── Misc / reserved fields ────────────────────────────────────────

    /// Miscellaneous object info array (4 elements).
    pub fn misc_obj_info(&self) -> [i64; 4] {
        unsafe { (*self.raw).misc_obj_info }
    }

    /// Set the miscellaneous object info array.
    pub fn set_misc_obj_info(&mut self, info: [i64; 4]) {
        unsafe { (*self.raw).misc_obj_info = info }
    }

    /// Reserved field array (4 elements).
    pub fn reserved(&self) -> [i64; 4] {
        unsafe { (*self.raw).reserved }
    }

    /// Set the reserved field array.
    pub fn set_reserved(&mut self, reserved: [i64; 4]) {
        unsafe { (*self.raw).reserved = reserved }
    }

    // ── User metadata ─────────────────────────────────────────────────

    /// All user metadata attached to this object.
    pub fn user_meta(&self) -> Vec<crate::UserMeta> {
        let mut list = Vec::new();
        let mut current = unsafe { (*self.raw).obj_user_meta_list };
        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(um) = unsafe {
                    crate::UserMeta::from_raw(
                        data as *mut deepstream_sys::NvDsUserMeta,
                        &self._batch_meta,
                    )
                } {
                    list.push(um);
                }
            }
            current = unsafe { (*current).next };
        }
        list
    }

    /// Whether this object has any user metadata.
    pub fn has_user_meta(&self) -> bool {
        !unsafe { (*self.raw).obj_user_meta_list.is_null() }
    }
}

impl Clone for ObjectMeta {
    fn clone(&self) -> Self {
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
            .field("label", &self.label().unwrap_or_default())
            .field("has_parent", &self.has_parent())
            .finish()
    }
}
