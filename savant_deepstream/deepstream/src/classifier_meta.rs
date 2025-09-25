use crate::{DeepStreamError, Result};
use deepstream_sys::NvDsClassifierMeta;
use std::{
    ffi::{CStr, CString},
    ptr,
};

/// Safe wrapper for DeepStream classifier metadata
///
/// This struct provides safe access to classifier metadata while managing
/// the underlying C memory properly.
pub struct ClassifierMeta {
    /// Raw pointer to the C structure
    raw: *mut NvDsClassifierMeta,
    /// Whether this instance owns the memory
    owned: bool,
}

impl ClassifierMeta {
    /// Create a new classifier metadata instance
    ///
    /// This allocates new memory and should be used when creating
    /// new classifier metadata from scratch.
    pub fn new() -> Result<Self> {
        // This would typically call nvds_acquire_classifier_meta_from_pool
        // For now, we'll create a placeholder
        Err(DeepStreamError::invalid_operation(
            "Direct creation not yet implemented. Use from_raw() instead.",
        ))
    }

    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    /// This is typically used internally or when working with existing
    /// classifier metadata.
    pub unsafe fn from_raw(raw: *mut NvDsClassifierMeta) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("ClassifierMeta::from_raw"));
        }

        Ok(Self {
            raw,
            owned: false,
        })
    }

    /// Get the raw pointer
    ///
    /// # Safety
    /// This returns the raw C pointer. Use with caution.
    pub fn as_raw(&self) -> *mut NvDsClassifierMeta {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvDsClassifierMeta {
        &*self.raw
    }

    /// Get the raw pointer as a mutable reference
    ///
    /// # Safety
    /// This returns a mutable reference to the raw C structure. Use with caution.
    pub unsafe fn as_mut(&mut self) -> &mut NvDsClassifierMeta {
        &mut *self.raw
    }

    // Basic properties

    /// Get the unique component ID
    pub fn unique_component_id(&self) -> u32 {
        unsafe { (*self.raw).unique_component_id }
    }

    /// Set the unique component ID
    pub fn set_unique_component_id(&mut self, id: u32) {
        unsafe { (*self.raw).unique_component_id = id }
    }

    /// Get the number of labels
    pub fn num_labels(&self) -> u32 {
        unsafe { (*self.raw).num_labels }
    }

    /// Set the number of labels
    pub fn set_num_labels(&mut self, num: u32) {
        unsafe { (*self.raw).num_labels = num }
    }

    // Label information

    /// Get the classifier label
    pub fn label(&self) -> Result<Option<String>> {
        unsafe {
            let label_ptr = (*self.raw).classifier_label;
            if label_ptr.is_null() {
                return Ok(None);
            }

            let c_str = CStr::from_ptr(label_ptr);
            let label = c_str.to_str()?.to_string();
            Ok(Some(label))
        }
    }

    /// Set the classifier label
    pub fn set_label(&mut self, label: &str) -> Result<()> {
        let c_string = CString::new(label)?;
        unsafe {
            (*self.raw).classifier_label = c_string.into_raw();
        }
        Ok(())
    }

    // Label info list

    /// Get the number of label info entries
    pub fn num_label_info(&self) -> u32 {
        unsafe { (*self.raw).num_label_info }
    }

    /// Set the number of label info entries
    pub fn set_num_label_info(&mut self, num: u32) {
        unsafe { (*self.raw).num_label_info = num }
    }

    /// Get the label info list
    pub fn label_info_list(&self) -> *mut std::ffi::c_void {
        unsafe { (*self.raw).label_info_list }
    }

    /// Set the label info list
    pub fn set_label_info_list(&mut self, list: *mut std::ffi::c_void) {
        unsafe { (*self.raw).label_info_list = list }
    }

    // Utility methods

    /// Check if the classifier has labels
    pub fn has_labels(&self) -> bool {
        self.num_labels() > 0
    }

    /// Check if the classifier has label info
    pub fn has_label_info(&self) -> bool {
        self.num_label_info() > 0
    }

    /// Clear the classifier label
    pub fn clear_label(&mut self) {
        unsafe {
            (*self.raw).classifier_label = ptr::null();
        }
    }

    /// Clear the label info list
    pub fn clear_label_info(&mut self) {
        unsafe {
            (*self.raw).label_info_list = ptr::null_mut();
            (*self.raw).num_label_info = 0;
        }
    }

    /// Get the classifier type from meta type
    pub fn classifier_type(&self) -> u32 {
        unsafe { (*self.raw).base_meta.meta_type }
    }

    /// Set the classifier type
    pub fn set_classifier_type(&mut self, classifier_type: u32) {
        unsafe { (*self.raw).base_meta.meta_type = classifier_type }
    }

    /// Check if this is a specific type of classifier
    pub fn is_classifier_type(&self, classifier_type: u32) -> bool {
        self.classifier_type() == classifier_type
    }

    /// Get the classifier confidence (if available)
    pub fn confidence(&self) -> Option<f32> {
        // This would need to be implemented based on the specific classifier type
        // For now, return None as it's not directly available in the base structure
        None
    }

    /// Set the classifier confidence (if supported)
    pub fn set_confidence(&mut self, _confidence: f32) -> Result<()> {
        // This would need to be implemented based on the specific classifier type
        // For now, return an error
        Err(DeepStreamError::invalid_operation(
            "set_confidence not implemented for this classifier type",
        ))
    }
}

impl Drop for ClassifierMeta {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            // Release the classifier metadata back to the pool
            // This would call the appropriate release function
            log::debug!("Dropping owned ClassifierMeta");
        }
    }
}

impl Clone for ClassifierMeta {
    fn clone(&self) -> Self {
        // Create a shallow copy - the underlying memory is not duplicated
        Self {
            raw: self.raw,
            owned: false, // Clones don't own the memory
        }
    }
}

impl std::fmt::Debug for ClassifierMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassifierMeta")
            .field("unique_component_id", &self.unique_component_id())
            .field("num_labels", &self.num_labels())
            .field("num_label_info", &self.num_label_info())
            .field("label", &self.label().unwrap_or_default())
            .field("classifier_type", &self.classifier_type())
            .field("has_labels", &self.has_labels())
            .field("has_label_info", &self.has_label_info())
            .finish()
    }
}

// Helper struct for label info
#[derive(Debug, Clone)]
pub struct LabelInfo {
    /// Label text
    pub label: String,
    /// Confidence score
    pub confidence: f32,
    /// Label ID
    pub label_id: u32,
}

impl LabelInfo {
    /// Create a new label info instance
    pub fn new(label: String, confidence: f32, label_id: u32) -> Self {
        Self {
            label,
            confidence,
            label_id,
        }
    }

    /// Create from a tuple
    pub fn from_tuple((label, confidence, label_id): (String, f32, u32)) -> Self {
        Self::new(label, confidence, label_id)
    }

    /// Get the label text
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the confidence score
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get the label ID
    pub fn label_id(&self) -> u32 {
        self.label_id
    }

    /// Set the confidence score
    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence;
    }

    /// Check if the confidence is above a threshold
    pub fn confidence_above(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Check if the confidence is below a threshold
    pub fn confidence_below(&self, threshold: f32) -> bool {
        self.confidence < threshold
    }
}

impl From<(String, f32, u32)> for LabelInfo {
    fn from((label, confidence, label_id): (String, f32, u32)) -> Self {
        Self::new(label, confidence, label_id)
    }
}

impl From<(&str, f32, u32)> for LabelInfo {
    fn from((label, confidence, label_id): (&str, f32, u32)) -> Self {
        Self::new(label.to_string(), confidence, label_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_meta_creation() {
        // This test would require a proper setup
        // For now, we'll just test the error case
        let result = ClassifierMeta::new();
        assert!(result.is_err());
    }

    #[test]
    fn test_label_info_creation() {
        let label_info = LabelInfo::new("person".to_string(), 0.95, 1);
        assert_eq!(label_info.label(), "person");
        assert_eq!(label_info.confidence(), 0.95);
        assert_eq!(label_info.label_id(), 1);
    }

    #[test]
    fn test_label_info_from_tuple() {
        let label_info = LabelInfo::from_tuple(("car".to_string(), 0.87, 2));
        assert_eq!(label_info.label(), "car");
        assert_eq!(label_info.confidence(), 0.87);
        assert_eq!(label_info.label_id(), 2);
    }

    #[test]
    fn test_label_info_confidence_thresholds() {
        let mut label_info = LabelInfo::new("dog".to_string(), 0.75, 3);
        assert!(label_info.confidence_above(0.5));
        assert!(label_info.confidence_below(0.9));
        
        label_info.set_confidence(0.95);
        assert!(label_info.confidence_above(0.9));
        assert!(!label_info.confidence_below(0.9));
    }
}
