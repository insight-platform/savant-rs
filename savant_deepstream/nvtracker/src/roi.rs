//! Region-of-interest for tracker input (detector boxes).

use savant_core::primitives::RBBox;

/// Identifier and bounding box for one detection fed to the tracker.
#[derive(Debug, Clone)]
pub struct Roi {
    /// Caller-defined id stamped into `NvDsObjectMeta::misc_obj_info[0]`
    /// before tracking.  The DeepStream `NvMultiObjectTracker` preserves
    /// this slot on the corresponding current-frame
    /// [`crate::TrackedObject`] (use
    /// [`crate::TrackedObject::input_roi_id`] to read it back), so
    /// callers should set `id` to whatever handle they need to pair the
    /// tracker output back to the originating detection
    /// (e.g. `VideoObject::get_id()`).
    pub id: i64,
    /// Bounding box in frame space (axis-aligned or rotated; axis-aligned envelope is used).
    pub bbox: RBBox,
}
