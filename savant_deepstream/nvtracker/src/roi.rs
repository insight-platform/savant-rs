//! Region-of-interest for tracker input (detector boxes).

use savant_core::primitives::RBBox;

/// Identifier and bounding box for one detection fed to the tracker.
#[derive(Debug, Clone)]
pub struct Roi {
    /// Caller-defined id stored in `NvDsObjectMeta::misc_obj_info[0]`.
    pub id: i64,
    /// Bounding box in frame space (axis-aligned or rotated; axis-aligned envelope is used).
    pub bbox: RBBox,
}
