//! Region-of-interest type for NvInfer.

use savant_core::primitives::RBBox;

/// A region of interest to process: an identifier paired with a bounding box.
///
/// `Roi` values are passed to [`crate::pipeline::NvInfer::submit`] per batch
/// slot. DeepStream secondary mode receives them as [`NvDsObjectMeta`] entries
/// with `object_id = roi.id as u64`.
///
/// When the `RBBox` carries a non-zero angle the batch-meta builder
/// automatically computes the axis-aligned wrapping box and clamps it to
/// the frame boundaries before writing to `NvDsObjectMeta.rect_params`.
#[derive(Debug, Clone)]
pub struct Roi {
    /// Caller-defined identifier returned verbatim in [`crate::output::ElementOutput::roi_id`].
    pub id: i64,
    /// Bounding box (center-based, optionally rotated).
    pub bbox: RBBox,
}
