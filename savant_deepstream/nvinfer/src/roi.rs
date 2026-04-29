//! Region-of-interest types for NvInfer.

use savant_core::primitives::RBBox;

/// A region of interest to process: an identifier paired with a bounding box.
///
/// `Roi` values are passed to [`crate::pipeline::NvInfer::submit`] per batch
/// slot. DeepStream secondary mode receives them as `NvDsObjectMeta` entries
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

impl Roi {
    /// Create a new region of interest.
    ///
    /// # Arguments
    ///
    /// * `id` — Caller-defined identifier, returned in
    ///   [`ElementOutput::roi_id`](crate::output::ElementOutput::roi_id).
    /// * `bbox` — Bounding box (center-based, optionally rotated).
    pub fn new(id: i64, bbox: RBBox) -> Self {
        Self { id, bbox }
    }
}

/// Per-slot ROI specification for the [`crate::batching_operator`] layer.
///
/// Each batch slot is either inferred as a full frame (no explicit ROIs)
/// or with a caller-supplied list of regions.
#[derive(Debug, Clone)]
pub enum RoiKind {
    /// Infer on the entire frame (no per-object ROIs for this slot).
    FullFrame,
    /// Infer on specific regions within the frame.
    Rois(Vec<Roi>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_new() {
        let bbox = RBBox::new(100.0, 200.0, 50.0, 60.0, None);
        let roi = Roi::new(42, bbox.clone());
        assert_eq!(roi.id, 42);
        assert_eq!(roi.bbox.get_xc(), bbox.get_xc());
        assert_eq!(roi.bbox.get_yc(), bbox.get_yc());
        assert_eq!(roi.bbox.get_width(), bbox.get_width());
        assert_eq!(roi.bbox.get_height(), bbox.get_height());
    }
}
