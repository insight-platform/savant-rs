//! Region-of-interest type for NvInfer.

use deepstream_nvbufsurface::Rect;

/// A region of interest to process: an identifier paired with a bounding box.
///
/// `Roi` values are passed to [`crate::pipeline::NvInfer::submit`] per batch
/// slot. DeepStream secondary mode receives them as [`NvDsObjectMeta`] entries
/// with `object_id = roi.id as u64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roi {
    /// Caller-defined identifier returned verbatim in [`crate::output::ElementOutput::roi_id`].
    pub id: i64,
    /// Bounding box (left, top, width, height) in pixel coordinates.
    pub rect: Rect,
}
