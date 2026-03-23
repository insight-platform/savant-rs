//! How input frames are scaled to the model's fixed input dimensions.

/// Controls nvinfer's `maintain-aspect-ratio` and `symmetric-padding` settings.
///
/// These keys are injected from [`crate::config::NvInferConfig::scaling`] during
/// [`crate::config::NvInferConfig::validate_and_materialize`] and must not appear in
/// `nvinfer_properties`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelInputScaling {
    /// Stretch to model input size (`maintain-aspect-ratio=0`).
    #[default]
    Fill,
    /// Preserve aspect ratio with padding on the right/bottom (`maintain-aspect-ratio=1`,
    /// `symmetric-padding=0`).
    KeepAspectRatio,
    /// Preserve aspect ratio with symmetric (centered) padding (`maintain-aspect-ratio=1`,
    /// `symmetric-padding=1`).
    KeepAspectRatioSymmetric,
}
