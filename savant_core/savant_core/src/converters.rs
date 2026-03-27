//! Detection model output converters.
//!
//! Provides postprocessing utilities for converting raw inference tensor
//! output into structured detection results.

pub mod nms;
pub mod yolo;

use std::fmt;

/// How NMS is applied after confidence filtering.
#[derive(Debug, Clone)]
pub enum NmsKind {
    /// No suppression — just keep the top-k detections by confidence.
    None { top_k: usize },
    /// All classes compete — a high-confidence box can suppress a nearby
    /// lower-confidence box of a different class.
    ClassAgnostic { iou_threshold: f32, top_k: usize },
    /// NMS runs independently per class — boxes of different classes
    /// never suppress each other.
    ClassAware { iou_threshold: f32, top_k: usize },
}

/// YOLO output tensor layout.
#[derive(Debug, Clone)]
pub enum YoloFormat {
    /// Single tensor `(num_classes+4) x N` — YOLOv8 / v11.
    ///
    /// Row-major layout with rows:
    /// `[xc, yc, w, h, cls_0_score, ..., cls_K_score]`.
    /// Requires logical transpose to iterate per-detection.
    V8 { num_classes: usize },
    /// Single tensor `N x (num_classes+5)` — YOLOv5 / v7.
    ///
    /// Columns: `[xc, yc, w, h, obj_conf, cls_0, ..., cls_K]`.
    /// Final confidence = `obj_conf * max(cls_scores)`.
    V5 { num_classes: usize },
    /// Two tensors — YOLOv4.
    ///
    /// Tensor 0: boxes `N x 4` (LTRB in 0..1 normalised coordinates).
    /// Tensor 1: scores `N x K`.
    V4 { model_width: f32, model_height: f32 },
    /// Three tensors — raw separate outputs.
    ///
    /// Tensor 0: boxes `N x 4` (xc, yc, w, h).
    /// Tensor 1: scores `N x K`.
    /// Tensor 2: class_ids `N` (pre-selected).
    V3Raw,
    /// Four post-NMS tensors.
    ///
    /// Tensor 0: `[1]` num_dets.
    /// Tensor 1: `N x 4` boxes (LTRB 0..1).
    /// Tensor 2: `N` confidence scores.
    /// Tensor 3: `N` class IDs.
    V4PostNms { model_width: f32, model_height: f32 },
}

/// Errors from detection converters.
#[derive(Debug, Clone)]
pub enum ConverterError {
    /// Wrong number of input tensors for the configured format.
    UnexpectedTensorCount { expected: &'static str, got: usize },
    /// Tensor shape does not match expected layout.
    ShapeMismatch {
        tensor_index: usize,
        expected: String,
        got: Vec<usize>,
    },
    /// Input tensor has zero elements.
    EmptyTensor,
    /// NMS input validation failed (length mismatch between boxes/confidences/class_ids).
    NmsInputError(String),
}

impl fmt::Display for ConverterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedTensorCount { expected, got } => {
                write!(f, "expected {expected} tensor(s), got {got}")
            }
            Self::ShapeMismatch {
                tensor_index,
                expected,
                got,
            } => {
                write!(
                    f,
                    "tensor {tensor_index}: expected shape {expected}, got {got:?}"
                )
            }
            Self::EmptyTensor => write!(f, "input tensor is empty"),
            Self::NmsInputError(msg) => write!(f, "NMS input error: {msg}"),
        }
    }
}

impl std::error::Error for ConverterError {}

pub use nms::{nms_class_agnostic, nms_class_aware};
pub use yolo::YoloDetectionConverter;
