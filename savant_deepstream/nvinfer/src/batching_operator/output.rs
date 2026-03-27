use crate::batch_meta_builder::clear_all_frame_objects;
use crate::model_input_scaling::ModelInputScaling;
use crate::output::ElementOutput;
use deepstream_buffers::SharedBuffer;
use deepstream_sys::GstBuffer;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::RBBox;
use std::cell::OnceCell;
use std::ops::Deref;

use super::scaler::CoordinateScaler;

/// Per-element inference output with lazy coordinate scaling.
///
/// Wraps an [`ElementOutput`] and carries the ROI geometry and model config
/// needed to build a [`CoordinateScaler`] on first use.  Implements
/// [`Deref<Target = ElementOutput>`] so `roi_id`, `tensors`, etc. are
/// accessible directly.
pub struct OperatorElement {
    output: ElementOutput,
    roi_left: f32,
    roi_top: f32,
    roi_w: f32,
    roi_h: f32,
    model_width: f32,
    model_height: f32,
    scaling: ModelInputScaling,
    scaler: OnceCell<CoordinateScaler>,
}

// SAFETY: `OnceCell` is not `Sync` but we only need `Send`.
// The raw pointers in `ElementOutput` are valid while the parent
// `OperatorInferenceOutput` (and its output buffer) is alive.
unsafe impl Send for OperatorElement {}

impl OperatorElement {
    /// Build a new `OperatorElement` from its parts.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        output: ElementOutput,
        roi_left: f32,
        roi_top: f32,
        roi_w: f32,
        roi_h: f32,
        model_width: f32,
        model_height: f32,
        scaling: ModelInputScaling,
    ) -> Self {
        Self {
            output,
            roi_left,
            roi_top,
            roi_w,
            roi_h,
            model_width,
            model_height,
            scaling,
            scaler: OnceCell::new(),
        }
    }

    fn get_scaler(&self) -> &CoordinateScaler {
        self.scaler.get_or_init(|| {
            CoordinateScaler::new(
                self.roi_left,
                self.roi_top,
                self.roi_w,
                self.roi_h,
                self.model_width,
                self.model_height,
                self.scaling,
            )
        })
    }

    /// Return a copy of the lazily-initialized [`CoordinateScaler`].
    ///
    /// This is useful when the scaler needs to be used across a thread
    /// boundary (e.g. inside `py.detach()`) where the `&self` reference
    /// cannot be captured.
    pub fn coordinate_scaler(&self) -> CoordinateScaler {
        *self.get_scaler()
    }

    /// Transform points from model space to absolute frame coordinates.
    pub fn scale_points(&self, points: &[(f32, f32)]) -> Vec<(f32, f32)> {
        self.get_scaler().scale_points(points)
    }

    /// Transform axis-aligned boxes (left, top, width, height) from model space
    /// to absolute frame coordinates.
    pub fn scale_ltwh(&self, boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.get_scaler().scale_ltwh_batch(boxes)
    }

    /// Transform axis-aligned boxes (left, top, right, bottom) from model space
    /// to absolute frame coordinates.
    pub fn scale_ltrb(&self, boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.get_scaler().scale_ltrb_batch(boxes)
    }

    /// Transform [`RBBox`]es from model space to absolute frame coordinates.
    pub fn scale_rbboxes(&self, boxes: &[RBBox]) -> Vec<RBBox> {
        self.get_scaler().scale_rbboxes(boxes)
    }
}

impl Deref for OperatorElement {
    type Target = ElementOutput;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}

impl std::fmt::Debug for OperatorElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorElement")
            .field("roi_id", &self.output.roi_id)
            .field("slot_number", &self.output.slot_number)
            .field("num_tensors", &self.output.tensors.len())
            .field("scaler_initialized", &self.scaler.get().is_some())
            .finish()
    }
}

/// Per-frame inference result paired with the original frame and buffer.
pub struct OperatorFrameOutput {
    /// The original [`VideoFrameProxy`] submitted for this frame.
    pub frame: VideoFrameProxy,
    /// The original individual frame buffer submitted for this frame.
    pub buffer: SharedBuffer,
    /// Inference results for this frame, wrapped with lazy coordinate scaling.
    pub elements: Vec<OperatorElement>,
}

// SAFETY: Same reasoning as `ElementOutput` / `BatchInferenceOutput` — raw
// pointers are valid while the owning `OperatorInferenceOutput` (and its
// `output_buffer`) is alive, so sending across threads is safe.
unsafe impl Send for OperatorFrameOutput {}

/// Full batch inference result.
///
/// Takes over lifetime management from [`crate::output::BatchInferenceOutput`].
///
/// **Important**: `output_buffer` **must** be declared last so that Rust's
/// field drop order destroys `frames` (and their `TensorView` raw pointers)
/// before the buffer that backs them.
pub struct OperatorInferenceOutput {
    frames: Vec<OperatorFrameOutput>,
    host_copy_enabled: bool,
    /// Must be last field — `TensorView` pointers in `frames` borrow from this.
    output_buffer: SharedBuffer,
}

// SAFETY: Same pattern as `BatchInferenceOutput`.
unsafe impl Send for OperatorInferenceOutput {}

impl OperatorInferenceOutput {
    /// Build a new `OperatorInferenceOutput` from its constituent parts.
    pub(super) fn new(
        frames: Vec<OperatorFrameOutput>,
        host_copy_enabled: bool,
        output_buffer: SharedBuffer,
    ) -> Self {
        Self {
            frames,
            host_copy_enabled,
            output_buffer,
        }
    }

    /// Per-frame outputs with the original frame and buffer.
    pub fn frames(&self) -> &[OperatorFrameOutput] {
        &self.frames
    }

    /// Whether host (CPU) tensor buffers contain valid data.
    pub fn host_copy_enabled(&self) -> bool {
        self.host_copy_enabled
    }
}

impl Drop for OperatorInferenceOutput {
    fn drop(&mut self) {
        let guard = self.output_buffer.lock();
        unsafe {
            clear_all_frame_objects(guard.as_ref().as_ptr() as *mut GstBuffer);
        }
    }
}

impl std::fmt::Debug for OperatorInferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorInferenceOutput")
            .field("num_frames", &self.frames.len())
            .field("host_copy_enabled", &self.host_copy_enabled)
            .finish()
    }
}
