use crate::batch_meta_builder::clear_all_frame_objects;
use crate::model_input_scaling::ModelInputScaling;
use crate::output::ElementOutput;
use deepstream_buffers::SharedBuffer;
use deepstream_sys::GstBuffer;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::RBBox;
use savant_core::utils::release_seal::ReleaseSeal;
use std::cell::OnceCell;
use std::ops::Deref;
use std::sync::Arc;

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

/// Per-frame inference result (callback view — no buffer access).
///
/// The per-frame [`SharedBuffer`] is held internally by the parent
/// [`OperatorInferenceOutput`] and is only accessible after calling
/// [`OperatorInferenceOutput::take_deliveries`] and then
/// [`SealedDeliveries::unseal`].
pub struct OperatorFrameOutput {
    /// The original [`VideoFrameProxy`] submitted for this frame.
    pub frame: VideoFrameProxy,
    /// Inference results for this frame, wrapped with lazy coordinate scaling.
    pub elements: Vec<OperatorElement>,
}

// SAFETY: Same reasoning as `ElementOutput` / `BatchInferenceOutput` — raw
// pointers are valid while the owning `OperatorInferenceOutput` (and its
// `output_buffer`) is alive, so sending across threads is safe.
unsafe impl Send for OperatorFrameOutput {}

// ---------------------------------------------------------------------------
// SealedDeliveries
// ---------------------------------------------------------------------------

/// A batch of `(VideoFrameProxy, SharedBuffer)` pairs sealed until the
/// associated [`OperatorInferenceOutput`] is dropped.
///
/// The inner `Vec` is private; individual buffers are inaccessible while
/// sealed.  Call [`unseal`](Self::unseal) (blocking) or
/// [`try_unseal`](Self::try_unseal) (non-blocking) to obtain the pairs.
///
/// # Drop safety
///
/// Dropping `SealedDeliveries` without calling `unseal()` is safe — the
/// contained `SharedBuffer`s are freed and the `Condvar::notify_all` in
/// [`OperatorInferenceOutput::drop`] runs against zero waiters (a no-op).
pub struct SealedDeliveries {
    deliveries: Vec<(VideoFrameProxy, SharedBuffer)>,
    seal: Arc<ReleaseSeal>,
}

// SAFETY: All fields are `Send`.  `ReleaseSeal` uses `parking_lot::Mutex` +
// `Condvar` which are `Send + Sync`.  `VideoFrameProxy` and `SharedBuffer`
// are `Send`.
unsafe impl Send for SealedDeliveries {}

impl SealedDeliveries {
    /// Number of frames in the sealed batch.
    pub fn len(&self) -> usize {
        self.deliveries.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.deliveries.is_empty()
    }

    /// Whether the seal has been released (non-blocking check).
    pub fn is_released(&self) -> bool {
        self.seal.is_released()
    }

    /// Block until the [`OperatorInferenceOutput`] is dropped, then return
    /// all deliveries as `(frame, buffer)` pairs.
    ///
    /// After unsealing, each `SharedBuffer` should have a single strong
    /// reference, allowing `buffer.into_buffer()` for sole ownership.
    pub fn unseal(self) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        self.seal.wait();
        self.deliveries
    }

    /// Non-blocking attempt to unseal.  Returns `Err(self)` if the seal
    /// has not yet been released.
    pub fn try_unseal(self) -> Result<Vec<(VideoFrameProxy, SharedBuffer)>, Self> {
        if self.seal.is_released() {
            Ok(self.deliveries)
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Debug for SealedDeliveries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedDeliveries")
            .field("len", &self.deliveries.len())
            .field("released", &self.seal.is_released())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// OperatorInferenceOutput
// ---------------------------------------------------------------------------

/// Full batch inference result with sealed buffer delivery.
///
/// Owns the output `GstBuffer` whose memory backs the `TensorView` raw
/// pointers in `frames[].elements[].tensors`.  The output buffer is wrapped
/// in `Option` so that [`Drop`] can manually release it **before** signaling
/// the seal (see below).
///
/// # Delivery flow
///
/// 1. Callback receives `OperatorInferenceOutput` — reads tensors via
///    `frames()`, modifies `VideoFrameProxy` metadata, etc.
/// 2. Callback calls [`take_deliveries`](Self::take_deliveries) to get a
///    [`SealedDeliveries`] containing the `(frame, buffer)` pairs.
/// 3. Callback (or its scope end) drops this struct.  [`Drop`] clears tensor
///    metadata, releases `output_buffer`, then signals the seal.
/// 4. Downstream calls [`SealedDeliveries::unseal`] (or `try_unseal`) to
///    obtain the pairs — the call blocks until step 3 completes.
pub struct OperatorInferenceOutput {
    frames: Vec<OperatorFrameOutput>,
    deliveries: Option<Vec<(VideoFrameProxy, SharedBuffer)>>,
    host_copy_enabled: bool,
    seal: Arc<ReleaseSeal>,
    /// Wrapped in `Option` so `Drop` can manually release it **before**
    /// `seal.release()`.  This buffer is a parent to the per-frame buffers
    /// in `deliveries` — it must be fully gone before downstream is unblocked.
    /// `TensorView` pointers in `frames` borrow from this.
    output_buffer: Option<SharedBuffer>,
}

// SAFETY: Same pattern as `BatchInferenceOutput`.
unsafe impl Send for OperatorInferenceOutput {}

impl OperatorInferenceOutput {
    /// Build a new `OperatorInferenceOutput` from its constituent parts.
    pub(super) fn new(
        frames: Vec<OperatorFrameOutput>,
        deliveries: Vec<(VideoFrameProxy, SharedBuffer)>,
        host_copy_enabled: bool,
        output_buffer: SharedBuffer,
    ) -> Self {
        Self {
            frames,
            deliveries: Some(deliveries),
            host_copy_enabled,
            seal: Arc::new(ReleaseSeal::new()),
            output_buffer: Some(output_buffer),
        }
    }

    /// Per-frame outputs (inference results only — no buffer access).
    pub fn frames(&self) -> &[OperatorFrameOutput] {
        &self.frames
    }

    /// Whether host (CPU) tensor buffers contain valid data.
    pub fn host_copy_enabled(&self) -> bool {
        self.host_copy_enabled
    }

    /// Extract sealed deliveries while keeping the inference output alive
    /// (tensor pointers remain valid).
    ///
    /// Returns `Some(SealedDeliveries)` on the first call.  Subsequent calls
    /// return `None`.
    ///
    /// The returned [`SealedDeliveries`] blocks on [`unseal()`](SealedDeliveries::unseal)
    /// until this `OperatorInferenceOutput` is dropped.
    pub fn take_deliveries(&mut self) -> Option<SealedDeliveries> {
        self.deliveries.take().map(|d| SealedDeliveries {
            deliveries: d,
            seal: self.seal.clone(),
        })
    }
}

impl Drop for OperatorInferenceOutput {
    fn drop(&mut self) {
        // 1. Clean up tensor metadata on the output buffer.
        if let Some(ref buf) = self.output_buffer {
            let guard = buf.lock();
            unsafe {
                clear_all_frame_objects(guard.as_ref().as_ptr() as *mut GstBuffer);
            }
        }
        // 2. Release the output buffer BEFORE unblocking downstream.
        //    output_buffer is a parent to the per-frame buffers in deliveries;
        //    it must be fully dropped before downstream touches those buffers.
        self.output_buffer.take();
        // 3. Now safe to unblock — output buffer is gone.
        self.seal.release();
    }
}

impl std::fmt::Debug for OperatorInferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorInferenceOutput")
            .field("num_frames", &self.frames.len())
            .field("host_copy_enabled", &self.host_copy_enabled)
            .field("deliveries_taken", &self.deliveries.is_none())
            .finish()
    }
}
