use crate::error::NvTrackerError;
use crate::{MiscTrackData, TrackedObject};
use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::utils::release_seal::ReleaseSeal;
use std::sync::Arc;

/// Per-frame tracking result.
///
/// The per-frame [`SharedBuffer`] is held internally by the parent
/// [`TrackerOperatorTrackingOutput`] and is only accessible after calling
/// [`TrackerOperatorTrackingOutput::take_deliveries`] and then
/// [`SealedDeliveries::unseal`].
pub struct TrackerOperatorFrameOutput {
    /// The original [`VideoFrameProxy`] submitted for this frame.
    pub frame: VideoFrameProxy,
    /// Tracked objects for this frame.
    pub tracked_objects: Vec<TrackedObject>,
    /// Shadow tracks relevant to this frame source.
    pub shadow_tracks: Vec<MiscTrackData>,
    /// Terminated tracks relevant to this frame source.
    pub terminated_tracks: Vec<MiscTrackData>,
    /// Past-frame data relevant to this frame source.
    pub past_frame_data: Vec<MiscTrackData>,
}

/// A batch of `(VideoFrameProxy, SharedBuffer)` pairs sealed until the
/// associated [`TrackerOperatorTrackingOutput`] is dropped.
pub struct SealedDeliveries {
    deliveries: Vec<(VideoFrameProxy, SharedBuffer)>,
    seal: Arc<ReleaseSeal>,
}

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

    /// Block until the [`TrackerOperatorTrackingOutput`] is dropped, then return all
    /// deliveries as `(frame, buffer)` pairs.
    pub fn unseal(self) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        self.seal.wait();
        self.deliveries
    }

    /// Block until the seal is released, with a timeout.
    ///
    /// Returns the deliveries if the seal is released within `timeout`,
    /// or returns `Err(self)` if the timeout expires.
    pub fn unseal_timeout(
        self,
        timeout: std::time::Duration,
    ) -> Result<Vec<(VideoFrameProxy, SharedBuffer)>, Self> {
        if self.seal.wait_timeout(timeout) {
            Ok(self.deliveries)
        } else {
            Err(self)
        }
    }

    /// Non-blocking attempt to unseal.
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

/// Full batch tracking result with sealed buffer delivery.
pub struct TrackerOperatorTrackingOutput {
    frames: Vec<TrackerOperatorFrameOutput>,
    deliveries: Option<Vec<(VideoFrameProxy, SharedBuffer)>>,
    seal: Arc<ReleaseSeal>,
}

unsafe impl Send for TrackerOperatorTrackingOutput {}

impl TrackerOperatorTrackingOutput {
    /// Build a new tracking output from its constituent parts.
    pub(super) fn new(
        frames: Vec<TrackerOperatorFrameOutput>,
        deliveries: Vec<(VideoFrameProxy, SharedBuffer)>,
    ) -> Self {
        Self {
            frames,
            deliveries: Some(deliveries),
            seal: Arc::new(ReleaseSeal::new()),
        }
    }

    /// Per-frame outputs (tracking results only — no direct buffer access).
    pub fn frames(&self) -> &[TrackerOperatorFrameOutput] {
        &self.frames
    }

    /// Extract sealed deliveries while keeping the tracking output alive.
    ///
    /// Returns `Some(SealedDeliveries)` on the first call, otherwise `None`.
    pub fn take_deliveries(&mut self) -> Option<SealedDeliveries> {
        self.deliveries.take().map(|d| SealedDeliveries {
            deliveries: d,
            seal: self.seal.clone(),
        })
    }
}

impl Drop for TrackerOperatorTrackingOutput {
    fn drop(&mut self) {
        self.deliveries.take();
        self.seal.release();
    }
}

impl std::fmt::Debug for TrackerOperatorTrackingOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackerOperatorTrackingOutput")
            .field("num_frames", &self.frames.len())
            .field("deliveries_taken", &self.deliveries.is_none())
            .finish()
    }
}

/// Callback payload from [`super::NvTrackerBatchingOperator`].
#[derive(Debug)]
pub enum TrackerOperatorOutput {
    /// Completed tracking for one submitted batch.
    Tracking(TrackerOperatorTrackingOutput),
    /// Logical per-source EOS from the underlying [`crate::pipeline::NvTracker`].
    Eos { source_id: String },
    /// Pipeline or operator runtime error.
    Error(NvTrackerError),
}

impl TrackerOperatorOutput {
    /// `true` if this is a [`TrackerOperatorOutput::Tracking`] variant.
    pub fn is_tracking(&self) -> bool {
        matches!(self, Self::Tracking(_))
    }

    /// `true` if this is [`TrackerOperatorOutput::Eos`].
    pub fn is_eos(&self) -> bool {
        matches!(self, Self::Eos { .. })
    }

    /// `true` if this is [`TrackerOperatorOutput::Error`].
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Borrow the tracking payload when [`Self::is_tracking`].
    pub fn as_tracking(&self) -> Option<&TrackerOperatorTrackingOutput> {
        match self {
            Self::Tracking(t) => Some(t),
            _ => None,
        }
    }

    /// Mutably borrow the tracking payload when [`Self::is_tracking`].
    pub fn as_tracking_mut(&mut self) -> Option<&mut TrackerOperatorTrackingOutput> {
        match self {
            Self::Tracking(t) => Some(t),
            _ => None,
        }
    }
}
