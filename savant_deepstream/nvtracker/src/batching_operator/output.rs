use crate::{MiscTrackData, TrackedObject};
use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::utils::release_seal::ReleaseSeal;
use std::sync::Arc;

/// Per-frame tracking result.
///
/// The per-frame [`SharedBuffer`] is held internally by the parent
/// [`TrackerOperatorOutput`] and is only accessible after calling
/// [`TrackerOperatorOutput::take_deliveries`] and then
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
/// associated [`TrackerOperatorOutput`] is dropped.
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

    /// Block until the [`TrackerOperatorOutput`] is dropped, then return all
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
pub struct TrackerOperatorOutput {
    frames: Vec<TrackerOperatorFrameOutput>,
    deliveries: Option<Vec<(VideoFrameProxy, SharedBuffer)>>,
    seal: Arc<ReleaseSeal>,
}

unsafe impl Send for TrackerOperatorOutput {}

impl TrackerOperatorOutput {
    /// Build a new `TrackerOperatorOutput` from its constituent parts.
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

impl Drop for TrackerOperatorOutput {
    fn drop(&mut self) {
        self.deliveries.take();
        self.seal.release();
    }
}

impl std::fmt::Debug for TrackerOperatorOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackerOperatorOutput")
            .field("num_frames", &self.frames.len())
            .field("deliveries_taken", &self.deliveries.is_none())
            .finish()
    }
}
