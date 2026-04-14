//! Shared internal types for the [`super::FlexibleDecoder`] state machine.

use crate::codec_resolve::DetectionStrategy;
use deepstream_decoders::{DecoderConfig, NvDecoder};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::gstreamer_frame_time::FrameClockNs;
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::Codec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Shared map from frame UUID to the original [`VideoFrameProxy`].
///
/// Populated by [`super::FlexibleDecoder::submit`] when a packet is handed to
/// [`NvDecoder`], consumed by the worker thread / drain callback when the
/// decoded frame comes back.
pub(crate) type FrameMap = Arc<Mutex<HashMap<u128, VideoFrameProxy>>>;

/// Tuple returned from decoder activation: `(decoder, worker_join, worker_stop)`.
pub(crate) type ActivatedDecoder = (Arc<NvDecoder>, JoinHandle<()>, Arc<AtomicBool>);

/// Activation callback signature shared by all state handlers.
///
/// In production this delegates to [`super::FlexibleDecoder::activate`];
/// in tests it can be replaced with a closure that returns a canned result.
///
/// The lifetime `'a` allows the closure to borrow from the calling context
/// (e.g. capturing `&self` in [`super::FlexibleDecoder::submit`]).
pub(crate) type ActivateFn<'a> = dyn Fn(DecoderConfig, Codec, i64, i64, &VideoFrameProxy) -> Result<ActivatedDecoder, String>
    + 'a;

/// Packet buffered during H.264/HEVC stream detection.
pub(crate) struct BufferedPacket {
    pub frame: VideoFrameProxy,
    pub frame_id: u128,
    pub data: Vec<u8>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
}

/// Internal decoder lifecycle state.
pub(crate) enum DecoderState {
    Idle,
    Detecting {
        strategy: DetectionStrategy,
        video_codec: VideoCodec,
        width: i64,
        height: i64,
        buffered: Vec<BufferedPacket>,
    },
    Active {
        decoder: Arc<NvDecoder>,
        worker_join: Option<JoinHandle<()>>,
        worker_stop: Arc<AtomicBool>,
        gst_codec: Codec,
        video_codec: VideoCodec,
        width: i64,
        height: i64,
    },
    ShutDown,
}

/// Guard that ensures a [`DecoderState`] is always committed after being taken.
///
/// Created by [`StateGuard::take`], which atomically removes the current state
/// and puts a temporary placeholder.  The caller **must** call
/// [`commit`](Self::commit) with the final state before the guard is dropped.
///
/// If dropped without a commit (programming error), the drop impl forces
/// `ShutDown` in release builds and panics in debug builds so the bug is
/// immediately visible in tests.
pub(crate) struct StateGuard<'a> {
    slot: &'a mut DecoderState,
    committed: bool,
}

impl<'a> StateGuard<'a> {
    /// Take ownership of the current state, leaving `Idle` as a temporary
    /// placeholder (never observable outside the lock).
    pub(crate) fn take(slot: &'a mut DecoderState) -> (Self, DecoderState) {
        let old = std::mem::replace(slot, DecoderState::Idle);
        (
            Self {
                slot,
                committed: false,
            },
            old,
        )
    }

    /// Write the final state.  Must be called exactly once per guard.
    pub(crate) fn commit(mut self, new_state: DecoderState) {
        *self.slot = new_state;
        self.committed = true;
    }
}

impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            *self.slot = DecoderState::ShutDown;
            debug_assert!(
                false,
                "StateGuard dropped without commit — state forced to ShutDown"
            );
        }
    }
}

/// Per-frame submission data shared by all state handlers.
pub(crate) struct SubmitContext<'a> {
    pub video_codec: Option<VideoCodec>,
    pub width: i64,
    pub height: i64,
    pub frame: &'a VideoFrameProxy,
    pub payload: &'a [u8],
    pub frame_id: u128,
    pub clk: &'a FrameClockNs,
}

/// Create an empty [`FrameMap`].
pub(crate) fn new_frame_map() -> FrameMap {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Insert a [`VideoFrameProxy`] into the shared frame map.
pub(crate) fn register_frame(frame_map: &FrameMap, frame_id: u128, frame: &VideoFrameProxy) {
    frame_map.lock().insert(frame_id, frame.clone());
}

/// Stop the worker, shut down the decoder, and join the thread.
///
/// Used to clean up a newly-activated session when a subsequent operation
/// (e.g. `submit_packet`) fails before the state is committed to `Active`.
pub(crate) fn teardown_activated(
    decoder: &NvDecoder,
    worker_join: JoinHandle<()>,
    worker_stop: &AtomicBool,
) {
    worker_stop.store(true, Ordering::Relaxed);
    let _ = decoder.shutdown();
    let _ = worker_join.join();
}
