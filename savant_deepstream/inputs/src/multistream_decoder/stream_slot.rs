//! Per-stream state shared between the decoder engine and watchdog.

use super::frame_tracker::FrameTracker;
use crossbeam::channel::Sender;
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

/// Queued work for a feeder thread.
#[derive(Debug)]
pub(crate) enum QueueItem {
    Packet {
        frame_id: u128,
        data: Vec<u8>,
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    },
    Eos,
}

/// H.264/HEVC (and similar) waiting for first RAP.
pub(crate) struct DetectingState {
    pub pending: Vec<(VideoFrameProxy, Vec<u8>)>,
    pub codec_str: String,
    pub resolve: super::codec_resolve::CodecResolve,
    pub last_seen: Instant,
}

/// Running worker with bounded queue to feeder thread.
pub(crate) struct ActiveHandle {
    pub queue_tx: Sender<QueueItem>,
    pub alive: Arc<AtomicBool>,
    pub join: Arc<Mutex<Option<JoinHandle<()>>>>,
    pub codec_str: String,
    /// Last frame dimensions (Savant width/height) for session-boundary detection.
    pub last_width: Arc<Mutex<i64>>,
    pub last_height: Arc<Mutex<i64>>,
    /// Last `submission_order_ns` passed to `NvDecoder::submit_packet` (strictly increasing per session).
    pub last_order_key_ns: Arc<Mutex<Option<u64>>>,
    pub pending_frames: Arc<Mutex<FrameTracker>>,
    /// Shared with the feeder thread and the NvDecoder EOS callback so that
    /// the session-reset path can inject the correct [`StopReason`](super::error::StopReason).
    pub eos_kind: Arc<Mutex<Option<super::decoder::EosKind>>>,
}

pub(crate) enum StreamEntry {
    Detecting(DetectingState),
    Active(ActiveHandle),
    /// Detection buffer exceeded or fatal error; message for diagnostics.
    Failed {
        #[allow(dead_code)]
        message: String,
    },
}

/// Join feeder thread and drop queue sender.
pub(crate) fn teardown_stream_entry(entry: StreamEntry) {
    if let StreamEntry::Active(a) = entry {
        drop(a.queue_tx);
        if let Some(j) = a.join.lock().take() {
            let _ = j.join();
        }
    }
}
