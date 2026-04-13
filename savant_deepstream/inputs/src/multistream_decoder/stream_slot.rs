//! Per-stream state shared between the decoder engine and watchdog.

use super::error::StopReason;
use super::frame_tracker::FrameTracker;
use crossbeam::channel::Sender;
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

/// Queued work for the per-stream worker thread.
#[derive(Debug)]
pub(crate) enum QueueItem {
    Packet {
        frame_id: u128,
        data: Vec<u8>,
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    },
    /// User-initiated EOS: drain remaining frames and stop.
    Eos,
    /// Internal stop (session reset, idle eviction): optionally emit
    /// [`DecoderOutput::Eos`](super::error::DecoderOutput::Eos) before
    /// [`DecoderOutput::StreamStopped`](super::error::DecoderOutput::StreamStopped).
    Stop {
        stop_reason: StopReason,
        emit_eos: bool,
    },
    /// Per-source logical EOS passthrough — sent through the GStreamer pipeline
    /// so it is ordered with decoded frames.
    ForwardEos { source_id: String },
}

/// H.264/HEVC (and similar) waiting for first RAP.
pub(crate) struct DetectingState {
    pub pending: Vec<(VideoFrameProxy, Vec<u8>)>,
    pub codec_str: String,
    pub resolve: super::codec_resolve::CodecResolve,
    pub last_seen: Instant,
}

/// Running worker with bounded queue.
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

/// Join worker thread and drop queue sender.
pub(crate) fn teardown_stream_entry(entry: StreamEntry) {
    if let StreamEntry::Active(a) = entry {
        drop(a.queue_tx);
        if let Some(j) = a.join.lock().take() {
            let _ = j.join();
        }
    }
}
