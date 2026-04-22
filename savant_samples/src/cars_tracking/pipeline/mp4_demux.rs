//! Demux actor for the `cars_tracking` sample.
//!
//! Thin wrapper around
//! [`savant_gstreamer::mp4_demuxer::Mp4Demuxer::new_parsed`] that stamps
//! the sample-level `source_id` onto the in-band messages it emits:
//!
//! 1. A single [`EncodedMsg::StreamInfo { source_id, info }`](EncodedMsg)
//!    header, emitted *once* before the first packet.  The downstream
//!    decode actor caches both fields — the `info` drives the
//!    NVDEC/RGBA pool dimensions; the `source_id` is stamped onto
//!    every [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//!    the decoder emits.
//! 2. [`EncodedMsg::Packet`] per demuxed access unit.
//! 3. Exactly one terminal [`EncodedMsg::SourceEos { source_id }`](EncodedMsg)
//!    when the underlying demuxer drains.  This is the in-band
//!    end-of-source sentinel; every downstream stage propagates its
//!    own `SourceEos` after local flush.
//!
//! Alternative ingress: producers that already own a fully-built
//! [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//! (e.g. bridges from an external message bus, or replay from stored
//! frames) can bypass the `StreamInfo` + `Packet` pair and feed the
//! decode actor directly via [`EncodedMsg::Frame`] — see the variant
//! docs for payload-resolution semantics.
//!
//! # Failure model
//!
//! Demuxer failures are handled **inside the demux actor**: a
//! `Mp4DemuxerOutput::Error` callback logs, stashes the error
//! locally, and arms the abort flag.  After `demuxer.wait()` returns
//! the demux thread body observes the stashed error and returns
//! `Err(_)` from its `JoinHandle`, which the orchestrator surfaces
//! as the pipeline exit code.  Errors are **never** propagated as
//! in-band downstream messages — downstream observes the demuxer's
//! exit the normal way (channel closure after a terminal `SourceEos`
//! is attempted or on abort).
//!
//! `source_id` is a **sample-level** concept — `Mp4Demuxer` itself has
//! no notion of it; the demux *thread* is the authoritative place
//! where it enters the pipeline.
//!
//! # Shutdown signalling
//!
//! The sample uses a single pure actor-model shutdown flow:
//!
//! * **In-band [`EncodedMsg::SourceEos`]** — normal end-of-source
//!   signal.  Propagated stage-by-stage stream-aligned via each
//!   operator's completion callback.
//! * **[`EncodedMsg::Shutdown`]** — cooperative exit sentinel,
//!   broadcast by the orchestrator onto every inter-actor channel
//!   after the supervisor back-channel observes a shutdown-worthy
//!   exit, or from the Ctrl+C handler.  Demux itself does **not**
//!   consume `Shutdown` — it has no input channel — but the variant
//!   exists on `EncodedMsg` so the decode actor can observe it.
//!
//! Fatal errors surface as `Err(_)` returned from the demux
//! thread; the orchestrator propagates the failure once it joins
//! the thread handle.

use anyhow::{anyhow, bail, Context, Result};
use crossbeam::channel::Sender;
use parking_lot::Mutex;
use savant_gstreamer::mp4_demuxer::{Mp4Demuxer, Mp4DemuxerOutput};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::cars_tracking::message::EncodedMsg;
use crate::cars_tracking::stats::PipelineStats;
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};
// StageKind is not referenced directly here — callers pass a fully
// constructed StageName so a single pipeline can host multiple
// demuxers with distinct instance tags.

/// Spawn the demux actor.
///
/// The actor reads `input` via [`Mp4Demuxer::new_parsed`] and emits
/// [`EncodedMsg`]s on `tx` until the demuxer drains or errors.  Fatal
/// errors surface as `Err(_)` from the thread handle.
///
/// * `exit_tx` is the orchestrator's supervisor back-channel.
/// * `name` identifies this demuxer on the back-channel; pass
///   [`StageName::unnamed`](crate::cars_tracking::supervisor::StageName::unnamed)
///   (e.g. `StageName::unnamed(StageKind::Mp4Demux)`) for the
///   single-demux pipeline, or
///   [`StageName::new`](crate::cars_tracking::supervisor::StageName::new)
///   with an instance tag to disambiguate concurrent demuxers.
///
/// A [`StageExitGuard`] tagged with `name` is installed at the top
/// of the thread body so the orchestrator learns about *any* exit
/// (Ok, Err, or panic unwind).
pub fn spawn_mp4_demux_thread(
    source_id: String,
    input: String,
    tx: Sender<EncodedMsg>,
    stats: Arc<PipelineStats>,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    let guard_name = name.clone();
    thread::Builder::new()
        .name("cars-mp4-demux".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(guard_name, exit_tx);
            mp4_demux_thread(source_id, input, tx, stats, name)
        })
        .context("spawn mp4_demux thread")
}

fn mp4_demux_thread(
    source_id: String,
    input: String,
    tx: Sender<EncodedMsg>,
    stats: Arc<PipelineStats>,
    name: StageName,
) -> Result<()> {
    log::info!("[{name}] starting source_id={source_id} input={input}");
    // Local abort flag — purely internal to the demux actor.  The
    // callback runs inside `Mp4Demuxer`'s GStreamer thread pool; once
    // it has observed that downstream is closed (a `send` returned
    // `Err`) there is nothing useful to do for subsequent callback
    // invocations, but the GStreamer pipeline keeps pumping until it
    // finishes the file.  This flag short-circuits the callback to a
    // quick return instead of spamming "receiver closed" warnings.
    //
    // This is **not** pipeline-wide state — the other actors have no
    // visibility into it, and nothing else in the pipeline sets it.
    let aborted = Arc::new(AtomicBool::new(false));
    // Latched error slot — on a `Mp4DemuxerOutput::Error` callback
    // we stash the error message here so the thread body can
    // surface it as `Err(_)` from its `JoinHandle` after
    // `demuxer.wait()` returns.  Errors are a component-local
    // concern; they are **not** forwarded downstream as in-band
    // messages.  First writer wins (the demuxer pipeline may emit
    // multiple `Error`s; only the first one is diagnostically
    // useful).
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let tx_cb = tx.clone();
    let aborted_cb = aborted.clone();
    let stats_cb = stats.clone();
    let first_error_cb = first_error.clone();
    let source_id_cb = source_id.clone();
    let name_cb = name.clone();
    let demuxer = Mp4Demuxer::new_parsed(&input, move |output| {
        if aborted_cb.load(Ordering::Acquire) {
            return;
        }
        match output {
            Mp4DemuxerOutput::StreamInfo(info) => {
                log::info!(
                    "[{name_cb}] stream info: source_id={source_id_cb} {}x{} @ {}/{} codec={:?}",
                    info.width,
                    info.height,
                    info.framerate_num,
                    info.framerate_den,
                    info.codec
                );
                let msg = EncodedMsg::StreamInfo {
                    source_id: source_id_cb.clone(),
                    info,
                };
                if tx_cb.send(msg).is_err() {
                    log::warn!("[{name_cb}] receiver closed; dropping stream info");
                    aborted_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Packet(pkt) => {
                stats_cb.demux_packets.fetch_add(1, Ordering::Relaxed);
                let msg = EncodedMsg::Packet {
                    source_id: source_id_cb.clone(),
                    packet: pkt,
                };
                if tx_cb.send(msg).is_err() {
                    log::warn!("[{name_cb}] receiver closed; dropping packet");
                    aborted_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Eos => {
                log::info!("[{name_cb}] EOS (source_id={source_id_cb})");
                let _ = tx_cb.send(EncodedMsg::SourceEos {
                    source_id: source_id_cb.clone(),
                });
            }
            Mp4DemuxerOutput::Error(e) => {
                // Component-local failure: log, latch the first
                // error for the thread body to surface, and arm the
                // abort flag so subsequent callbacks short-circuit.
                // Downstream is **not** notified via an in-band
                // error message — it observes the demuxer's exit
                // through normal channel closure once the thread
                // body returns `Err(_)`.
                let msg = e.to_string();
                log::error!("[{name_cb}] pipeline error: {msg}");
                let mut slot = first_error_cb.lock();
                if slot.is_none() {
                    *slot = Some(msg);
                }
                aborted_cb.store(true, Ordering::Release);
            }
        }
    })
    .map_err(|e| anyhow!("Mp4Demuxer::new_parsed: {e}"))?;

    demuxer.wait();
    let codec = demuxer.detected_codec();
    log::info!("[{name}] finished, detected_codec={codec:?}");
    // Drop `tx` last — it plus the callback-held `tx_cb` inside
    // `demuxer` (dropped when `demuxer` goes out of scope below)
    // are the only demux-side producers on the demux→decode channel.
    // Both must go away for downstream to observe `Disconnected`
    // (after consuming any in-flight `EncodedMsg::Shutdown` broadcast
    // by the orchestrator).
    drop(tx);
    drop(demuxer);
    if let Some(err) = first_error.lock().take() {
        bail!("mp4_demux: {err}");
    }
    if codec.is_none() {
        bail!("demuxer did not detect a video codec (empty stream?)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::primitives::frame::{
        VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
    };
    use savant_core::primitives::video_codec::VideoCodec;

    fn prebuilt_frame(source_id: &str, content: VideoFrameContent) -> VideoFrameProxy {
        VideoFrameProxy::new(
            source_id,
            (30, 1),
            1920,
            1080,
            content,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::H264),
            Some(true),
            (1, 1_000_000_000),
            0,
            None,
            None,
        )
        .expect("VideoFrameProxy::new")
    }

    /// The `EncodedMsg::Frame` variant round-trips a pre-built frame +
    /// explicit payload through a channel unchanged — covers the
    /// "upstream producer already owns the frame" path.
    #[test]
    fn frame_variant_round_trips_payload() {
        let frame = prebuilt_frame("cam-1", VideoFrameContent::None);
        let payload = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        let msg = EncodedMsg::Frame {
            frame: frame.clone(),
            payload: Some(payload.clone()),
        };
        match msg {
            EncodedMsg::Frame {
                frame: f,
                payload: p,
            } => {
                assert_eq!(f.get_source_id(), "cam-1");
                assert_eq!(p.as_deref(), Some(payload.as_slice()));
            }
            _ => panic!("expected Frame variant"),
        }
    }

    /// With `payload = None`, the consumer must fall back to the
    /// frame's `VideoFrameContent::Internal` bytes — this mirrors the
    /// `FlexibleDecoder::submit` contract.  We verify here only that
    /// the variant construction preserves the internal content so
    /// the decoder can extract it.
    #[test]
    fn frame_variant_preserves_internal_content_when_payload_none() {
        let blob = vec![1u8, 2, 3, 4, 5];
        let frame = prebuilt_frame("cam-1", VideoFrameContent::Internal(blob.clone()));
        let msg = EncodedMsg::Frame {
            frame,
            payload: None,
        };
        match msg {
            EncodedMsg::Frame {
                frame: f,
                payload: None,
            } => match &*f.get_content() {
                VideoFrameContent::Internal(bytes) => assert_eq!(bytes, &blob),
                _ => panic!("expected Internal content"),
            },
            _ => panic!("expected Frame with payload=None"),
        }
    }
}
