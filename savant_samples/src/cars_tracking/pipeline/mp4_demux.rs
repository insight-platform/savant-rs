//! Demux actor for the `cars_tracking` sample.
//!
//! Thin wrapper around
//! [`savant_gstreamer::mp4_demuxer::Mp4Demuxer::new_parsed`] that stamps
//! the sample-level `source_id` onto the in-band messages it emits:
//!
//! 1. A single [`DecoderMsg::StreamInfo { source_id, info }`](DecoderMsg)
//!    header, emitted *once* before the first packet.  The downstream
//!    decode actor caches both fields — the `info` drives the
//!    NVDEC/RGBA pool dimensions; the `source_id` is stamped onto
//!    every [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//!    the decoder emits.
//! 2. [`DecoderMsg::Packet`] per demuxed access unit.
//! 3. Exactly one terminal [`DecoderMsg::SourceEos { source_id }`](DecoderMsg)
//!    when the underlying demuxer drains.  This is the in-band
//!    end-of-source sentinel; every downstream stage propagates its
//!    own `SourceEos` after local flush.
//!
//! Alternative ingress: producers that already own a fully-built
//! [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//! (e.g. bridges from an external message bus, or replay from stored
//! frames) can bypass the `StreamInfo` + `Packet` pair and feed the
//! decode actor directly via [`DecoderMsg::Frame`] — see the variant
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
//! * **In-band [`DecoderMsg::SourceEos`]** — normal end-of-source
//!   signal.  Propagated stage-by-stage stream-aligned via each
//!   operator's completion callback.
//! * **[`DecoderMsg::Shutdown`]** — cooperative exit sentinel,
//!   broadcast by the orchestrator onto every inter-actor channel
//!   after the terminus (muxer / drain) has been joined, or from
//!   the Ctrl+C handler.  Demux itself does **not** consume
//!   `Shutdown` — it has no input channel — but the variant exists
//!   on `DecoderMsg` so the decode actor can observe it.
//!
//! Fatal errors surface as `Err(_)` returned from the demux
//! thread; the orchestrator propagates the failure once it joins
//! the thread handle.

use anyhow::{anyhow, bail, Context, Result};
use crossbeam::channel::{Receiver, Sender};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, Mp4Demuxer, Mp4DemuxerOutput, VideoInfo};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::cars_tracking::stats::PipelineStats;
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};
// StageKind is not referenced directly here — callers pass a fully
// constructed StageName so a single pipeline can host multiple
// demuxers with distinct instance tags.

/// Demux -> decode boundary message type.
///
/// Source EOS is an **in-band** message (not channel closure) — every
/// inter-stage channel in the sample follows the same rule: upstream
/// communicates to downstream by sending both main deliverables *and*
/// an explicit `SourceEos { source_id }` sentinel.  The same channel
/// also carries the out-of-band [`DecoderMsg::Shutdown`] sentinel,
/// which the orchestrator broadcasts once the terminus actor has
/// joined (or when Ctrl+C fires).
#[derive(Debug)]
pub enum DecoderMsg {
    /// Stream metadata discovered from the container caps paired with
    /// the sample-level source id.  Fires once, before any
    /// [`DecoderMsg::Packet`].
    StreamInfo {
        /// Source id this stream belongs to.  Every decoded frame the
        /// decoder emits is stamped with this value.
        source_id: String,
        /// Stream-level metadata (width, height, framerate, codec).
        info: VideoInfo,
    },
    /// A single demuxed access unit ready for the decoder, tagged
    /// with the `source_id` it belongs to.  Tagging every packet
    /// (rather than relying on a preceding [`DecoderMsg::StreamInfo`])
    /// keeps the channel protocol self-describing: a multiplexed
    /// decode actor can look up the matching [`VideoInfo`] for each
    /// packet without assuming a single-source pipeline.
    Packet {
        /// Source id this packet belongs to.
        source_id: String,
        /// The demuxed access unit.
        packet: DemuxedPacket,
    },
    /// A pre-built [`VideoFrameProxy`] delivered with its encoded
    /// payload, ready for direct submission to the decoder.
    ///
    /// Unlike [`DecoderMsg::Packet`] — which is raw container output
    /// that the decode actor wraps into a fresh frame via
    /// `make_decode_frame` — this variant lets an upstream producer
    /// hand off a fully-populated frame (source id, dimensions,
    /// codec, fps, uuid, keyframe, timestamps) and side-step the
    /// `StreamInfo` / `make_decode_frame` path entirely.
    ///
    /// Payload resolution follows the `FlexibleDecoder::submit`
    /// contract:
    ///
    /// * `payload = Some(bytes)` — the decoder uses `bytes` directly.
    /// * `payload = None` — the decoder extracts the bitstream from
    ///   the frame's internal content
    ///   ([`VideoFrameContent::Internal`](savant_core::primitives::frame::VideoFrameContent)).
    ///   If the frame's content is `External` or `None`, the decoder
    ///   emits a `Skipped { NoPayload }` callback rather than
    ///   erroring.
    ///
    /// This is the entrypoint for producers that already own the
    /// frame (e.g. reading previously-captured `savant_core` frames
    /// back from storage, or bridging from another message-bus
    /// format) without paying the cost of re-deriving frame
    /// metadata from demuxer caps.
    Frame {
        /// Pre-built frame.  Its `source_id`, `codec`, `width`,
        /// `height`, `fps`, `uuid`, and `keyframe` fields must
        /// already be set — the decoder routes on
        /// `frame.get_source_id()` and consults the other fields at
        /// `submit` time.
        frame: VideoFrameProxy,
        /// Encoded bitstream for this frame, or `None` to have the
        /// decoder extract it from `frame.get_content()`.
        payload: Option<Vec<u8>>,
    },
    /// End-of-stream sentinel carrying the source id.  The decoder
    /// runs its own `decoder.source_eos(source_id)` + drain on
    /// receipt, then forwards an equivalent sentinel on its
    /// downstream channel.
    SourceEos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
    /// Cooperative shutdown sentinel, broadcast by the orchestrator
    /// after the terminus actor has joined (or from the Ctrl+C
    /// handler).  Decode consumes this variant and exits its
    /// receive loop either immediately (when `grace` is `None`) or
    /// after the grace deadline elapses.
    Shutdown {
        /// `None` — break after the current message is handled.
        /// `Some(d)` — keep processing, break when `recv_timeout`
        /// next fires at or past `now + d`.
        grace: Option<Duration>,
        /// Human-readable reason (logged on receipt).
        reason: Cow<'static, str>,
    },
}

/// Alias for the demux -> decode channel sender.
pub type DecoderSender = Sender<DecoderMsg>;
/// Alias for the demux -> decode channel receiver.
pub type DecoderReceiver = Receiver<DecoderMsg>;

/// Spawn the demux actor.
///
/// The actor reads `input` via [`Mp4Demuxer::new_parsed`] and emits
/// [`DecoderMsg`]s on `tx` until the demuxer drains or errors.  Fatal
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
    tx: DecoderSender,
    stats: Arc<PipelineStats>,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-mp4-demux".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            mp4_demux_thread(source_id, input, tx, stats)
        })
        .context("spawn mp4_demux thread")
}

fn mp4_demux_thread(
    source_id: String,
    input: String,
    tx: DecoderSender,
    stats: Arc<PipelineStats>,
) -> Result<()> {
    log::info!("[mp4_demux] starting source_id={source_id} input={input}");
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
    let demuxer = Mp4Demuxer::new_parsed(&input, move |output| {
        if aborted_cb.load(Ordering::Acquire) {
            return;
        }
        match output {
            Mp4DemuxerOutput::StreamInfo(info) => {
                log::info!(
                    "[mp4_demux] stream info: source_id={source_id_cb} {}x{} @ {}/{} codec={:?}",
                    info.width,
                    info.height,
                    info.framerate_num,
                    info.framerate_den,
                    info.codec
                );
                let msg = DecoderMsg::StreamInfo {
                    source_id: source_id_cb.clone(),
                    info,
                };
                if tx_cb.send(msg).is_err() {
                    log::warn!("[mp4_demux] receiver closed; dropping stream info");
                    aborted_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Packet(pkt) => {
                stats_cb.demux_packets.fetch_add(1, Ordering::Relaxed);
                let msg = DecoderMsg::Packet {
                    source_id: source_id_cb.clone(),
                    packet: pkt,
                };
                if tx_cb.send(msg).is_err() {
                    log::warn!("[mp4_demux] receiver closed; dropping packet");
                    aborted_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Eos => {
                log::info!("[mp4_demux] EOS (source_id={source_id_cb})");
                let _ = tx_cb.send(DecoderMsg::SourceEos {
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
                log::error!("[mp4_demux] pipeline error: {msg}");
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
    log::info!("[mp4_demux] finished, detected_codec={codec:?}");
    // Drop `tx` last — it plus the callback-held `tx_cb` inside
    // `demuxer` (dropped when `demuxer` goes out of scope below)
    // are the only demux-side producers on the demux→decode channel.
    // Both must go away for downstream to observe `Disconnected`
    // (after consuming any in-flight `DecoderMsg::Shutdown` broadcast
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

    /// The `DecoderMsg::Frame` variant round-trips a pre-built frame +
    /// explicit payload through a channel unchanged — covers the
    /// "upstream producer already owns the frame" path.
    #[test]
    fn frame_variant_round_trips_payload() {
        let frame = prebuilt_frame("cam-1", VideoFrameContent::None);
        let payload = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        let msg = DecoderMsg::Frame {
            frame: frame.clone(),
            payload: Some(payload.clone()),
        };
        match msg {
            DecoderMsg::Frame {
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
        let msg = DecoderMsg::Frame {
            frame,
            payload: None,
        };
        match msg {
            DecoderMsg::Frame {
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
