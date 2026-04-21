//! Decode actor for the `cars_tracking` sample.
//!
//! Consumes [`DecoderMsg`](super::mp4_demux::DecoderMsg) from the demux actor,
//! drives [`FlexibleDecoderPool`] to decode H.264 / HEVC / AV1 / VP8 /
//! VP9 / JPEG access units into GPU surfaces, and emits [`PipelineMsg`]
//! to the infer actor.
//!
//! # Responsibilities
//!
//! * Build the [`FlexibleDecoderPool`] with this sample's tuning.
//! * Cache the `source_id` + [`VideoInfo`] from the single
//!   [`DecoderMsg::StreamInfo`](super::mp4_demux::DecoderMsg::StreamInfo)
//!   header and stamp every emitted
//!   [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//!   with them (so every downstream actor can read `source_id` off
//!   the frame without needing a constructor argument).
//! * Translate the terminal
//!   [`DecoderMsg::SourceEos`](super::mp4_demux::DecoderMsg::SourceEos) into a
//!   local `decoder.source_eos(sid)` call that *initiates* the
//!   per-source drain inside [`FlexibleDecoderPool`].  The downstream
//!   [`PipelineMsg::SourceEos`] sentinel is emitted by the decoder's
//!   callback on [`FlexibleDecoderOutput::SourceEos`], which fires
//!   strictly after the last [`FlexibleDecoderOutput::Frame`] for
//!   that source — that keeps the sentinel stream-aligned with the
//!   delivery stream (i.e. in-band on the decode->infer channel).
//!
//! Summary of EOS responsibility:
//!
//! * **Decode thread** — initiates the drain
//!   (`decoder.source_eos(sid)`).  Never touches the downstream
//!   channel for `SourceEos`.
//! * **Decoder callback** — forwards `PipelineMsg::SourceEos` on
//!   `FlexibleDecoderOutput::SourceEos`.  This is the only
//!   propagation path.
//!
//! # Shutdown
//!
//! The decode actor consumes
//! [`DecoderMsg::Shutdown`](super::mp4_demux::DecoderMsg::Shutdown) — a
//! cooperative-exit sentinel broadcast by the orchestrator after the
//! terminus (muxer / drain) has finished.  With `grace = None` the
//! loop breaks after the current message is handled; with
//! `grace = Some(d)` the actor sets a deadline and breaks the next
//! time `recv_timeout` fires at or past the deadline.

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};
use deepstream_inputs::prelude::{
    FlexibleDecoderOutput, FlexibleDecoderPool, FlexibleDecoderPoolConfig,
};
use hashbrown::HashMap;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::mp4_demux::DecoderMsg;
use crate::cars_tracking::message::PipelineMsg;
use crate::cars_tracking::stats::{stage_frames, tick_stage};
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};
// Caller supplies the fully-constructed StageName so multiple
// decoders in one pipeline can be distinguished by instance tag.
use savant_core::pipeline::stats::StageStats;

/// Decoder buffer-pool size.  Enough to hide NVDEC queue depth
/// without holding on to memory across many frames.
const DECODER_POOL_SIZE: u32 = 8;
/// Eviction TTL for [`FlexibleDecoderPool`].  Large enough that our
/// single long-running file is never evicted mid-run; a shorter
/// value is suitable for multi-source live-stream pipelines.
const DECODER_EVICTION_TTL: Duration = Duration::from_secs(3600);
/// Fallback framerate used only when the container does not
/// advertise one (per [`VideoInfo`] contract: `framerate_num == 0`).
const FALLBACK_FPS_NUM: i64 = 30;
const FALLBACK_FPS_DEN: i64 = 1;
/// Receive poll timeout.  The decode actor is multiplexed via
/// [`FlexibleDecoderPool`] (keyed by source id): per-source
/// [`DecoderMsg::SourceEos`] translates to a `decoder.source_eos(sid)`
/// flush + a downstream propagation; the receive loop exits on
/// [`DecoderMsg::Shutdown`] (grace expiry) or upstream channel closure.
const DECODE_RECV_POLL: Duration = Duration::from_millis(100);

/// Alias for the decode -> infer channel sender.  The decoder pool
/// produces the singular [`PipelineMsg::Delivery`] variant; the
/// `Deliveries` variant is never emitted on this channel.
pub type DecodedSender = Sender<PipelineMsg>;
/// Alias for the decode -> infer channel receiver.
pub type DecodedReceiver = Receiver<PipelineMsg>;

/// Spawn the decode actor.
///
/// Fatal errors (decoder submit failure, demux error, etc.) return
/// `Err(_)` from the thread handle.  The loop also breaks on
/// [`DecoderMsg::Shutdown`] broadcast by the orchestrator.  A
/// [`StageExitGuard`] tagged with `name` is installed at the top of
/// the thread body so the supervisor is notified on any exit path;
/// pass `StageName::unnamed(StageKind::Decoder)` for a single-decode
/// pipeline or a named [`StageName`] to disambiguate concurrent
/// decoders.
pub fn spawn_decoder_thread(
    gpu_id: u32,
    rx: Receiver<DecoderMsg>,
    tx: DecodedSender,
    stage: StageStats,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-decoder".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            decoder_thread(gpu_id, rx, tx, stage)
        })
        .context("spawn decode thread")
}

fn decoder_thread(
    gpu_id: u32,
    rx: Receiver<DecoderMsg>,
    tx: DecodedSender,
    stage: StageStats,
) -> Result<()> {
    log::info!("[decoder] starting");
    let dec_cfg = FlexibleDecoderPoolConfig::new(gpu_id, DECODER_POOL_SIZE, DECODER_EVICTION_TTL)
        .idle_timeout(Duration::from_secs(5))
        .detect_buffer_limit(60);

    // Local-only callback abort flag — short-circuits the callback
    // after the first send failure; not visible to other actors.
    let cb_aborted = Arc::new(AtomicBool::new(false));
    let cb_aborted_cb = cb_aborted.clone();
    let tx_cb = tx.clone();
    let stage_cb = stage.clone();
    let mut decoder = FlexibleDecoderPool::new(dec_cfg, move |mut out| {
        if cb_aborted_cb.load(Ordering::Acquire) {
            return;
        }
        match &out {
            FlexibleDecoderOutput::Frame { .. } => {
                if let Some(sealed) = out.take_delivery() {
                    tick_stage(&stage_cb, 1, 0);
                    if tx_cb.send(PipelineMsg::Delivery(sealed)).is_err() {
                        log::warn!("[decoder] downstream closed; dropping decoded frame");
                        cb_aborted_cb.store(true, Ordering::Release);
                    }
                }
            }
            FlexibleDecoderOutput::ParameterChange { old, new } => {
                log::info!("[decoder] parameter change: {old:?} -> {new:?}");
            }
            FlexibleDecoderOutput::Skipped { reason, .. } => {
                log::debug!("[decoder] skipped: {reason:?}");
            }
            FlexibleDecoderOutput::OrphanFrame { .. } => {
                log::debug!("[decoder] orphan frame (source id mismatch?)");
            }
            FlexibleDecoderOutput::SourceEos { source_id } => {
                log::info!(
                    "[decoder/cb] FlexibleDecoderOutput::SourceEos for source_id={source_id}; propagating"
                );
                if tx_cb
                    .send(PipelineMsg::SourceEos {
                        source_id: source_id.clone(),
                    })
                    .is_err()
                {
                    log::warn!("[decoder/cb] downstream closed; dropping SourceEos({source_id})");
                    cb_aborted_cb.store(true, Ordering::Release);
                }
            }
            FlexibleDecoderOutput::Event(_) => {}
            FlexibleDecoderOutput::Error(err) => {
                log::error!("[decoder] decoder error: {err}");
                cb_aborted_cb.store(true, Ordering::Release);
            }
        }
    });

    // Per-source stream metadata cache.  `FlexibleDecoderPool` is
    // multiplexed: every `source_id` it sees gets its own decoder
    // slot.  Each [`DecoderMsg::StreamInfo`] populates this map, and
    // every [`DecoderMsg::Packet`] resolves its `VideoInfo` by looking
    // up the tagged `source_id` — no "take the first entry" fallback.
    let mut stream_info: HashMap<String, VideoInfo> = HashMap::new();
    // Cooperative-exit deadline.  `None` means "keep running".  Set
    // to `Some(Instant::now() + grace)` on receipt of a
    // `Shutdown { grace: Some(d) }` message; break when
    // `recv_timeout` next fires at or past the deadline.
    let mut deadline: Option<Instant> = None;
    // `break_now` is set when a `Shutdown { grace: None }` arrives —
    // we finish handling the current message and exit.
    let mut break_now = false;

    // Termination contract:
    //
    // * `DecoderMsg::Shutdown`         → cooperative exit.  `grace =
    //                                  None` breaks after the current
    //                                  message; `grace = Some(d)` sets
    //                                  a deadline and keeps running.
    // * `recv_timeout(Timeout)`      → flush the GstPipeline's idle
    //                                  event queue, re-check the
    //                                  deadline, continue.
    // * `recv_timeout(Disconnected)` → upstream is done; break and
    //                                  let `graceful_shutdown` flush
    //                                  the decoder.
    loop {
        match rx.recv_timeout(DECODE_RECV_POLL) {
            Ok(DecoderMsg::StreamInfo {
                source_id: sid,
                info,
            }) => {
                log::info!(
                    "[decoder] stream info: source_id={sid} {}x{} @ {}/{}",
                    info.width,
                    info.height,
                    info.framerate_num,
                    info.framerate_den
                );
                if stream_info.insert(sid.clone(), info).is_some() {
                    log::warn!("[decoder] duplicate StreamInfo for {sid}; overwriting");
                }
            }
            Ok(DecoderMsg::Packet {
                source_id: sid,
                packet: pkt,
            }) => {
                let Some(info) = stream_info.get(&sid) else {
                    log::error!(
                        "[decoder] packet for source_id={sid} received before its StreamInfo — aborting"
                    );
                    return Err(anyhow!(
                        "decoder: packet for source_id={sid} received before StreamInfo"
                    ));
                };
                let frame = make_decode_frame(&sid, &pkt, info);
                if let Err(e) = decoder.submit(&frame, Some(&pkt.data)) {
                    log::error!("[decoder] submit failed (source_id={sid}): {e}");
                    return Err(anyhow!("decoder submit (source_id={sid}): {e}"));
                }
            }
            Ok(DecoderMsg::Frame { frame, payload }) => {
                // Pre-built frame path — the upstream producer
                // already populated source_id / codec / dims / fps,
                // so no `StreamInfo` lookup or
                // [`make_decode_frame`] call is needed.  Payload
                // resolution follows the
                // [`FlexibleDecoder::submit`] contract: `Some(bytes)`
                // is used verbatim; `None` makes the decoder extract
                // the bitstream from
                // [`VideoFrameContent::Internal`](savant_core::primitives::frame::VideoFrameContent)
                // (and emit a `Skipped { NoPayload }` callback if
                // the frame carries `External` / `None` content).
                let sid = frame.get_source_id();
                if let Err(e) = decoder.submit(&frame, payload.as_deref()) {
                    log::error!("[decoder] submit failed (source_id={sid}): {e}");
                    return Err(anyhow!("decoder submit (source_id={sid}): {e}"));
                }
            }
            Ok(DecoderMsg::SourceEos { source_id: sid }) => {
                // Per-source flush only — `decoder.source_eos(sid)`
                // initiates the drain, and the decoder's callback
                // will emit `PipelineMsg::SourceEos` once the last
                // frame for `sid` has been delivered.  Do NOT break
                // — other sources may still be streaming through
                // this multiplexed actor.
                log::info!("[decoder] SourceEos {sid}: initiating operator drain");
                if let Err(e) = decoder.source_eos(&sid) {
                    log::warn!("[decoder] source_eos({sid}) failed: {e}");
                }
            }
            Ok(DecoderMsg::Shutdown { grace, reason }) => {
                handle_shutdown("decoder", grace, &reason, &mut deadline, &mut break_now);
            }
            Err(RecvTimeoutError::Timeout) => {
                // Flush rescue-eligible custom events inside the
                // decoder's GStreamer pipeline when it's idle.  Same
                // rationale as in infer/pipeline.rs / tracker.rs.
                if let Err(e) = decoder.flush_idle() {
                    log::warn!("[decoder] flush_idle failed: {e}");
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[decoder] upstream channel disconnected; exiting receive loop");
                break;
            }
        }
        // Per-iteration exit checks.  The grace deadline check lives
        // here rather than in the `Timeout` branch so a chatty
        // upstream cannot keep the loop alive past the deadline by
        // sending messages faster than the poll interval.
        if break_now {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                log::info!("[decoder] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }

    if let Err(e) = decoder.graceful_shutdown() {
        log::warn!("[decoder] graceful_shutdown failed: {e}");
    }
    drop(tx);
    drop(decoder);
    log::info!("[decoder] finished ({} frames)", stage_frames(&stage));
    Ok(())
}

/// Centralised handler for the cooperative `Shutdown` variant.
/// Logging is uniform across every actor.
pub(crate) fn handle_shutdown(
    actor: &'static str,
    grace: Option<Duration>,
    reason: &str,
    deadline: &mut Option<Instant>,
    break_now: &mut bool,
) {
    match grace {
        None => {
            log::info!(
                "[{actor}] Shutdown (reason={reason}, grace=none); exiting after current message"
            );
            *break_now = true;
        }
        Some(d) => {
            let new_deadline = Instant::now() + d;
            // Honour the earlier deadline if one is already set.
            *deadline = Some(match *deadline {
                Some(existing) if existing < new_deadline => existing,
                _ => new_deadline,
            });
            log::info!(
                "[{actor}] Shutdown (reason={reason}, grace={d:?}); deadline set, continuing"
            );
        }
    }
}

/// Build the per-packet [`VideoFrameProxy`] the decoder pool
/// consumes.  Exposed for unit tests that verify the
/// [`VideoInfo::codec`] round-trip.
pub(crate) fn make_decode_frame(
    source_id: &str,
    pkt: &DemuxedPacket,
    info: &VideoInfo,
) -> VideoFrameProxy {
    let (fps_num, fps_den) = if info.framerate_num == 0 {
        (FALLBACK_FPS_NUM, FALLBACK_FPS_DEN)
    } else {
        (info.framerate_num as i64, info.framerate_den.max(1) as i64)
    };
    VideoFrameProxy::new(
        source_id,
        (fps_num, fps_den),
        info.width as i64,
        info.height as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(info.codec),
        Some(pkt.is_keyframe),
        (1, 1_000_000_000),
        pkt.pts_ns as i64,
        pkt.dts_ns.map(|v| v as i64),
        pkt.duration_ns.map(|v| v as i64),
    )
    .expect("VideoFrameProxy::new (decode)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::primitives::video_codec::VideoCodec;

    /// `make_decode_frame` must propagate `VideoInfo::codec` onto the
    /// resulting `VideoFrameProxy` — otherwise `FlexibleDecoderPool`
    /// resolves the wrong codec for non-H.264 containers.
    #[test]
    fn make_decode_frame_forwards_info_codec() {
        let info = VideoInfo {
            codec: VideoCodec::Hevc,
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
        };
        let pkt = DemuxedPacket {
            data: Vec::new(),
            pts_ns: 0,
            dts_ns: None,
            duration_ns: None,
            is_keyframe: true,
        };
        let frame = make_decode_frame("src", &pkt, &info);
        assert_eq!(frame.get_codec(), Some(VideoCodec::Hevc));
    }

    /// `handle_shutdown` records a deadline on graceful shutdown and
    /// sets `break_now` on instant shutdown.
    #[test]
    fn handle_shutdown_semantics() {
        let mut deadline: Option<Instant> = None;
        let mut break_now = false;

        handle_shutdown(
            "test",
            Some(Duration::from_secs(1)),
            "graceful",
            &mut deadline,
            &mut break_now,
        );
        assert!(deadline.is_some());
        assert!(!break_now);

        handle_shutdown("test", None, "instant", &mut deadline, &mut break_now);
        assert!(break_now);

        // Second graceful call must not push the deadline further
        // into the future — earliest wins.
        let earlier = deadline;
        handle_shutdown(
            "test",
            Some(Duration::from_secs(5)),
            "later-graceful",
            &mut deadline,
            &mut break_now,
        );
        assert_eq!(deadline, earlier, "earliest deadline must win");
    }
}
