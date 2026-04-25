//! Active-state handler for [`super::FlexibleDecoder`].

use crate::codec_resolve::CodecResolve;
use deepstream_decoders::NvDecoder;
use savant_core::primitives::video_codec::VideoCodec;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use super::error::FlexibleDecoderError;
use super::output::{DecoderParameters, FlexibleDecoderOutput, SkipReason};
use super::state::{register_frame, DecoderState, FrameMap, StateGuard, SubmitContext};

/// Result of processing a submit in the `Active` state.
pub(crate) enum ActiveResult {
    /// Steady-state submit completed (same codec/resolution).
    SteadyState(Result<(), FlexibleDecoderError>),
    /// Parameter change detected.  Caller must drain `old_decoder` outside
    /// the lock, then start a new session via [`super::handle_idle::handle_idle`].
    NeedDrain {
        old_decoder: Arc<NvDecoder>,
        old_params: DecoderParameters,
        new_params: DecoderParameters,
    },
    /// The underlying [`NvDecoder`] worker died (e.g. the GstPipeline
    /// watchdog tripped while frames were stuck in the `in_flight` map).
    /// The state has been committed to `Idle`, the dead decoder has been
    /// torn down, and `pending` has been populated with one
    /// [`FlexibleDecoderOutput::Skipped`] per orphaned frame plus a
    /// trailing aggregate
    /// [`FlexibleDecoderOutput::Restarted`].  The caller must
    /// re-acquire the state lock and re-run [`super::handle_idle::handle_idle`]
    /// for `ctx`, exactly like the [`NeedDrain`](Self::NeedDrain) recovery
    /// path — the restart is therefore transparent to
    /// [`super::FlexibleDecoder::submit`].
    Restarted,
}

/// Reason string surfaced via [`FlexibleDecoderOutput::Restarted`] when the
/// worker thread exits unexpectedly.  The upstream
/// [`PipelineError`](savant_gstreamer::pipeline::PipelineError) is reported
/// separately via [`FlexibleDecoderOutput::Error`].
const WORKER_DIED_REASON: &str = "worker thread exited";

/// Handle a submit when the decoder is in `Active` state.
///
/// On steady-state (same codec/resolution): submits the packet, commits
/// guard to `Active`, returns `SteadyState(Ok(...))`.
///
/// On parameter change: stops the worker, commits guard to `Idle`, returns
/// `NeedDrain { ... }` so the caller can drain outside the lock.
///
/// On submit error: commits guard to `Active` (decoder still valid), returns
/// `SteadyState(Err(...))`.
///
/// On dead worker: tears down the decoder, drains the entire frame map
/// into `pending` as
/// [`Skipped { reason: DecoderRestarted(_) }`](FlexibleDecoderOutput::Skipped),
/// pushes a [`FlexibleDecoderOutput::Restarted`] aggregate event, commits
/// guard to `Idle`, and returns [`ActiveResult::Restarted`].  The caller is
/// expected to re-run [`super::handle_idle::handle_idle`] for `ctx` so the
/// restart is transparent to [`super::FlexibleDecoder::submit`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_active(
    guard: StateGuard<'_>,
    pending: &mut Vec<FlexibleDecoderOutput>,
    frame_map: &FrameMap,
    source_id: &str,
    decoder: Arc<NvDecoder>,
    worker_join: Option<JoinHandle<()>>,
    worker_stop: Arc<AtomicBool>,
    active_gst_codec: VideoCodec,
    active_video_codec: VideoCodec,
    active_width: i64,
    active_height: i64,
    resolve: &CodecResolve,
    ctx: &SubmitContext<'_>,
) -> ActiveResult {
    let worker_died = !worker_stop.load(Ordering::Relaxed)
        && worker_join.as_ref().is_some_and(|jh| jh.is_finished());

    if worker_died {
        if let Some(jh) = worker_join {
            let _ = jh.join();
        }
        let _ = decoder.shutdown();

        // Drain any frames still mapped to the dead decoder so users see
        // a per-frame Skipped(DecoderRestarted) plus the aggregate
        // Restarted event below.
        let lost_frames = {
            let mut fm = frame_map.lock();
            let mut lost = 0usize;
            for (_, frame) in fm.drain() {
                lost += 1;
                pending.push(FlexibleDecoderOutput::Skipped {
                    frame,
                    data: None,
                    reason: SkipReason::DecoderRestarted(WORKER_DIED_REASON.to_string()),
                });
            }
            lost
        };

        pending.push(FlexibleDecoderOutput::Restarted {
            source_id: source_id.to_string(),
            reason: WORKER_DIED_REASON.to_string(),
            lost_frames,
        });

        guard.commit(DecoderState::Idle);
        return ActiveResult::Restarted;
    }

    let codec_changed = ctx.video_codec != Some(active_video_codec);
    let dims_changed = ctx.width != active_width || ctx.height != active_height;

    if codec_changed || dims_changed {
        worker_stop.store(true, Ordering::Relaxed);
        if let Some(jh) = worker_join {
            let _ = jh.join();
        }

        let old_params = DecoderParameters {
            codec: active_gst_codec,
            width: active_width,
            height: active_height,
        };
        let new_gst_codec = match resolve {
            CodecResolve::Ready(cfg) => cfg.codec(),
            CodecResolve::NeedDetection(strategy) => strategy.codec(),
        };
        let new_params = DecoderParameters {
            codec: new_gst_codec,
            width: ctx.width,
            height: ctx.height,
        };

        // Intentional Idle: phase 2 re-acquires the lock and starts a new
        // session via handle_idle.
        guard.commit(DecoderState::Idle);
        ActiveResult::NeedDrain {
            old_decoder: decoder,
            old_params,
            new_params,
        }
    } else {
        register_frame(frame_map, ctx.frame_id, ctx.frame);
        match decoder.submit_packet(
            ctx.payload,
            ctx.frame_id,
            ctx.clk.submission_order_ns,
            ctx.clk.dts_ns,
            ctx.clk.duration_ns,
        ) {
            Ok(()) => {
                guard.commit(DecoderState::Active {
                    decoder,
                    worker_join,
                    worker_stop,
                    gst_codec: active_gst_codec,
                    video_codec: active_video_codec,
                    width: active_width,
                    height: active_height,
                });
                ActiveResult::SteadyState(Ok(()))
            }
            Err(e) => {
                guard.commit(DecoderState::Active {
                    decoder,
                    worker_join,
                    worker_stop,
                    gst_codec: active_gst_codec,
                    video_codec: active_video_codec,
                    width: active_width,
                    height: active_height,
                });
                frame_map.lock().remove(&ctx.frame_id);
                ActiveResult::SteadyState(Err(e.into()))
            }
        }
    }
}
