//! Inference actor — YOLO11n detection for the `cars_tracking` sample.
//!
//! Responsibilities are split between this module and two
//! sibling submodules:
//!
//! * [`model`] — nvinfer engine/config, label tables, YOLO class
//!   constants.
//! * [`output`] — concrete tensor-output processing: the
//!   [`OperatorResultCallback`] that decodes the YOLO `output0` tensor
//!   into `VideoObject`s, the [`InferResultSender`] /
//!   [`InferResultReceiver`] channel aliases that carry the sealed
//!   deliveries to the tracker, and the [`InferStats`] counters.
//! * *This module* — operator lifecycle: batch-formation callback,
//!   the multiplexed receive loop, `send_eos` fan-out per source,
//!   `graceful_shutdown` + drain at the end of the pipeline.
//!
//! The nvinfer result callback runs on the operator's internal completion
//! thread.  It is forbidden to `unseal()` there — doing so would block that
//! thread and pin GPU slots for the whole time the downstream stage takes
//! to consume the frame.  Instead the callback:
//!
//! 1. Reads every output tensor while the [`OperatorInferenceOutput`] is
//!    still alive (pointers are only valid for that lifetime),
//! 2. Attaches decoded detections as `VideoObject`s onto the frame,
//! 3. Calls `take_deliveries()` and forwards the resulting
//!    [`SealedDeliveries`] through a bounded channel.
//!
//! A dedicated consumer thread (in [`crate::cars_tracking::pipeline`]) pulls
//! the sealed handles off the channel, calls `unseal()` and submits the
//! frames to the tracker — this keeps backpressure healthy and ensures GPU
//! slots are freed promptly.
//!
//! Source-EOS propagation is **stream-aligned** — it must leave the
//! stage only *after* the last delivery for that source has left the
//! stage, otherwise downstream can observe `SourceEos` before a
//! still-in-flight frame for the same source.  The operator's
//! completion thread (i.e. the callback) is the only place where that
//! invariant holds: the operator delivers
//! [`OperatorOutput::Inference`] in per-source order and emits
//! [`OperatorOutput::Eos { source_id }`](OperatorOutput::Eos) *after*
//! the last delivery for that source.
//!
//! So the responsibilities split like this:
//!
//! * **Infer thread** — on upstream [`PipelineMsg::SourceEos`] calls
//!   [`NvInferBatchingOperator::send_eos`] to *initiate* the drain.
//!   It does **not** touch the downstream channel.
//! * **Operator callback** — on [`OperatorOutput::Eos { source_id }`](
//!   OperatorOutput::Eos) forwards a single
//!   [`PipelineMsg::SourceEos { source_id }`](PipelineMsg::SourceEos) on
//!   the downstream channel.  This keeps the sentinel in-band with
//!   the delivery stream, satisfying the ordering invariant.
//!
//! [`InferResultSender`]: output::InferResultSender
//! [`InferResultReceiver`]: output::InferResultReceiver
//! [`InferStats`]: output::InferStats
//! [`OperatorInferenceOutput`]: deepstream_nvinfer::OperatorInferenceOutput
//! [`OperatorResultCallback`]: deepstream_nvinfer::OperatorResultCallback
//! [`SealedDeliveries`]: deepstream_nvinfer::SealedDeliveries

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::RecvTimeoutError;
use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvinfer::prelude::NvInferBatchingOperator;
use deepstream_nvinfer::{BatchFormationCallback, BatchFormationResult, RoiKind};
use savant_core::converters::YoloDetectionConverter;
use savant_core::pipeline::stats::StageStats;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub mod model;
pub mod output;

use self::output::{process_infer_output, InferResultSender, InferStats};
use super::decoder::{handle_shutdown, DecodedReceiver};
use crate::cars_tracking::message::PipelineMsg;
use crate::cars_tracking::stats::tick_stage;
// `name` is passed in per spawn call so a pipeline with multiple
// concurrent nvinfer models (e.g. `infer[yolo11n]`,
// `infer[person_attr]`) can distinguish their back-channel exits.
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};

/// Maximum time [`NvInferBatchingOperator::graceful_shutdown`] waits
/// for in-flight batches to complete.  Large value because TensorRT
/// may still be processing the last frame when shutdown is signaled.
const INFER_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Receive poll timeout.  The infer actor is multiplexed — every
/// [`PipelineMsg::SourceEos`] is a per-source flush signal, *not* a
/// loop terminator; the loop exits on upstream channel closure or on
/// a [`PipelineMsg::Shutdown`] broadcast by the orchestrator's
/// shutdown manager.  A bounded `recv_timeout` wakes periodically so
/// grace-deadline checks can't starve and `flush_idle` still runs.
const INFER_RECV_POLL: Duration = Duration::from_millis(100);

/// Build the batch-formation callback used by `NvInferBatchingOperator`.
///
/// Each slot processes a single full-frame ROI (the whole image).
pub fn build_batch_formation() -> BatchFormationCallback {
    Arc::new(|frames| {
        let ids = frames
            .iter()
            .enumerate()
            .map(|(slot, _)| SavantIdMetaKind::Frame(slot as u128))
            .collect();
        let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
        BatchFormationResult { ids, rois }
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Infer actor thread
// ═══════════════════════════════════════════════════════════════════════

/// Arguments bundle for [`spawn_infer_thread`].  Bundled into a struct
/// because the thread body needs many distinct handles; a positional
/// function signature would be too easy to mis-call.
pub struct InferThreadArgs {
    /// The nvinfer operator whose lifecycle the infer actor drives.
    pub operator: NvInferBatchingOperator,
    /// Upstream receiver — decode -> infer channel.
    pub rx: DecodedReceiver,
    /// Downstream sender — infer -> tracker channel.  Also used by
    /// `graceful_shutdown` drain handling (via
    /// [`process_infer_output`]).
    pub drain_tx: InferResultSender,
    /// YOLO post-processing converter shared with the operator
    /// callback.
    pub converter: Arc<YoloDetectionConverter>,
    /// Detection + frame counters.
    pub stats: Arc<InferStats>,
    /// Per-stage counter registered with
    /// [`savant_core::pipeline::stats::Stats`].
    pub stage: StageStats,
}

/// Spawn the infer actor.
///
/// The actor is **multiplexed** — the receive loop serves every
/// `source_id` that flows through the decode -> infer channel.
/// Per-source events translate to operator calls only — downstream
/// propagation is the operator callback's job (see
/// [`process_infer_output`]):
///
/// * [`PipelineMsg::Delivery`] — unsealed and fed to
///   [`NvInferBatchingOperator::add_frame`].
/// * [`PipelineMsg::SourceEos { source_id }`](PipelineMsg::SourceEos) —
///   calls `operator.send_eos(&source_id)` to initiate the per-source
///   drain.  The callback will emit [`PipelineMsg::SourceEos`] once the
///   operator has delivered the last result for that source.  The
///   actor does **not** break on this event; downstream stages are
///   also multiplexed and the upstream channel may still carry
///   deliveries from other sources.
/// * [`PipelineMsg::Shutdown`] — cooperative-exit sentinel.  `None`
///   grace breaks after the current message; `Some(d)` sets a
///   deadline and keeps running.
///
/// After the main loop exits, the actor calls
/// [`NvInferBatchingOperator::graceful_shutdown`] once (pipeline
/// hammer, not per-source) and drains any remaining outputs through
/// [`process_infer_output`] before returning.
pub fn spawn_infer_thread(
    args: InferThreadArgs,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-infer".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            infer_thread(args)
        })
        .context("spawn infer thread")
}

fn infer_thread(args: InferThreadArgs) -> Result<()> {
    let InferThreadArgs {
        mut operator,
        rx,
        drain_tx,
        converter,
        stats,
        stage,
    } = args;
    log::info!("[infer] starting");
    // Baseline for per-frame detection deltas (the operator callback
    // increments `stats.detections()` asynchronously, so we snapshot
    // the cumulative count before each `add_frame` to derive the
    // per-call object delta for the pipeline stage counters).
    let mut det_baseline = stats.detections();
    let mut deadline: Option<Instant> = None;
    let mut break_now = false;
    loop {
        match rx.recv_timeout(INFER_RECV_POLL) {
            Ok(msg @ (PipelineMsg::Delivery(_) | PipelineMsg::Deliveries(_))) => {
                // Generalized ingress: `Delivery` (singular) and
                // `Deliveries` (batched) are both normalized to a
                // flat `Vec<(frame, buffer)>` and fed one-by-one to
                // the nvinfer operator.  This is the "one code
                // path per stage" shape that the unified
                // [`PipelineMsg`](PipelineMsg) was introduced for.
                for (frame, buffer) in msg.into_pairs() {
                    if let Err(e) = operator.add_frame(frame, buffer) {
                        log::error!("[infer] add_frame failed: {e}");
                        return Err(anyhow!("infer add_frame: {e}"));
                    }
                    let now = stats.detections();
                    tick_stage(&stage, 1, (now - det_baseline) as usize);
                    det_baseline = now;
                }
            }
            Ok(PipelineMsg::SourceEos { source_id }) => {
                // Per-source flush only.  The downstream `SourceEos`
                // sentinel is emitted by the operator callback on
                // `OperatorOutput::Eos` to stay stream-aligned with
                // the delivery stream — see [`process_infer_output`].
                // Do NOT break — other sources may still be
                // producing deliveries through this multiplexed
                // actor.
                log::info!("[infer] SourceEos {source_id}: initiating operator drain");
                if let Err(e) = operator.send_eos(&source_id) {
                    log::warn!("[infer] send_eos({source_id}) failed: {e}");
                }
            }
            Ok(PipelineMsg::Shutdown { grace, reason }) => {
                handle_shutdown("infer", grace, &reason, &mut deadline, &mut break_now);
            }
            Err(RecvTimeoutError::Timeout) => {
                // Force-flush pending rescue-eligible custom-downstream
                // events (e.g. a trailing `savant.pipeline.source_eos`)
                // through the operator's internal GStreamer pipeline
                // when it is idle.  Same rationale as decode.rs.
                if let Err(e) = operator.flush_idle() {
                    log::warn!("[infer] flush_idle failed: {e}");
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[infer] upstream channel disconnected; exiting receive loop");
                break;
            }
        }
        // Per-iteration exit checks (see decode.rs for rationale —
        // the grace deadline must not be gated on the `Timeout`
        // branch, otherwise a chatty upstream would keep us alive
        // past the deadline).
        if break_now {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                log::info!("[infer] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }

    // Pipeline-level drain.  `send_eos` has already been issued per
    // source, so this is the final hammer that joins the operator's
    // internal workers and harvests any outputs the callback hasn't
    // delivered yet.
    match operator.graceful_shutdown(INFER_DRAIN_TIMEOUT) {
        Ok(drained) => {
            log::info!("[infer] drained {} remaining outputs", drained.len());
            for out in drained {
                process_infer_output(out, converter.as_ref(), &drain_tx, Some(&stats));
            }
        }
        Err(e) => {
            log::error!("[infer] graceful_shutdown failed: {e}");
        }
    }
    // Absorb any detections registered after the last tick above
    // (including results produced by `graceful_shutdown`) so the
    // stage object counter matches the sample-level InferStats.
    let final_dets = stats.detections();
    if final_dets > det_baseline {
        tick_stage(&stage, 0, (final_dets - det_baseline) as usize);
    }
    drop(drain_tx);
    drop(operator);
    log::info!(
        "[infer] finished: frames={} detections={}",
        stats.frames(),
        stats.detections()
    );
    Ok(())
}
