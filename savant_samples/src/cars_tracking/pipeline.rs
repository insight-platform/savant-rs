//! End-to-end streaming orchestrator for the `cars_tracking` sample.
//!
//! ```text
//! Mp4Demuxer -> FlexibleDecoderPool -> NvInfer -> NvTracker -> Picasso -> Mp4Muxer
//! ```
//!
//! This module is **orchestration only** — per-actor behavior lives
//! inside each stage module (see [`crate::cars_tracking`] for the
//! actor / carrier discussion).  [`run`] is responsible for:
//!
//! 1. Validating the resolved CLI.
//! 2. Building the inference / tracker operators and — when enabled —
//!    the Picasso engine (phase 1, ahead of `Stats::kick_off` to keep
//!    engine-build time out of the FPS counter).
//! 3. Allocating the bounded inter-actor channels.
//! 4. Spawning each actor via its `spawn_X_thread` API.
//! 5. Joining the actors in stage order and emitting the final
//!    summary.

use anyhow::{anyhow, bail, Context, Result};
use crossbeam::channel::bounded;
use deepstream_buffers::cuda_init;
use savant_core::pipeline::stats::Stats;
use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub mod blackhole;
pub mod decoder;
pub mod infer;
pub mod mp4_demux;
pub mod mp4_mux;
pub mod picasso;
pub mod tracker;

use self::blackhole::spawn_blackhole_thread;
use self::decoder::spawn_decoder_thread;
use self::infer::output::{
    build_result_callback as build_infer_callback, build_yolo_converter, InferStats,
};
use self::infer::{
    build_batch_formation as build_infer_batch_formation, spawn_infer_thread, InferThreadArgs,
};
use self::mp4_demux::spawn_mp4_demux_thread;
use self::mp4_mux::spawn_mp4_mux_thread;
use self::picasso::{build_picasso_engine, spawn_picasso_thread};
use self::tracker::{
    build_batch_formation as build_tracker_batch_formation,
    build_result_callback as build_tracker_callback, build_tracker_config, spawn_tracker_thread,
    TrackerStats,
};
use crate::cars_tracking::message::{EncodedMsg, PipelineMsg};
use crate::cars_tracking::pipeline::infer::model::build_nvinfer_config;
use crate::cars_tracking::stats::{make_stage, stage_frames, stage_objects, PipelineStats};
use crate::cars_tracking::supervisor::{exit_channel, StageExit, StageKind, StageName};
use crate::cars_tracking::warmup::{prepare_nvinfer_operator, prepare_nvtracker_operator};
use crate::cli::ResolvedCli;

/// Default source identifier used throughout the pipeline.  Every
/// stage uses the same id because the sample processes exactly one
/// input file.  The constant is passed *only* to
/// [`spawn_demux_thread`]; every downstream stage discovers the
/// source id in-band via the `SourceEos { source_id }` sentinels or
/// by reading [`savant_core::primitives::frame::VideoFrameProxy::get_source_id`]
/// off the frames themselves.
const SOURCE_ID: &str = "cars-demo";
/// Period (ms) at which [`Stats`] logs timestamp-based FPS while the
/// pipeline runs.  One second matches the prior custom reporter.
const STATS_TIMESTAMP_PERIOD_MS: i64 = 1_000;
/// Retained history length for [`Stats`].
const STATS_HISTORY_LEN: usize = 100;

/// Grace window the supervisor sleeps between observing the first
/// back-channel `StageExit` and broadcasting a cooperative
/// `Shutdown` sentinel on every inter-actor channel.  See the
/// supervisor comment in [`run`] for the race this papers over.
const SUPERVISOR_QUIESCENCE_GRACE: Duration = Duration::from_secs(12);

/// Run the cars-tracking pipeline end-to-end.  Blocks until the input
/// MP4 is fully processed and the output MP4 is finalized.
pub fn run(cli: ResolvedCli) -> Result<()> {
    if !cli.input.is_file() {
        bail!("input file does not exist: {}", cli.input.display());
    }
    if cli.picasso_enabled {
        let out = cli
            .output
            .as_ref()
            .ok_or_else(|| anyhow!("picasso enabled but output path is missing"))?;
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                bail!("output directory does not exist: {}", parent.display());
            }
        }
    }

    gstreamer::init().context("gstreamer init")?;
    cuda_init(cli.gpu).map_err(|e| anyhow!("cuda init (gpu={}): {e}", cli.gpu))?;

    log::info!(
        "cars-demo starting: input={} output={} gpu={} conf={} iou={} channel_cap={} fps={}/{} picasso_enabled={} draw_enabled={} debug={}",
        cli.input.display(),
        cli.output
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        cli.gpu,
        cli.conf,
        cli.iou,
        cli.channel_cap,
        cli.fps_num,
        cli.fps_den,
        cli.picasso_enabled,
        cli.draw_enabled,
        cli.debug
    );

    let stats = PipelineStats::new();

    // ── Per-stage counters ──────────────────────────────────────────────
    // We construct `Stats` up front so the stage counters can be
    // registered, but we deliberately defer `Stats::kick_off` until
    // *after* engine preparation.  Otherwise the timestamp-based FPS
    // reporter would happily log "0 FPS" lines during the
    // (potentially multi-minute) TensorRT engine build —
    // indistinguishable to the user from a stall.  See
    // [`crate::cars_tracking::warmup`] for the heartbeat scheme that
    // surfaces progress during the preparation phase.
    let core_stats = Arc::new(Stats::new(
        STATS_HISTORY_LEN,
        None,
        Some(STATS_TIMESTAMP_PERIOD_MS),
    ));
    let decoder_stage = make_stage("decoder");
    let infer_stage = make_stage("infer");
    let track_stage = make_stage("track");
    // The "tail" stage terminates the pipeline and is the one that
    // drives `Stats::register_frame` (and therefore the time-based
    // FPS log line).  Its logical identity differs with
    // `--no-picasso`: with Picasso it counts encoded frames, without
    // it counts frames that reach the drain thread.
    let tail_stage_name = if cli.picasso_enabled {
        "encode"
    } else {
        "blackhole"
    };
    let tail_stage = make_stage(tail_stage_name);
    core_stats.add_stage_stats(decoder_stage.clone());
    core_stats.add_stage_stats(infer_stage.clone());
    core_stats.add_stage_stats(track_stage.clone());
    core_stats.add_stage_stats(tail_stage.clone());

    // ── Channels ────────────────────────────────────────────────────────
    let (tx_demux, rx_demux) = bounded::<EncodedMsg>(cli.channel_cap);
    let (tx_decoded, rx_decoded) = bounded::<PipelineMsg>(cli.channel_cap);
    let (tx_infer, rx_infer) = bounded::<PipelineMsg>(cli.channel_cap);
    let (tx_tracker, rx_tracker) = bounded::<PipelineMsg>(cli.channel_cap);
    // The encoded channel only exists when Picasso is active: it
    // carries Picasso's `OutputMessage::VideoFrame` payloads into
    // the MP4 muxer.  With `--no-picasso` there is no producer and
    // no consumer, so we skip allocating the channel entirely.
    let encoded = if cli.picasso_enabled {
        Some(bounded::<EncodedMsg>(cli.channel_cap))
    } else {
        None
    };

    // ── Phase 1: engine preparation ─────────────────────────────────────
    log::info!(
        "[warmup] phase 1: preparing inference + tracker{} (no frames yet)",
        if cli.picasso_enabled {
            " + renderer"
        } else {
            " (picasso excluded)"
        }
    );
    let warmup_t0 = Instant::now();

    let infer_cfg = build_nvinfer_config(cli.gpu)?;
    // Actor identities established once up-front so each name is
    // shared by (a) the spawn call that installs the
    // `StageExitGuard`, (b) the thread body's log records, and
    // (c) operator result callbacks — logs from every path carry
    // the same identifier, which matters as soon as a pipeline
    // hosts concurrent actors of the same kind (e.g.
    // `infer[yolo11n]` vs `infer[person_attr]`).
    let mp4_demux_name = StageName::unnamed(StageKind::Mp4Demux);
    let decoder_name = StageName::unnamed(StageKind::Decoder);
    let infer_name = StageName::unnamed(StageKind::Infer);
    let tracker_name = StageName::unnamed(StageKind::Tracker);
    let picasso_name = StageName::unnamed(StageKind::Picasso);
    let mp4_mux_name = StageName::unnamed(StageKind::Mp4Mux);
    let blackhole_name = StageName::unnamed(StageKind::Blackhole);

    let infer_batch_cb = build_infer_batch_formation();
    let converter = build_yolo_converter(cli.conf, cli.iou);
    let infer_stats = Arc::new(InferStats::new());
    let infer_result_tx_cb = tx_infer.clone();
    let infer_drain_tx = tx_infer.clone();
    drop(tx_infer);
    let infer_result_cb = build_infer_callback(
        converter.clone(),
        infer_result_tx_cb,
        Some(infer_stats.clone()),
        infer_name.clone(),
    );
    let (infer_operator, _infer_build) =
        prepare_nvinfer_operator(cli.gpu, infer_cfg, infer_batch_cb, infer_result_cb)?;

    let tracker_cfg = build_tracker_config(cli.gpu)?;
    let tracker_batch_cb = build_tracker_batch_formation();
    let tracker_stats = Arc::new(TrackerStats::new());
    let tracker_result_tx_cb = tx_tracker.clone();
    let tracker_drain_tx = tx_tracker.clone();
    drop(tx_tracker);
    let tracker_result_cb = build_tracker_callback(
        tracker_result_tx_cb,
        Some(tracker_stats.clone()),
        tracker_name.clone(),
    );
    let (tracker_operator, _tracker_build) =
        prepare_nvtracker_operator(tracker_cfg, tracker_batch_cb, tracker_result_cb)?;

    let (picasso, tx_encoded, rx_encoded) = if let Some((tx_encoded, rx_encoded)) = encoded {
        let engine = Arc::new(build_picasso_engine(
            tx_encoded.clone(),
            stats.clone(),
            tail_stage.clone(),
            core_stats.clone(),
            picasso_name.clone(),
        ));
        log::info!("[warmup/picasso] renderer engine ready");
        (Some(engine), Some(tx_encoded), Some(rx_encoded))
    } else {
        log::info!("[warmup/picasso] skipped (--no-picasso): picasso + mux stages disabled");
        (None, None, None)
    };

    log::info!(
        "[warmup] phase 1 complete ({:.2}s) - starting pipeline threads + FPS reporter",
        warmup_t0.elapsed().as_secs_f64()
    );

    // ── Phase 2: live pipeline ──────────────────────────────────────────
    core_stats.kick_off();
    let pipeline_t0 = Instant::now();

    // Sender clones retained by the orchestrator so
    // [`broadcast_shutdown`] can inject a cooperative `Shutdown`
    // sentinel onto every inter-actor channel.  Each clone targets
    // the channel that the corresponding downstream actor *reads
    // from*:
    //
    //   shutdown_tx_demux   -> decoder  reads this (EncodedMsg::Shutdown)
    //   shutdown_tx_decoded -> infer    reads this (PipelineMsg::Shutdown)
    //   shutdown_tx_infer   -> tracker  reads this (PipelineMsg::Shutdown)
    //   shutdown_tx_tracker -> picasso/blackhole reads this (PipelineMsg::Shutdown)
    //   shutdown_tx_encoded -> mp4_mux  reads this (EncodedMsg::Shutdown) — only
    //                          when picasso is enabled.
    let shutdown_tx_demux = tx_demux.clone();
    let shutdown_tx_decoded = tx_decoded.clone();
    let shutdown_tx_infer = infer_drain_tx.clone();
    let shutdown_tx_tracker = tracker_drain_tx.clone();
    let shutdown_tx_encoded = tx_encoded.as_ref().cloned();
    drop(tx_encoded);

    // ── Supervisor back-channel ────────────────────────────────────────
    //
    // Every stage thread owns a `StageExitGuard` that pushes a
    // `StageExit { stage }` onto this channel when it drops — on
    // `Ok`, on `Err`, and during a panic unwind.  The orchestrator
    // blocks on `exit_rx.recv()` for the *first* signal, then
    // broadcasts `Shutdown { grace: None }` on every inter-actor
    // channel and joins every thread.
    //
    // Ctrl+C is routed through the same back-channel (synthetic
    // `StageKind::CtrlC`) so user-cancel flows through one uniform
    // recv-point alongside natural exits and fatal stage errors.
    // A second Ctrl+C falls through to the default SIGINT handler
    // and aborts the process.
    //
    // Back-channel is unbounded — `StageExitGuard::drop` must never
    // block, including during a panic unwind.  See the module-level
    // `Why unbounded?` note in [`super::supervisor`].
    let (exit_tx, exit_rx) = exit_channel();
    {
        let exit_tx_ctrl = exit_tx.clone();
        if let Err(e) = ctrlc::try_set_handler(move || {
            log::warn!("[ctrl-c] received; notifying supervisor");
            let _ = exit_tx_ctrl.send(StageExit {
                stage: StageName::unnamed(StageKind::CtrlC),
            });
        }) {
            log::warn!("[ctrl-c] could not install handler: {e}");
        }
    }

    // This pipeline runs a single actor of each [`StageKind`], so
    // every stage gets an unnamed [`StageName`] (see the block
    // above where the names are bound).  A pipeline hosting
    // several nvinfer models (e.g. `yolo11n` + `person_attr`)
    // would bind those locals with
    // `StageName::new(StageKind::Infer, "yolo11n")` etc. to
    // disambiguate back-channel signals and log records.
    let mp4_demux_handle = spawn_mp4_demux_thread(
        SOURCE_ID.to_string(),
        cli.input.to_string_lossy().into_owned(),
        tx_demux,
        stats.clone(),
        exit_tx.clone(),
        mp4_demux_name,
    )?;

    let decoder_handle = spawn_decoder_thread(
        cli.gpu,
        rx_demux,
        tx_decoded,
        decoder_stage.clone(),
        exit_tx.clone(),
        decoder_name,
    )?;

    // Clones kept for the end-of-run summary (the originals are
    // moved into the stage threads below).
    let infer_stats_summary = infer_stats.clone();
    let tracker_stats_summary = tracker_stats.clone();

    let infer_handle = spawn_infer_thread(
        InferThreadArgs {
            operator: infer_operator,
            rx: rx_decoded,
            drain_tx: infer_drain_tx,
            converter,
            stats: infer_stats,
            stage: infer_stage.clone(),
        },
        exit_tx.clone(),
        infer_name,
    )?;

    let tracker_handle = spawn_tracker_thread(
        tracker_operator,
        rx_infer,
        tracker_drain_tx,
        tracker_stats,
        track_stage.clone(),
        exit_tx.clone(),
        tracker_name,
    )?;

    // Picasso (optional) + terminus (mux or drain).
    //
    //   * Picasso enabled  — picasso feeds mux via the encoded
    //     channel; mux is the pipeline terminus.
    //   * --no-picasso     — drain is the terminus, no picasso.
    //
    // The orchestrator no longer treats the terminus specially at
    // join time: the supervisor back-channel decides *when* to
    // start winding down, then every handle is joined in
    // downstream -> upstream order for a deterministic summary.
    let (picasso_handle, terminus_handle) = match picasso {
        Some(engine) => {
            let picasso_stage = spawn_picasso_thread(
                engine,
                rx_tracker,
                cli.draw_enabled,
                exit_tx.clone(),
                picasso_name,
            )?;
            let out = cli
                .output
                .as_ref()
                .ok_or_else(|| anyhow!("picasso enabled but output path is missing"))?
                .to_string_lossy()
                .into_owned();
            let rx_encoded = rx_encoded
                .ok_or_else(|| anyhow!("picasso enabled but encoded channel is missing"))?;
            let mux = spawn_mp4_mux_thread(
                out,
                cli.fps_num,
                cli.fps_den,
                rx_encoded,
                stats.clone(),
                exit_tx.clone(),
                mp4_mux_name,
            )?;
            (Some(picasso_stage), mux)
        }
        None => {
            let drain = spawn_blackhole_thread(
                rx_tracker,
                tail_stage.clone(),
                core_stats.clone(),
                exit_tx.clone(),
                blackhole_name,
            )?;
            (None, drain)
        }
    };

    // Drop our local `exit_tx`: the only remaining senders now are
    // the per-stage guards (and the Ctrl+C handler's clone, which
    // stays alive for the lifetime of the program).  This keeps
    // `exit_rx` live until all stages have dropped their guards.
    drop(exit_tx);

    // ── Supervisor: first exit → quiescence grace → broadcast Shutdown ─
    //
    // Every [`StageExit`] is treated as shutdown-worthy in this
    // sample — the demuxer draining a finite input file is the
    // normal trigger.  The [`StageKind`] + instance tag carried on
    // [`StageName`] is still threaded through every log record so
    // operators can see *which* stage won the race, which matters
    // as soon as the pipeline hosts multiple actors of the same
    // kind.
    //
    // After observing the first exit the supervisor sleeps for
    // [`SUPERVISOR_QUIESCENCE_GRACE`] before broadcasting.  The
    // grace matters because the first exit is usually `mp4_demux`
    // racing to end-of-file while many unconsumed packets are
    // still buffered inside NVDEC + downstream channels.  The
    // in-band `SourceEos` is always *behind* those packets on the
    // same channel (FIFO, same producer), but the supervisor's
    // `Shutdown` is a *parallel* producer on the same channels, so
    // without a grace window it can overtake the in-band drain
    // and force-exit stages while packets still linger.  Sleeping
    // briefly lets the natural drain reach each stage first;
    // anything still blocked afterwards (e.g. the blackhole, which
    // intentionally never self-terminates) is unblocked by the
    // broadcast.
    //
    // `recv()` can only return `Err` if every sender (including the
    // Ctrl+C handler's clone) has been dropped without pushing a
    // signal, which is impossible given every stage thread holds a
    // guard.  We still handle it defensively so the orchestrator
    // degrades to "broadcast anyway" rather than hanging.
    let first_exit = match exit_rx.recv() {
        Ok(exit) => {
            log::info!(
                "[supervisor] stage exit observed from [{}]; waiting {:?} for natural drain before broadcasting Shutdown",
                exit.stage,
                SUPERVISOR_QUIESCENCE_GRACE
            );
            Some(exit)
        }
        Err(e) => {
            log::error!("[supervisor] back-channel closed before any exit signal: {e}");
            None
        }
    };

    std::thread::sleep(SUPERVISOR_QUIESCENCE_GRACE);

    broadcast_shutdown(
        &shutdown_tx_demux,
        &shutdown_tx_decoded,
        &shutdown_tx_infer,
        &shutdown_tx_tracker,
        shutdown_tx_encoded.as_ref(),
        "supervisor",
    );
    // Drop our retained sender clones so the actors observe
    // `Disconnected` after consuming the `Shutdown` sentinel.
    drop(shutdown_tx_demux);
    drop(shutdown_tx_decoded);
    drop(shutdown_tx_infer);
    drop(shutdown_tx_tracker);
    drop(shutdown_tx_encoded);

    // Join every stage.  Order is downstream -> upstream so a panic
    // log surfaces at the leaf first, but correctness no longer
    // depends on the order — Shutdown has been broadcast and every
    // stage has either drained naturally or observed the sentinel.
    let terminus_result = terminus_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("terminus panic")));
    let picasso_result = match picasso_handle {
        Some(h) => h.join().unwrap_or_else(|_| Err(anyhow!("picasso panic"))),
        None => Ok(()),
    };
    let tracker_result = tracker_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("tracker panic")));
    let infer_result = infer_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("infer panic")));
    let decoder_result = decoder_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("decoder panic")));
    let mp4_demux_result = mp4_demux_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("mp4_demux panic")));

    // Drain any residual exit signals for diagnostic visibility.
    // Every joined thread fired its guard; we already consumed the
    // first, the rest sit in the back-channel.
    if let Some(first) = first_exit {
        log::debug!("[supervisor] triggering stage: {}", first.stage);
    }
    while let Ok(extra) = exit_rx.try_recv() {
        log::debug!("[supervisor] subsequent exit from [{}]", extra.stage);
    }

    core_stats.log_final_fps();

    let pipeline_elapsed = pipeline_t0.elapsed();
    let tail_frames = stage_frames(&tail_stage);
    let pipeline_fps = if pipeline_elapsed.as_secs_f64() > 0.0 {
        tail_frames as f64 / pipeline_elapsed.as_secs_f64()
    } else {
        0.0
    };
    log::info!(
        "cars-demo done: pipeline_runtime={:.2}s avg_fps={:.1} mp4_demux={} decoded={} infer_frames={} detections={} track_frames={} unique_tracks={} unmatched_updates={} {}={} bytes={}",
        pipeline_elapsed.as_secs_f64(),
        pipeline_fps,
        stats.demux_packets.load(Ordering::Relaxed),
        stage_frames(&decoder_stage),
        stage_frames(&infer_stage),
        stage_objects(&infer_stage),
        stage_frames(&track_stage),
        tracker_stats_summary.unique_tracks(),
        tracker_stats_summary.unmatched_updates(),
        tail_stage_name,
        tail_frames,
        stats.encoded_bytes.load(Ordering::Relaxed),
    );
    log::debug!(
        "final InferStats: frames={} detections={}",
        infer_stats_summary.frames(),
        infer_stats_summary.detections()
    );
    drop(core_stats);

    let terminus_label = if cli.picasso_enabled {
        "mp4_mux"
    } else {
        "blackhole"
    };
    for (name, r) in [
        ("mp4_demux", &mp4_demux_result),
        ("decoder", &decoder_result),
        ("infer", &infer_result),
        ("tracker", &tracker_result),
        ("picasso", &picasso_result),
        (terminus_label, &terminus_result),
    ] {
        if let Err(e) = r {
            log::error!("[{name}] stage error: {e:#}");
        }
    }
    mp4_demux_result?;
    decoder_result?;
    infer_result?;
    tracker_result?;
    picasso_result?;
    terminus_result?;
    Ok(())
}

/// Fan an immediate cooperative `Shutdown` sentinel out to every
/// inter-actor channel.  Sends are best-effort — receivers may have
/// already exited (e.g. the stage observed `Disconnected` after its
/// upstream dropped) and a closed channel is not a failure.  The
/// orchestrator calls this once it observes the first back-channel
/// stage exit.
fn broadcast_shutdown(
    tx_demux: &crossbeam::channel::Sender<EncodedMsg>,
    tx_decoded: &crossbeam::channel::Sender<PipelineMsg>,
    tx_infer: &crossbeam::channel::Sender<PipelineMsg>,
    tx_tracker: &crossbeam::channel::Sender<PipelineMsg>,
    tx_encoded: Option<&crossbeam::channel::Sender<EncodedMsg>>,
    reason: &'static str,
) {
    log::info!("[shutdown] broadcasting immediate Shutdown: reason={reason}");
    let r: Cow<'static, str> = Cow::Borrowed(reason);
    if tx_demux
        .send(EncodedMsg::Shutdown {
            grace: None,
            reason: r.clone(),
        })
        .is_err()
    {
        log::debug!("[shutdown] tx_demux already closed");
    }
    if tx_decoded
        .send(PipelineMsg::Shutdown {
            grace: None,
            reason: r.clone(),
        })
        .is_err()
    {
        log::debug!("[shutdown] tx_decoded already closed");
    }
    if tx_infer
        .send(PipelineMsg::Shutdown {
            grace: None,
            reason: r.clone(),
        })
        .is_err()
    {
        log::debug!("[shutdown] tx_infer already closed");
    }
    if tx_tracker
        .send(PipelineMsg::Shutdown {
            grace: None,
            reason: r.clone(),
        })
        .is_err()
    {
        log::debug!("[shutdown] tx_tracker already closed");
    }
    if let Some(tx_encoded) = tx_encoded {
        if tx_encoded
            .send(EncodedMsg::Shutdown {
                grace: None,
                reason: r,
            })
            .is_err()
        {
            log::debug!("[shutdown] tx_encoded already closed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Integration guard: the orchestrator must fail fast when the
    /// input file is missing, before touching GStreamer / CUDA.
    #[test]
    fn run_rejects_missing_input() {
        let cli = ResolvedCli {
            input: PathBuf::from("/definitely/does/not/exist.mp4"),
            output: Some(std::env::temp_dir().join("savant_samples_pipeline_out.mp4")),
            gpu: 0,
            conf: 0.25,
            iou: 0.45,
            channel_cap: 4,
            debug: false,
            fps_num: 25,
            fps_den: 1,
            draw_enabled: true,
            picasso_enabled: true,
        };
        let err = run(cli).unwrap_err();
        assert!(
            err.to_string().contains("input file does not exist"),
            "unexpected error: {err:#}"
        );
    }
}
