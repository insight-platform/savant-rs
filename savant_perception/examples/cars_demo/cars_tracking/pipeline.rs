//! End-to-end streaming orchestrator for the `cars_tracking` sample.
//!
//! ```text
//! Mp4Demuxer -> FlexibleDecoderPool -> NvInfer -> NvTracker -> Picasso -> Mp4Muxer
//! ```
//!
//! This module is **orchestration only** — per-actor behaviour lives
//! inside the framework Layer-B templates
//! ([`savant_perception::templates`]) plus a thin set of per-stage
//! output processors retained under this module's submodules
//! (nvinfer post-processing, nvtracker reconciliation, picasso
//! source-spec builder).  [`run`] is responsible for:
//!
//! 1. Validating the resolved CLI.
//! 2. Building the inference / tracker operators and — when enabled —
//!    the Picasso engine inside the actor/source factories run by
//!    [`System::build`](savant_perception::System).  Engine preparation
//!    happens lazily on the actor's own thread, which keeps slow TRT
//!    builds out of the main thread and allows the framework to
//!    surface spawn/factory errors uniformly.
//! 3. Registering every actor + source with the
//!    [`System`](savant_perception::System); the system handles
//!    channel allocation, address publication, thread spawning,
//!    cooperative shutdown, and join in a single call.
//! 4. Kicking off the time-based FPS reporter and blocking on
//!    [`System::run`](savant_perception::System::run) until the
//!    pipeline terminates naturally or the supervisor observes a
//!    fatal exit.
//! 5. Logging the final aggregated summary.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use ::picasso::prelude::OnEncodedFrame;
use ::picasso::{Callbacks, GeneralSpec, OutputMessage, PicassoEngine};
use anyhow::{anyhow, bail, Context, Result};
use deepstream_buffers::cuda_init;
use savant_core::pipeline::stats::{StageStats, Stats};
use savant_core::primitives::frame::VideoFrameContent;

pub mod infer;
pub mod picasso;
pub mod tracker;

use self::infer::build_batch_formation as build_infer_batch_formation;
use self::infer::model::build_nvinfer_config;
use self::infer::output::{build_yolo_converter, process_infer_output, InferStats};
use self::picasso::build_source_spec;
use self::picasso::draw_spec::attach_frame_id_overlay;
use self::tracker::{
    build_batch_formation as build_tracker_batch_formation, build_tracker_config,
    process_tracker_output, TrackerStats,
};
use crate::cars_tracking::stats::{
    make_stage, stage_frames, stage_objects, tick_stage, PipelineStats,
};
use crate::cars_tracking::warmup::prepare_nvinfer_operator;
use crate::cli::ResolvedCli;
use savant_perception::envelopes::{EncodedMsg, PipelineMsg};
use savant_perception::router::Router;
use savant_perception::shutdown::{ShutdownAction, ShutdownCause, ShutdownCtx};
use savant_perception::supervisor::{StageKind, StageName};
use savant_perception::templates::decoder::make_decode_frame;
use savant_perception::templates::{
    BitstreamFunction, BitstreamFunctionInbox, Decoder, DecoderResults, Function, FunctionInbox,
    Mp4DemuxerResults, Mp4DemuxerSource, Mp4Muxer, NvInfer, NvInferResults, NvTracker,
    NvTrackerResults, Picasso, PicassoInbox,
};
use savant_perception::{Flow, HookCtx, System};

/// Default source identifier used throughout the pipeline.  Every
/// stage uses the same id because the sample processes exactly one
/// input file.
const SOURCE_ID: &str = "cars-demo";
/// Period (ms) at which [`Stats`] logs timestamp-based FPS while the
/// pipeline runs.  One second matches the prior custom reporter.
const STATS_TIMESTAMP_PERIOD_MS: i64 = 1_000;
/// Retained history length for [`Stats`].
const STATS_HISTORY_LEN: usize = 100;

// ── Shared-state role keys ──────────────────────────────────────────────
//
// Stage-role names used as [`SharedStore`] keys for the per-stage
// [`StageStats`] handles.  Looked up from egress hooks via
// `HookCtx::shared_as::<StageStats>(role)` — the demo's whole
// point is to show that hook bodies don't need any per-closure
// state capture beyond the framework's shared-state lookup.
const ROLE_DECODER: &str = "decoder";
const ROLE_INFER: &str = "infer";
const ROLE_TRACKER: &str = "track";
const ROLE_TAIL: &str = "tail";

/// Run the cars-tracking pipeline end-to-end.  Blocks until the input
/// MP4 is fully processed and the output MP4 is finalized.
pub fn run(cli: ResolvedCli) -> Result<()> {
    if !cli.input.is_file() {
        bail!("input file does not exist: {}", cli.input.display());
    }
    // The `--output null` sentinel (`output_is_null = true`) keeps
    // Picasso running but redirects the encoded bitstream into a
    // `BitstreamFunction` terminus instead of an MP4 file, so in
    // that mode `cli.output` is intentionally `None` and the
    // filesystem check must be skipped.  Only require an output
    // path when Picasso is enabled *and* the user did not opt into
    // the null sentinel.
    if cli.picasso_enabled && !cli.output_is_null {
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

    // ── Shared state ────────────────────────────────────────────────────
    let stats = PipelineStats::new();
    let core_stats = Arc::new(Stats::new(
        STATS_HISTORY_LEN,
        None,
        Some(STATS_TIMESTAMP_PERIOD_MS),
    ));
    let decoder_stage = make_stage("decoder");
    let infer_stage = make_stage("infer");
    let track_stage = make_stage("track");
    let tail_stage_name = match (cli.picasso_enabled, cli.output_is_null) {
        (true, true) => "bitstream",
        (true, false) => "encode",
        (false, _) => "function",
    };
    let tail_stage = make_stage(tail_stage_name);
    core_stats.add_stage_stats(decoder_stage.clone());
    core_stats.add_stage_stats(infer_stage.clone());
    core_stats.add_stage_stats(track_stage.clone());
    core_stats.add_stage_stats(tail_stage.clone());

    let infer_stats = Arc::new(InferStats::new());
    let tracker_stats = Arc::new(TrackerStats::new());

    // ── Stage identities ────────────────────────────────────────────────
    let demux_name = StageName::unnamed(StageKind::Mp4Demux);
    let decoder_name = StageName::unnamed(StageKind::Decoder);
    let infer_name = StageName::unnamed(StageKind::Infer);
    let tracker_name = StageName::unnamed(StageKind::Tracker);
    let picasso_name = StageName::unnamed(StageKind::Picasso);
    let mp4_mux_name = StageName::unnamed(StageKind::Mp4Mux);
    let function_name = StageName::unnamed(StageKind::Function);

    // Pick the logical tail stage downstream of the tracker.  The
    // chosen name is used both as the tracker's downstream target
    // and in the final summary label.
    let tail_actor_name = if cli.picasso_enabled {
        picasso_name.clone()
    } else {
        function_name.clone()
    };

    // ── Build System ────────────────────────────────────────────────────
    //
    // Install a sample-specific shutdown policy: the Mp4Demuxer
    // finishing its input is an expected end-of-stream, not a
    // shutdown trigger — the in-band `SourceEos` sentinel already
    // flows through to the terminus (Mp4Muxer or Function).  We
    // wait for the *terminus* to exit (after draining) and treat
    // its `StageExit` as the real "pipeline is done" trigger.
    // No quiescence pause is needed: the terminus only exits once
    // everything upstream has already drained.
    let mut sys = System::new().on_shutdown(cars_shutdown_handler);

    // ── Publish shared state ────────────────────────────────────────────
    //
    // The demo hooks consume these handles via
    // [`HookCtx::shared`] / [`HookCtx::shared_as`] instead of
    // capturing them by closure — showcasing the framework's
    // [`SharedStore`] as the canonical hand-off point for
    // cross-stage counters in end-user pipelines.
    sys.insert_shared_arc::<PipelineStats>(stats.clone());
    sys.insert_shared_arc::<Stats>(core_stats.clone());
    sys.insert_shared_arc::<InferStats>(infer_stats.clone());
    sys.insert_shared_arc::<TrackerStats>(tracker_stats.clone());
    sys.insert_shared_as::<StageStats>(ROLE_DECODER, decoder_stage.clone());
    sys.insert_shared_as::<StageStats>(ROLE_INFER, infer_stage.clone());
    sys.insert_shared_as::<StageStats>(ROLE_TRACKER, track_stage.clone());
    sys.insert_shared_as::<StageStats>(ROLE_TAIL, tail_stage.clone());

    // ── Demuxer source ──────────────────────────────────────────────────
    //
    // The `Mp4DemuxerSource` template never calls `router.send` itself.
    // Every downstream dispatch below happens inside user code — the
    // variant hooks receive `&Router<EncodedMsg>` and the sample
    // chooses the default peer via plain `router.send(...)`.  Source
    // operators that need to route differently per `source_id` (e.g.
    // fan out to distinct decoder pools) would call
    // `router.send_to(&decoder_for(&source_id), msg)` here instead.
    //
    // Note: we build the `VideoFrameProxy` on the **demuxer side**
    // via `make_decode_frame` + `EncodedMsg::Frame`.  Swap this for
    // `Mp4DemuxerSource::default_on_packet()` if you want the
    // downstream decoder to own frame construction — the decoder
    // no longer tracks `VideoInfo` state, every
    // `EncodedMsg::Packet` carries it in-band instead.
    let demux_src = Mp4DemuxerSource::builder(demux_name.clone())
        .input(cli.input.to_string_lossy().into_owned())
        .source_id(SOURCE_ID)
        .downstream(decoder_name.clone())
        .results(
            Mp4DemuxerResults::builder()
                .on_packet(|pkt, info, source_id, router, ctx| {
                    if let Some(stats) = ctx.shared::<PipelineStats>() {
                        stats.demux_packets.fetch_add(1, Ordering::Relaxed);
                    }
                    let frame = make_decode_frame(source_id, &pkt, info);
                    router.send(EncodedMsg::Frame {
                        frame,
                        payload: Some(pkt.data),
                    });
                    Ok(())
                })
                .build(),
        )
        .build()?;
    sys.register_source(demux_src)?;

    // ── Decoder actor ───────────────────────────────────────────────────
    //
    // The `Decoder` template never sends: every variant hook
    // below is the user-facing send site.  Because every hook receives
    // `&Router<PipelineMsg>`, source-id-based routing (e.g.
    // `router.send_to(&engine_for(&sealed), PipelineMsg::Delivery(sealed))`
    // for fan-out to multiple inference engines) would plug in here.
    //
    // Only `on_frame` needs sample-specific code (forward + stage
    // tick).  `on_source_eos`, `on_parameter_change`, `on_skipped`,
    // `on_orphan_frame`, `on_decoder_event` and `on_decoder_error`
    // fall through to the template's `Decoder::default_on_*`
    // forwarders/loggers, installed automatically by the builder
    // when their setters are omitted.
    let decoder = Decoder::builder(decoder_name.clone(), cli.channel_cap)
        .downstream(infer_name.clone())
        .gpu_id(cli.gpu)
        .results(
            DecoderResults::builder()
                .on_frame(|sealed, router, ctx| {
                    if router.send(PipelineMsg::Delivery(sealed)) {
                        if let Some(stage) = ctx.shared_as::<StageStats>(ROLE_DECODER) {
                            tick_stage(&stage, 1, 0);
                        }
                    }
                })
                .build(),
        )
        .build()?;
    sys.register_actor(decoder)?;

    // ── NvInfer actor ───────────────────────────────────────────────────
    //
    // Routing style 1 — DEFAULT PEER.  `.downstream(tracker_name)`
    // installs a default peer on the actor's `Router<PipelineMsg>`,
    // so `router.send(msg)` inside the output callback always
    // targets the tracker.  This is the terse, config-once idiom for
    // stages whose downstream is known at build time.
    let infer_cfg = build_nvinfer_config(cli.gpu)?;
    let infer_batch_cb = build_infer_batch_formation();
    let converter = build_yolo_converter(cli.conf, cli.iou);
    let gpu = cli.gpu;
    let infer = NvInfer::builder(infer_name.clone(), cli.channel_cap)
        .downstream(tracker_name.clone())
        .operator_factory(move |_bx, result_cb| {
            let (op, _elapsed) =
                prepare_nvinfer_operator(gpu, infer_cfg, infer_batch_cb, result_cb)?;
            Ok(op)
        })
        // Only the Inference branch needs sample-specific code — YOLO
        // decode + stats tick.  Source-EOS propagation and operator-
        // error logging fall through to the template defaults
        // (`NvInfer::default_on_source_eos` /
        // `NvInfer::default_on_error`), installed automatically
        // by the `NvInferResults` bundle when their setters are
        // omitted.  All counters are pulled from the shared store
        // via the `HookCtx` — no per-closure `Arc` captures.
        .results(
            NvInferResults::builder()
                .on_inference({
                    let converter = converter.clone();
                    move |inf, router: &Router<PipelineMsg>, ctx| {
                        let frame_count = inf.frames().len() as u64;
                        let infer_stats = ctx.shared::<InferStats>();
                        let det_before = infer_stats.as_ref().map(|s| s.detections()).unwrap_or(0);
                        // Pure transform: decode tensors, attach
                        // detections, tick `InferStats`.  No I/O
                        // inside the processor.
                        let sealed = process_infer_output(
                            inf,
                            converter.as_ref(),
                            infer_stats.as_deref(),
                            ctx.own_name(),
                        );
                        // Routing decision lives here — the hook
                        // body is the single send site.
                        if let Some(sealed) = sealed {
                            if !router.send(PipelineMsg::Deliveries(sealed)) {
                                log::warn!(
                                    "[{}] result receiver closed; dropping sealed batch",
                                    ctx.own_name()
                                );
                            }
                        }
                        if frame_count > 0 {
                            let det_after =
                                infer_stats.as_ref().map(|s| s.detections()).unwrap_or(0);
                            if let Some(stage) = ctx.shared_as::<StageStats>(ROLE_INFER) {
                                tick_stage(
                                    &stage,
                                    frame_count as usize,
                                    det_after.saturating_sub(det_before) as usize,
                                );
                            }
                        }
                    }
                })
                .build(),
        )
        .build()?;
    sys.register_actor(infer)?;

    // ── NvTracker actor ─────────────────────────────────────────────────
    //
    // Routing style 2 — EXPLICIT PEER BY NAME.  The tracker's
    // downstream is chosen at runtime from CLI flags
    // (`tail_actor_name` above), so we deliberately DO NOT call
    // `.downstream(...)` on the builder.  Instead, the output
    // callback uses `router.send_to(&tail_actor_name, msg)` to
    // resolve the target by name.  The first call pays one registry
    // lookup; every subsequent call is a cached
    // `OperatorSink::send`.  This idiom makes the routing decision
    // visible right at the send site — handy when the destination is
    // a proper dependency of the decision rather than a static
    // config default.
    let tracker_cfg = build_tracker_config(cli.gpu)?;
    let tracker_batch_cb = build_tracker_batch_formation();
    let tracker = NvTracker::builder(tracker_name.clone(), cli.channel_cap)
        .operator_factory(move |_bx, result_cb| {
            // NvDCF comes up in well under a second, so no heartbeat
            // wrapper is warranted here — unlike the nvinfer path,
            // which may spend minutes rebuilding the TensorRT plan.
            deepstream_nvtracker::NvTrackerBatchingOperator::new(
                tracker_cfg,
                tracker_batch_cb,
                result_cb,
            )
            .map_err(|e| anyhow::anyhow!("build NvTrackerBatchingOperator: {e}"))
        })
        // Tracking branch routes explicitly to the tail peer by name
        // (see routing style 2 above).  Source-EOS and operator
        // errors use the template defaults — but we need `SourceEos`
        // to flow to the *same* tail peer we chose at runtime, so we
        // install an explicit `on_source_eos` that routes by name
        // AND calls `ctx.reset_stream(..)` to drop the tracker's
        // per-stream state once a source has ended.  The hook ctx
        // carries the operator handle automatically — no separate
        // `NvTrackerResetHandle` wiring required.
        .results(
            NvTrackerResults::builder()
                .on_tracking({
                    let tail_peer = tail_actor_name.clone();
                    move |tracking, router: &Router<PipelineMsg>, ctx| {
                        let frame_count = tracking.frames().len() as u64;
                        let tracker_stats = ctx.shared::<TrackerStats>();
                        // Pure transform: reconcile tracker updates
                        // onto frame metadata, tick `TrackerStats`.
                        // No I/O inside the processor.
                        let sealed = process_tracker_output(
                            tracking,
                            tracker_stats.as_deref(),
                            ctx.own_name(),
                        );
                        // Routing decision lives here — the tracker
                        // has no default peer, so the hook
                        // explicitly fans out to `tail_peer` by
                        // name.
                        if let Some(sealed) = sealed {
                            let msg = PipelineMsg::Deliveries(sealed);
                            if !router.send_to(&tail_peer, msg).unwrap_or(false) {
                                log::warn!(
                                    "[{}] result receiver closed; dropping sealed batch",
                                    ctx.own_name()
                                );
                            }
                        }
                        if frame_count > 0 {
                            if let Some(stage) = ctx.shared_as::<StageStats>(ROLE_TRACKER) {
                                tick_stage(&stage, frame_count as usize, 0);
                            }
                        }
                    }
                })
                .on_source_eos({
                    let tail_peer = tail_actor_name.clone();
                    move |source_id, router: &Router<PipelineMsg>, ctx| {
                        log::info!(
                            "TrackerOperatorOutput::Eos for source_id={source_id}; propagating"
                        );
                        match ctx.reset_stream(source_id) {
                            Ok(()) => {
                                log::info!("[tracker] reset_stream({source_id}) completed on EOS")
                            }
                            Err(e) => {
                                log::warn!("[tracker] reset_stream({source_id}) failed on EOS: {e}")
                            }
                        }
                        let msg = PipelineMsg::SourceEos {
                            source_id: source_id.to_string(),
                        };
                        if !router.send_to(&tail_peer, msg).unwrap_or(false) {
                            log::warn!("downstream closed; dropping SourceEos({source_id})");
                        }
                        Ok(Flow::Cont)
                    }
                })
                .build(),
        )
        .build()?;
    sys.register_actor(tracker)?;

    // ── Pipeline tail ───────────────────────────────────────────────────
    //
    // Back to routing style 1 — `.downstream(mp4_mux_name)` installs
    // the muxer as the picasso router's default peer, so
    // `router.send(msg)` inside `StatsEncodedSink::call` dispatches
    // there.
    if cli.picasso_enabled {
        let draw_enabled = cli.draw_enabled;
        let picasso = Picasso::builder(picasso_name.clone(), cli.channel_cap)
            .downstream(mp4_mux_name.clone())
            .engine_factory(move |bx, router| {
                // Engine-factory snapshot: captures the full hook
                // context so the `OnEncodedFrame` sink can pull
                // per-frame counters out of the shared store on
                // every emit.  No per-closure `Arc` bookkeeping.
                let on_encoded = StatsEncodedSink {
                    router,
                    hook_ctx: bx.hook_ctx(),
                };
                let callbacks = Callbacks::builder().on_encoded_frame(on_encoded).build();
                let general = GeneralSpec::builder()
                    .name("cars-demo")
                    .idle_timeout_secs(600)
                    .inflight_queue_size(8)
                    .build();
                Ok(Arc::new(PicassoEngine::new(general, callbacks)))
            })
            .source_spec_factory(move |_sid, w, h, fn_, fd| {
                build_source_spec(w, h, fn_, fd, draw_enabled)
            })
            .inbox(
                PicassoInbox::builder()
                    .on_delivery({
                        // User-space per-delivery counter — the
                        // framework no longer injects one.  An
                        // `AtomicU64` lets us treat the overlay id
                        // as a stable monotonically-increasing
                        // value across closure reinvocations.
                        let frame_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
                        move |pairs, _ctx| {
                            if draw_enabled {
                                for (frame, _buf) in pairs {
                                    let cnt = frame_counter.fetch_add(1, Ordering::Relaxed);
                                    if let Err(e) = attach_frame_id_overlay(frame, cnt) {
                                        log::warn!("attach_frame_id_overlay failed: {e}");
                                    }
                                }
                            }
                            Ok(())
                        }
                    })
                    .build(),
            )
            .build()?;
        sys.register_actor(picasso)?;

        if cli.output_is_null {
            // `--output null`: Picasso's encoder callback still
            // produces `EncodedMsg::Frame` deliveries, but instead
            // of muxing them into an MP4 we route them into a
            // `BitstreamFunction` terminus that simply drops
            // them.  The BitstreamFunction is registered under
            // `mp4_mux_name` so picasso's already-installed
            // default downstream peer resolves to it without any
            // change in the picasso block above.
            //
            // IMPORTANT — stats accounting:
            //
            // In **every** picasso-enabled configuration
            // (`--output <file>` *and* `--output null`) the
            // canonical per-encoded-frame tick site is
            // [`StatsEncodedSink::call`] on picasso's encoder
            // thread: it increments the tail [`StageStats`], the
            // core [`Stats`], and `PipelineStats.encoded_bytes`
            // exactly once before routing the
            // [`EncodedMsg::Frame`] onward.
            //
            // Because of that, the `on_frame` hook on the
            // `BitstreamFunction` terminus **must not** tick
            // stats — doing so would double-count every encoded
            // frame (once on the encoder thread, once on the
            // bitstream-terminus thread) and report 2× the real
            // FPS / bytes.  The symptom was observed during
            // bring-up (250 FPS instead of the ~125 FPS measured
            // on the Mp4Muxer path).  We therefore leave
            // `on_frame` / `on_packet` / `on_stream_info` at the
            // default drop-on-receive no-op and only override
            // `on_source_eos` to give the terminus the
            // loop-exit semantics the supervisor needs.
            let bitstream_terminus =
                BitstreamFunction::builder(mp4_mux_name.clone(), cli.channel_cap)
                    .inbox(
                        BitstreamFunctionInbox::builder()
                            // Terminus semantics: the first
                            // `SourceEos` means the picasso
                            // encoder has drained, so exit the
                            // loop and let
                            // `cars_shutdown_handler`
                            // broadcast Shutdown to upstream
                            // stages.
                            .on_source_eos(|source_id, _router, ctx| {
                                log::info!(
                                    "[{}] SourceEos {source_id}: bitstream terminus exiting",
                                    ctx.own_name()
                                );
                                Ok(Flow::Stop)
                            })
                            .build(),
                    )
                    .build();
            sys.register_actor(bitstream_terminus)?;
        } else {
            let output_path = cli
                .output
                .as_ref()
                .ok_or_else(|| anyhow!("picasso enabled but output path is missing"))?
                .to_string_lossy()
                .into_owned();
            let muxer = Mp4Muxer::builder(mp4_mux_name.clone(), cli.channel_cap)
                .output(output_path)
                .framerate(cli.fps_num, cli.fps_den)
                .build()?;
            sys.register_actor(muxer)?;
        }
    } else {
        let function = Function::builder(function_name.clone(), cli.channel_cap)
            .inbox(
                FunctionInbox::builder()
                    .on_delivery(|pairs, ctx| {
                        let tail = ctx.shared_as::<StageStats>(ROLE_TAIL);
                        let core = ctx.shared::<Stats>();
                        for _ in 0..pairs.len() {
                            if let Some(stage) = tail.as_deref() {
                                tick_stage(stage, 1, 0);
                            }
                            if let Some(core) = core.as_deref() {
                                core.register_frame(0);
                            }
                        }
                        Ok(())
                    })
                    // Terminus semantics: the Function is the last
                    // stage in the --no-picasso path, so the first
                    // `SourceEos` means upstream has drained — exit
                    // the loop so the supervisor's
                    // `cars_shutdown_handler` can broadcast Shutdown
                    // to everyone else.
                    .on_source_eos(|source_id, ctx| {
                        log::info!(
                            "[{}] SourceEos {source_id}: terminus exiting",
                            ctx.own_name()
                        );
                        Ok(Flow::Stop)
                    })
                    .build(),
            )
            .build();
        sys.register_actor(function)?;
    }

    // ── Kick off FPS reporter + run ─────────────────────────────────────
    core_stats.kick_off();
    let pipeline_t0 = Instant::now();

    let report = sys.run()?;

    // ── Summary ─────────────────────────────────────────────────────────
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
        tracker_stats.unique_tracks(),
        tracker_stats.unmatched_updates(),
        tail_stage_name,
        tail_frames,
        stats.encoded_bytes.load(Ordering::Relaxed),
    );
    log::debug!(
        "final InferStats: frames={} detections={}",
        infer_stats.frames(),
        infer_stats.detections()
    );
    drop(core_stats);

    report.into_result()
}

/// Custom [`ShutdownHandler`](savant_perception::shutdown::ShutdownHandler)
/// policy for the `cars_tracking` sample.
///
/// Pipeline shape: `Mp4Demuxer → Decoder → NvInfer → NvTracker →
/// (Picasso → Mp4Muxer | Function terminus)`.  The demuxer finishing
/// its input file is an **expected** event — it emits an in-band
/// `SourceEos` sentinel that propagates through every stage to the
/// terminus (Mp4Muxer or Function).  The terminus then
/// self-terminates on that EOS (`Flow::Stop`), which is the real
/// "pipeline drained" signal.
///
/// The policy therefore:
///
/// * Ignores `StageExit { stage.kind == Mp4Demux }` — the demuxer
///   is expected to exit first; wait for the terminus.
/// * On any other `StageExit` (terminus drained, mid-pipeline
///   error, or a stage panic) broadcasts a `Shutdown { grace:
///   None }` immediately — no quiescence pause is needed because
///   the terminus only exits *after* upstream has drained.
/// * On `CtrlC`, broadcasts immediately — the user asked to stop.
fn cars_shutdown_handler(
    cause: ShutdownCause,
    _ctx: &mut ShutdownCtx<'_>,
) -> Result<ShutdownAction> {
    match cause {
        ShutdownCause::StageExit { stage } if stage.kind == StageKind::Mp4Demux => {
            log::info!(
                "[supervisor] {stage} exited naturally (end-of-input); \
                 waiting for terminus to drain before broadcasting Shutdown"
            );
            Ok(ShutdownAction::Ignore)
        }
        ShutdownCause::StageExit { stage } => {
            log::info!("[supervisor] {stage} exited; broadcasting Shutdown");
            Ok(ShutdownAction::Broadcast {
                grace: None,
                reason: std::borrow::Cow::Owned(format!("{stage} exited")),
            })
        }
        ShutdownCause::CtrlC => {
            log::warn!("[supervisor] Ctrl+C received; broadcasting Shutdown");
            Ok(ShutdownAction::Broadcast {
                grace: None,
                reason: std::borrow::Cow::Borrowed("ctrl-c"),
            })
        }
    }
}

/// `OnEncodedFrame` implementation used when Picasso is active.
/// Wraps a [`Router<EncodedMsg>`] and a [`HookCtx`]; every
/// per-frame counter (tail [`StageStats`], core [`Stats`],
/// encoded-byte tally) is pulled from the framework's
/// [`SharedStore`] on the fly — the sink holds no bespoke
/// counter references.
///
/// Mirrors the legacy [`picasso::EncodedSink`] but uses the framework
/// [`Router`](savant_perception::router::Router) for forwarding so the
/// default-peer + cached-name-routing contract is enforced in one
/// place, and the [`HookCtx`](savant_perception::HookCtx) for shared
/// state so the sample's hook bodies demonstrate the intended
/// pull-from-store pattern end-to-end.
struct StatsEncodedSink {
    router: Router<EncodedMsg>,
    hook_ctx: HookCtx,
}

impl OnEncodedFrame for StatsEncodedSink {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                let bytes = match frame.get_content().as_ref() {
                    VideoFrameContent::Internal(d) => d.len(),
                    other => {
                        log::error!(
                            "[{}/encode-cb] unexpected content variant: {:?}",
                            self.hook_ctx.own_name(),
                            std::mem::discriminant(other)
                        );
                        return;
                    }
                };
                if let Some(stage) = self.hook_ctx.shared_as::<StageStats>(ROLE_TAIL) {
                    tick_stage(&stage, 1, 0);
                }
                if let Some(core) = self.hook_ctx.shared::<Stats>() {
                    core.register_frame(0);
                }
                if let Some(stats) = self.hook_ctx.shared::<PipelineStats>() {
                    stats
                        .encoded_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                }
                log::debug!(
                    "[{}/encode-cb] frame source_id={} pts={} bytes={bytes}",
                    self.hook_ctx.own_name(),
                    frame.get_source_id(),
                    frame.get_pts(),
                );
                let _ = self.router.send(EncodedMsg::Frame {
                    frame,
                    payload: None,
                });
            }
            OutputMessage::EndOfStream(eos) => {
                log::info!(
                    "[{}/encode-cb] EndOfStream source_id={}",
                    self.hook_ctx.own_name(),
                    eos.source_id
                );
                let _ = self.router.send(EncodedMsg::SourceEos {
                    source_id: eos.source_id,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Demuxer natural exit must not trigger a pipeline shutdown —
    /// the terminus is still draining in-band.
    #[test]
    fn shutdown_handler_ignores_mp4_demux_stage_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::Mp4Demux),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        assert!(matches!(action, ShutdownAction::Ignore));
    }

    /// Mp4Muxer (picasso path terminus) draining triggers an
    /// immediate broadcast with no grace.
    #[test]
    fn shutdown_handler_broadcasts_on_mp4_mux_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::Mp4Mux),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        match action {
            ShutdownAction::Broadcast { grace, reason } => {
                assert_eq!(grace, None, "no quiescence pause needed on terminus exit");
                assert!(reason.contains("mp4_mux"), "unexpected reason: {reason}");
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }

    /// Function (--no-picasso terminus) draining triggers an
    /// immediate broadcast with no grace.
    #[test]
    fn shutdown_handler_broadcasts_on_function_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::Function),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        match action {
            ShutdownAction::Broadcast { grace, reason } => {
                assert_eq!(grace, None);
                assert!(reason.contains("function"), "unexpected reason: {reason}");
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }

    /// A mid-pipeline stage exiting (e.g. a panic or fatal error
    /// in the tracker) also triggers a broadcast — the demuxer is
    /// the *only* kind whose natural exit is ignored.
    #[test]
    fn shutdown_handler_broadcasts_on_midpipeline_stage_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::Tracker),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        assert!(matches!(
            action,
            ShutdownAction::Broadcast { grace: None, .. }
        ));
    }

    /// Ctrl+C immediately broadcasts Shutdown — no quiescence sleep.
    #[test]
    fn shutdown_handler_broadcasts_on_ctrlc() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let action = cars_shutdown_handler(ShutdownCause::CtrlC, &mut ctx).unwrap();
        match action {
            ShutdownAction::Broadcast { grace, reason } => {
                assert_eq!(grace, None);
                assert_eq!(reason, "ctrl-c");
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }

    /// Integration guard: the orchestrator must fail fast when the
    /// input file is missing, before touching GStreamer / CUDA.
    #[test]
    fn run_rejects_missing_input() {
        let cli = ResolvedCli {
            input: PathBuf::from("/definitely/does/not/exist.mp4"),
            output: Some(std::env::temp_dir().join("savant_perception_pipeline_out.mp4")),
            gpu: 0,
            conf: 0.25,
            iou: 0.45,
            channel_cap: 4,
            debug: false,
            fps_num: 25,
            fps_den: 1,
            draw_enabled: true,
            picasso_enabled: true,
            output_is_null: false,
        };
        let err = run(cli).unwrap_err();
        assert!(
            err.to_string().contains("input file does not exist"),
            "unexpected error: {err:#}"
        );
    }
}
