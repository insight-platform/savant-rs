//! End-to-end streaming orchestrator for the `cars_demo` sample.
//!
//! ```text
//! Mp4Demuxer | UriDemuxer -> FlexibleDecoderPool -> NvInfer -> NvTracker -> Picasso -> Mp4Muxer
//! ```
//!
//! The demuxer actor is selected from the resolved CLI input: a
//! filesystem path uses [`Mp4DemuxerSource`] (bit-identical legacy
//! behaviour); a URI (`file://`, `http://`, `rtsp://`, `rtmp://`,
//! `hls://`, …) uses [`UriDemuxerSource`].
//!
//! This module is **orchestration only** — per-actor behaviour lives
//! inside the framework Layer-B stages
//! ([`savant_perception::stages`]) plus a thin set of per-stage
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

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use ::picasso::prelude::OnEncodedFrame;
use ::picasso::{Callbacks, GeneralSpec, OutputMessage, PicassoEngine};
use anyhow::{anyhow, bail, Context, Result};
use deepstream_buffers::cuda_init;
use savant_core::pipeline::stats::{StageStats, Stats};
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::transport::zeromq::{ReaderConfig, WriterConfig};

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
use crate::cars_demo::stats::{make_stage, stage_frames, stage_objects, tick_stage, PipelineStats};
use crate::cars_demo::warmup::prepare_nvinfer_operator;
use crate::cli::{InputSource, ResolvedCli};
use savant_perception::envelopes::{EncodedMsg, PipelineMsg};
use savant_perception::router::Router;
use savant_perception::shutdown::{ShutdownAction, ShutdownCause, ShutdownCtx};
use savant_perception::stages::decoder::make_decode_frame;
use savant_perception::stages::{
    BitstreamFunction, BitstreamFunctionInbox, Decoder, DecoderResults, Function, FunctionInbox,
    Mp4DemuxerResults, Mp4DemuxerSource, Mp4Muxer, NvInfer, NvInferResults, NvTracker,
    NvTrackerResults, PayloadCarrier, Picasso, PicassoInbox, UriDemuxerResults, UriDemuxerSource,
    ZmqSink, ZmqSource,
};
use savant_perception::supervisor::{StageKind, StageName};
use savant_perception::{Flow, HookCtx, System};

/// Default source identifier used throughout the pipeline.  Every
/// stage uses the same id because the sample processes exactly one
/// input file.
const SOURCE_ID: &str = "cars-demo";
/// Default period (seconds) at which [`Stats`] logs timestamp-based
/// FPS while the pipeline runs.  Used when the CLI does not override
/// it via `--stats-period`.  Thirty seconds keeps the example logs
/// readable when the pipeline idles between input streams; the final
/// stats are always emitted on shutdown regardless of the period.
pub const DEFAULT_STATS_PERIOD_SECS: u64 = 30;
/// Retained history length for [`Stats`].
const STATS_HISTORY_LEN: usize = 100;

/// Pipeline head — the input source.
///
/// Selected by the binary at startup.  `cars-demo` selects
/// [`PipelineHead::Mp4Demux`] / [`PipelineHead::UriDemux`] from the
/// CLI; `cars-demo-zmq pipeline` selects [`PipelineHead::Zmq`].
pub enum PipelineHead {
    /// MP4 demuxer driven by `qtdemux` (filesystem path input).
    Mp4Demux {
        /// Filesystem path passed verbatim to the demuxer.
        input: PathBuf,
        /// Source id stamped on every emitted frame envelope.
        source_id: String,
    },
    /// Generic URI demuxer driven by `urisourcebin` + `parsebin`.
    UriDemux {
        /// URI string passed verbatim to the demuxer.
        input: String,
        /// Source id stamped on every emitted frame envelope.
        source_id: String,
    },
    /// ZeroMQ ingress (encoded frames + EOS arrive on the wire).
    ///
    /// ZMQ is always multi-stream: wire EOS is forwarded as
    /// `EncodedMsg::SourceEos`, but the source stays subscribed.
    Zmq {
        /// Reader config (URL + topic prefix + sockopts).
        config: Box<ReaderConfig>,
    },
}

/// Pipeline tail — the terminus actor.
///
/// Selected by the binary at startup.  `cars-demo` picks one of the
/// existing terminus stages from the CLI; `cars-demo-zmq pipeline`
/// picks [`PipelineTail::Zmq`]; `cars-demo-zmq consumer` picks
/// [`PipelineTail::Mp4Mux`] without going through the middle stages
/// (consumer is a different binary entry point that doesn't use this
/// `run` function).
pub enum PipelineTail {
    /// MP4 muxer terminus (used by `cars-demo` / `cars-demo-zmq
    /// consumer`).
    Mp4Mux {
        /// Output `.mp4` path; parent must exist.
        output: PathBuf,
    },
    /// `--output null` sentinel terminus that drops the encoded
    /// bitstream while still measuring throughput.  Requires
    /// `picasso_enabled = true`.
    Bitstream,
    /// `--no-picasso` terminus.  No transform / overlay / encode /
    /// mux; the tail just counts the inference / tracker output.
    /// Requires `picasso_enabled = false`.
    Function,
    /// ZeroMQ egress terminus (used by `cars-demo-zmq pipeline` and
    /// `cars-demo-zmq producer`, but the producer is a different
    /// binary entry point that doesn't use this `run` function).
    /// ZeroMQ egress (encoded frames + EOS go out on the wire).
    ///
    /// ZMQ is always multi-stream: per-stream EOS is forwarded over
    /// the wire, but the actor keeps running.
    Zmq {
        /// Writer config (URL + sockopts).
        config: Box<WriterConfig>,
        /// Carrier policy for the encoded bitstream — multipart vs
        /// embedded inside the protobuf [`Message`](savant_core::message::Message).
        payload_carrier: PayloadCarrier,
    },
}

/// Numeric / behavioural pipeline knobs shared across head and tail.
///
/// `cars-demo` builds these from [`ResolvedCli`].  `cars-demo-zmq
/// pipeline` builds them with `picasso_enabled = true`,
/// `draw_enabled = true`, and CLI-driven thresholds.
pub struct PipelineKnobs {
    /// CUDA device id.
    pub gpu: u32,
    /// Detection confidence threshold.
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Inter-stage channel capacity.
    pub channel_cap: usize,
    /// Output container framerate numerator (used by
    /// [`PipelineTail::Mp4Mux`] only).
    pub fps_num: i32,
    /// Output container framerate denominator.
    pub fps_den: i32,
    /// Whether Picasso draw stage runs.
    pub draw_enabled: bool,
    /// Whether Picasso is in the pipeline at all.
    pub picasso_enabled: bool,
    /// Verbose per-frame logging.
    pub debug: bool,
    /// Period (milliseconds) between time-based stats reports while
    /// the pipeline runs.  A final report is always emitted on
    /// shutdown regardless of the period; this knob only controls
    /// the cadence of intermediate samples.
    pub stats_period_ms: i64,
}

/// Supervisor shutdown policy choice.
///
/// * [`ShutdownPolicy::TerminusBroadcast`] — `cars-demo` legacy
///   policy: ignore demuxer-source natural exits, broadcast on
///   any other stage exit (terminus drained or fatal mid-pipeline).
///   Used by `cars-demo`, `cars-demo-zmq consumer`, and
///   `cars-demo-zmq producer`.
/// * [`ShutdownPolicy::CtrlCOnly`] — `cars-demo-zmq pipeline`
///   policy: ignore *all* clean stage exits, only broadcast on
///   `Ctrl+C`.  Mid-pipeline panics still bubble up via
///   [`StageExit`](ShutdownCause::StageExit) where they get logged
///   and ignored — the supervisor's `--abort-on-stage-error` flag
///   is the configured escape hatch for those cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownPolicy {
    /// Default: terminus exit triggers broadcast.
    TerminusBroadcast,
    /// Only Ctrl+C triggers shutdown.
    CtrlCOnly,
}

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
const ROLE_PICASSO: &str = "picasso";

/// Run the cars-tracking pipeline end-to-end (legacy `cars-demo`
/// entry point).  Blocks until the input MP4 is fully processed and
/// the output MP4 is finalized.
///
/// Thin wrapper around [`run_pipeline`] that maps the [`ResolvedCli`]
/// onto the head/tail enums and the legacy
/// [`ShutdownPolicy::TerminusBroadcast`] policy.
pub fn run(cli: ResolvedCli) -> Result<()> {
    // Path inputs still need the existence / file-type guard here
    // (the CLI validates existence, but re-checking at pipeline
    // entry catches unit-test ResolvedCli literals and late
    // filesystem changes).  URI inputs are opaque — any addressing
    // error surfaces when the URI demuxer actually starts.
    match &cli.input {
        InputSource::Path(p) => {
            if !p.is_file() {
                bail!("input file does not exist: {}", p.display());
            }
        }
        InputSource::Uri(_) => {}
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

    let head = match cli.input.clone() {
        InputSource::Path(input) => PipelineHead::Mp4Demux {
            input,
            source_id: SOURCE_ID.to_string(),
        },
        InputSource::Uri(input) => PipelineHead::UriDemux {
            input,
            source_id: SOURCE_ID.to_string(),
        },
    };
    let tail = match (cli.picasso_enabled, cli.output_is_null) {
        (true, true) => PipelineTail::Bitstream,
        (true, false) => PipelineTail::Mp4Mux {
            output: cli
                .output
                .clone()
                .ok_or_else(|| anyhow!("picasso enabled but output path is missing"))?,
        },
        (false, _) => PipelineTail::Function,
    };
    let knobs = PipelineKnobs {
        gpu: cli.gpu,
        conf: cli.conf,
        iou: cli.iou,
        channel_cap: cli.channel_cap,
        fps_num: cli.fps_num,
        fps_den: cli.fps_den,
        draw_enabled: cli.draw_enabled,
        picasso_enabled: cli.picasso_enabled,
        debug: cli.debug,
        stats_period_ms: cli.stats_period_ms,
    };
    run_pipeline(head, tail, knobs, ShutdownPolicy::TerminusBroadcast)
}

/// Run the cars-tracking pipeline with explicit head/tail/policy.
///
/// This is the actual orchestrator.  [`run`] is a thin wrapper for
/// the legacy `cars-demo` CLI; the `cars-demo-zmq pipeline`
/// subcommand calls this directly with [`PipelineHead::Zmq`] /
/// [`PipelineTail::Zmq`] and [`ShutdownPolicy::CtrlCOnly`].
pub fn run_pipeline(
    head: PipelineHead,
    tail: PipelineTail,
    knobs: PipelineKnobs,
    shutdown_policy: ShutdownPolicy,
) -> Result<()> {
    if knobs.picasso_enabled {
        if matches!(tail, PipelineTail::Function) {
            bail!("PipelineTail::Function requires picasso_enabled = false");
        }
    } else if matches!(
        tail,
        PipelineTail::Mp4Mux { .. } | PipelineTail::Bitstream | PipelineTail::Zmq { .. }
    ) {
        bail!("encoder-bearing tail requires picasso_enabled = true");
    }

    gstreamer::init().context("gstreamer init")?;
    cuda_init(knobs.gpu).map_err(|e| anyhow!("cuda init (gpu={}): {e}", knobs.gpu))?;

    // ── Diagnostic CUDA-poison probe ────────────────────────────────────
    //
    // Spawn a background thread that calls `cudaDeviceSynchronize` once
    // per second.  Whenever the rc transitions from healthy → poisoned
    // we log a one-shot warning with a wall-clock timestamp so we can
    // pinpoint the *exact* moment the device flips during idle.  Once
    // poisoned, the thread keeps probing every 5 s just to confirm the
    // sticky-state semantics; never logs again until cleared.
    //
    // Opt-in via `SAVANT_DIAG_DEVICE_PROBE=1` to keep production runs
    // free of background CUDA traffic.
    if std::env::var("SAVANT_DIAG_DEVICE_PROBE").as_deref() == Ok("1") {
        std::thread::Builder::new()
            .name("savant-diag-device-probe".into())
            .spawn(|| {
                let mut last_rc: i32 = 0;
                let mut tick: u64 = 0;
                loop {
                    let rc = deepstream_buffers::cuda_device::cuda_device_synchronize(
                        "diag::device_probe",
                        format_args!("tick={tick}"),
                    );
                    if rc != last_rc {
                        log::error!(
                            "DIAG-PROBE: cudaDeviceSynchronize rc transitioned {last_rc} \
                             → {rc} at tick={tick}"
                        );
                        last_rc = rc;
                    }
                    tick += 1;
                    let nap_ms = if rc == 0 { 10 } else { 5_000 };
                    std::thread::sleep(std::time::Duration::from_millis(nap_ms));
                }
            })
            .context("spawn device-probe thread")?;
        log::info!("DIAG-PROBE: device-sync probe enabled (1 Hz when healthy)");
    }

    log::info!(
        "cars-demo starting: head={} tail={} gpu={} conf={} iou={} channel_cap={} fps={}/{} picasso_enabled={} draw_enabled={} debug={} stats_period_ms={} policy={:?}",
        head_label(&head),
        tail_label(&tail),
        knobs.gpu,
        knobs.conf,
        knobs.iou,
        knobs.channel_cap,
        knobs.fps_num,
        knobs.fps_den,
        knobs.picasso_enabled,
        knobs.draw_enabled,
        knobs.debug,
        knobs.stats_period_ms,
        shutdown_policy,
    );

    // ── Shared state ────────────────────────────────────────────────────
    let stats = PipelineStats::new();
    let core_stats = Arc::new(Stats::new(
        STATS_HISTORY_LEN,
        None,
        Some(knobs.stats_period_ms),
    ));
    let decoder_stage = make_stage("decoder");
    let infer_stage = make_stage("infer");
    let track_stage = make_stage("track");
    let tail_stage_name = match &tail {
        PipelineTail::Mp4Mux { .. } => "encode",
        PipelineTail::Bitstream => "bitstream",
        PipelineTail::Function => "function",
        PipelineTail::Zmq { .. } => "zmq_sink",
    };
    let tail_stage = make_stage(tail_stage_name);
    // Picasso lives between the tracker and the tail when it is
    // enabled; we register a dedicated `picasso` `StageStats` so the
    // 📊 line shows whether the encoder is actually receiving frames.
    // The tick site is the `PicassoInbox::on_delivery` hook below — we
    // intentionally do **not** also bump `core.register_frame(...)`
    // there, that aggregator is owned by the tail.
    let picasso_stage = knobs.picasso_enabled.then(|| make_stage("picasso"));
    core_stats.add_stage_stats(decoder_stage.clone());
    core_stats.add_stage_stats(infer_stage.clone());
    core_stats.add_stage_stats(track_stage.clone());
    if let Some(s) = picasso_stage.as_ref() {
        core_stats.add_stage_stats(s.clone());
    }
    core_stats.add_stage_stats(tail_stage.clone());

    let infer_stats = Arc::new(InferStats::new());
    let tracker_stats = Arc::new(TrackerStats::new());

    // ── Stage identities ────────────────────────────────────────────────
    //
    // The source's stage-kind tracks the resolved input variant so
    // logs and supervisor policy can disambiguate `mp4_demux` /
    // `uri_demux` / `zmq_source` pipelines.
    let head_name = match &head {
        PipelineHead::Mp4Demux { .. } => StageName::unnamed(StageKind::Mp4Demux),
        PipelineHead::UriDemux { .. } => StageName::unnamed(StageKind::UriDemux),
        PipelineHead::Zmq { .. } => StageName::unnamed(StageKind::ZmqSource),
    };
    let decoder_name = StageName::unnamed(StageKind::Decoder);
    let infer_name = StageName::unnamed(StageKind::Infer);
    let tracker_name = StageName::unnamed(StageKind::Tracker);
    let picasso_name = StageName::unnamed(StageKind::Picasso);
    // Tail-actor name is selected from the chosen tail variant so
    // multiple tail kinds can coexist in the supervisor's
    // role-routing tables without name clashes.
    let tail_name = match &tail {
        PipelineTail::Mp4Mux { .. } | PipelineTail::Bitstream => {
            StageName::unnamed(StageKind::Mp4Mux)
        }
        PipelineTail::Function => StageName::unnamed(StageKind::Function),
        PipelineTail::Zmq { .. } => StageName::unnamed(StageKind::ZmqSink),
    };
    let function_name = StageName::unnamed(StageKind::Function);

    // Pick the logical tail stage downstream of the tracker.  The
    // chosen name is used both as the tracker's downstream target
    // and in the final summary label.
    let tail_actor_name = if knobs.picasso_enabled {
        picasso_name.clone()
    } else {
        function_name.clone()
    };

    // ── Build System ────────────────────────────────────────────────────
    //
    // The shutdown policy is parametric so the same orchestrator
    // serves both `cars-demo` (terminus exit broadcasts) and
    // `cars-demo-zmq pipeline` (only Ctrl+C broadcasts; the
    // pipeline runs forever otherwise).
    let mut sys = System::new().on_shutdown(match shutdown_policy {
        ShutdownPolicy::TerminusBroadcast => cars_shutdown_handler,
        ShutdownPolicy::CtrlCOnly => ctrlc_only_shutdown_handler,
    });

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
    if let Some(s) = picasso_stage.as_ref() {
        sys.insert_shared_as::<StageStats>(ROLE_PICASSO, s.clone());
    }

    // ── Source / head ───────────────────────────────────────────────────
    //
    // Three head variants are supported:
    //
    // * [`PipelineHead::Mp4Demux`] / [`PipelineHead::UriDemux`] —
    //   build the [`VideoFrame`] on the demuxer side via
    //   `make_decode_frame` + [`EncodedMsg::Frame`] (carrier =
    //   multipart payload).  The demuxer stage doesn't call
    //   `router.send` itself — every downstream dispatch happens
    //   inside the variant hooks below.
    // * [`PipelineHead::Zmq`] — defer to the
    //   [`ZmqSource`](savant_perception::stages::ZmqSource)
    //   stage's default forwarder; the wire `Message::video_frame`
    //   already carries everything the downstream `Decoder` needs and
    //   the multipart-first carrier resolution lives inside
    //   [`ZmqSource::default_on_message`].
    match head {
        PipelineHead::Mp4Demux { input, source_id } => {
            let demux_src = Mp4DemuxerSource::builder(head_name.clone())
                .one_shot(input.to_string_lossy().into_owned(), source_id)
                .downstream(decoder_name.clone())
                .results(
                    Mp4DemuxerResults::builder()
                        .on_packet(|_input, source_id, info, pkt, router, ctx| {
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
        }
        PipelineHead::UriDemux { input, source_id } => {
            // `UriDemuxerSource` drives `urisourcebin` + `parsebin`
            // under the hood and is suitable for file://, http://,
            // rtsp://, rtmp://, hls://, … URIs.  Bin/source property
            // maps are empty: defaults work for file:// and most
            // public RTSP/HTTP sources.
            let demux_src = UriDemuxerSource::builder(head_name.clone())
                .one_shot(input, source_id)
                .downstream(decoder_name.clone())
                .results(
                    UriDemuxerResults::builder()
                        .on_packet(|_uri, source_id, info, pkt, router, ctx| {
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
        }
        PipelineHead::Zmq { config } => {
            // ZMQ is always multi-stream: defaults forward
            // per-stream EOS but never terminate the source.
            let zmq_src = ZmqSource::builder(head_name.clone())
                .config(*config)
                .downstream(decoder_name.clone())
                .build()?;
            sys.register_source(zmq_src)?;
        }
    }

    // ── Decoder actor ───────────────────────────────────────────────────
    //
    // The `Decoder` stage never sends: every variant hook
    // below is the user-facing send site.  Because every hook receives
    // `&Router<PipelineMsg>`, source-id-based routing (e.g.
    // `router.send_to(&engine_for(&sealed), PipelineMsg::Delivery(sealed))`
    // for fan-out to multiple inference engines) would plug in here.
    //
    // Only `on_frame` needs sample-specific code (forward + stage
    // tick).  `on_source_eos`, `on_parameter_change`, `on_skipped`,
    // `on_orphan_frame`, `on_decoder_event` and `on_decoder_error`
    // fall through to the stage's `Decoder::default_on_*`
    // forwarders/loggers, installed automatically by the builder
    // when their setters are omitted.
    let decoder = Decoder::builder(decoder_name.clone(), knobs.channel_cap)
        .downstream(infer_name.clone())
        .gpu_id(knobs.gpu)
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
    let infer_cfg = build_nvinfer_config(knobs.gpu)?;
    let infer_batch_cb = build_infer_batch_formation();
    let converter = build_yolo_converter(knobs.conf, knobs.iou);
    let gpu = knobs.gpu;
    let infer = NvInfer::builder(infer_name.clone(), knobs.channel_cap)
        .downstream(tracker_name.clone())
        .operator_factory(move |_bx, result_cb| {
            let (op, _elapsed) =
                prepare_nvinfer_operator(gpu, infer_cfg, infer_batch_cb, result_cb)?;
            Ok(op)
        })
        // Only the Inference branch needs sample-specific code — YOLO
        // decode + stats tick.  Source-EOS propagation and operator-
        // error logging fall through to the stage defaults
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
    let tracker_cfg = build_tracker_config(knobs.gpu)?;
    let tracker_batch_cb = build_tracker_batch_formation();
    let tracker = NvTracker::builder(tracker_name.clone(), knobs.channel_cap)
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
        // errors use the stage defaults — but we need `SourceEos`
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
    // Back to routing style 1 — `.downstream(tail_name)` installs the
    // chosen tail actor as the picasso router's default peer, so
    // `router.send(msg)` inside `StatsEncodedSink::call` dispatches
    // there regardless of whether the tail is `Mp4Muxer`,
    // `BitstreamFunction`, or `ZmqSink`.
    if knobs.picasso_enabled {
        let draw_enabled = knobs.draw_enabled;
        let picasso = Picasso::builder(picasso_name.clone(), knobs.channel_cap)
            .downstream(tail_name.clone())
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
                        move |pairs, ctx| {
                            if draw_enabled {
                                for (frame, _buf) in pairs {
                                    let cnt = frame_counter.fetch_add(1, Ordering::Relaxed);
                                    if let Err(e) = attach_frame_id_overlay(frame, cnt) {
                                        log::warn!("attach_frame_id_overlay failed: {e}");
                                    }
                                }
                            }
                            // Per-stage 📊 counter — bump once per frame
                            // delivered into Picasso (i.e. per pair),
                            // matching how decoder/infer/track tick on
                            // their own egress hooks.  The tail stage's
                            // tick happens later inside
                            // `StatsEncodedSink::call`, so picasso ≠
                            // tail in the stats lines (frames in =
                            // picasso, frames out = tail).
                            if !pairs.is_empty() {
                                if let Some(stage) = ctx.shared_as::<StageStats>(ROLE_PICASSO) {
                                    tick_stage(&stage, pairs.len(), 0);
                                }
                            }
                            Ok(())
                        }
                    })
                    .build(),
            )
            .build()?;
        sys.register_actor(picasso)?;

        match tail {
            PipelineTail::Bitstream => {
                // `--output null`: Picasso's encoder callback still
                // produces `EncodedMsg::Frame` deliveries, but instead
                // of muxing them into an MP4 we route them into a
                // `BitstreamFunction` terminus that simply drops
                // them.  The BitstreamFunction is registered under
                // `tail_name` so picasso's already-installed default
                // downstream peer resolves to it without any change in
                // the picasso block above.
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
                // EOS semantics depend on the supervisor's shutdown
                // policy:
                //
                // * `TerminusBroadcast` (the legacy `cars-demo
                //   --output null` path) treats the terminus exit
                //   as the broadcast trigger — return `Flow::Stop`
                //   on the first per-source EOS so
                //   `cars_shutdown_handler` brings the rest of the
                //   pipeline down.
                // * `CtrlCOnly` (the `cars-demo-zmq pipeline
                //   --no-sink` path) is multi-stream and runs
                //   forever — return `Flow::Cont` so back-to-back
                //   producer cycles (each ending in their own
                //   `SourceEos`) do not silently kill the
                //   terminus and starve every upstream stage on
                //   the next cycle.
                let exit_on_eos = matches!(shutdown_policy, ShutdownPolicy::TerminusBroadcast);
                let bitstream_terminus =
                    BitstreamFunction::builder(tail_name.clone(), knobs.channel_cap)
                        .inbox(
                            BitstreamFunctionInbox::builder()
                                .on_source_eos(move |source_id, _router, ctx| {
                                    if exit_on_eos {
                                        log::info!(
                                            "[{}] SourceEos {source_id}: bitstream terminus exiting",
                                            ctx.own_name()
                                        );
                                        Ok(Flow::Stop)
                                    } else {
                                        log::info!(
                                            "[{}] SourceEos {source_id}: bitstream terminus continuing (multi-stream)",
                                            ctx.own_name()
                                        );
                                        Ok(Flow::Cont)
                                    }
                                })
                                .build(),
                        )
                        .build();
                sys.register_actor(bitstream_terminus)?;
            }
            PipelineTail::Mp4Mux { output } => {
                let output_path = output.to_string_lossy().into_owned();
                let muxer = Mp4Muxer::builder(tail_name.clone(), knobs.channel_cap)
                    .output(output_path)
                    .framerate(knobs.fps_num, knobs.fps_den)
                    .build()?;
                sys.register_actor(muxer)?;
            }
            PipelineTail::Zmq {
                config,
                payload_carrier,
            } => {
                // ZMQ is always multi-stream: default `on_source_eos`
                // forwards the wire EOS but keeps the sink running.
                let zmq_sink = ZmqSink::builder(tail_name.clone(), knobs.channel_cap)
                    .config(*config)
                    .payload_carrier(payload_carrier)
                    .build()?;
                sys.register_actor(zmq_sink)?;
            }
            PipelineTail::Function => {
                bail!("PipelineTail::Function requires picasso_enabled = false");
            }
        }
    } else {
        // EOS semantics on the Function terminus track the
        // supervisor's shutdown policy, mirroring the Bitstream
        // branch above:
        //
        // * `TerminusBroadcast` (legacy `cars-demo --no-picasso` and
        //   `cars-demo-zmq consumer`) — first per-source EOS means
        //   upstream has drained, exit the loop so the supervisor's
        //   `cars_shutdown_handler` can broadcast Shutdown.
        // * `CtrlCOnly` (`cars-demo-zmq pipeline --no-sink`) — the
        //   pipeline is multi-stream and runs across back-to-back
        //   producer cycles, each ending in its own `SourceEos`.
        //   Returning `Flow::Stop` on the first one would silently
        //   kill the terminus and starve every upstream stage on
        //   the next cycle, so we keep the actor alive on EOS and
        //   let `Ctrl+C` be the only shutdown trigger.
        let exit_on_eos = matches!(shutdown_policy, ShutdownPolicy::TerminusBroadcast);
        let function = Function::builder(function_name.clone(), knobs.channel_cap)
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
                    .on_source_eos(move |source_id, ctx| {
                        if exit_on_eos {
                            log::info!(
                                "[{}] SourceEos {source_id}: terminus exiting",
                                ctx.own_name()
                            );
                            Ok(Flow::Stop)
                        } else {
                            log::info!(
                                "[{}] SourceEos {source_id}: terminus continuing (multi-stream)",
                                ctx.own_name()
                            );
                            Ok(Flow::Cont)
                        }
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
        "cars-demo done: pipeline_runtime={:.2}s avg_fps={:.1} demux_packets={} decoded={} infer_frames={} detections={} track_frames={} unique_tracks={} unmatched_updates={} {}={} bytes={}",
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
/// policy for the `cars_demo` sample.
///
/// Pipeline shape: `(Mp4Demuxer | UriDemuxer) → Decoder → NvInfer →
/// NvTracker → (Picasso → Mp4Muxer | Function terminus)`.  Either
/// demuxer finishing its input is an **expected** event — it emits
/// an in-band `SourceEos` sentinel that propagates through every
/// stage to the terminus (Mp4Muxer or Function).  The terminus
/// then self-terminates on that EOS (`Flow::Stop`), which is the
/// real "pipeline drained" signal.
///
/// The policy therefore:
///
/// * Ignores `StageExit { stage.kind == Mp4Demux | UriDemux }` —
///   the demuxer is expected to exit first; wait for the terminus.
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
        ShutdownCause::StageExit { stage }
            if matches!(
                stage.kind,
                StageKind::Mp4Demux | StageKind::UriDemux | StageKind::ZmqSource
            ) =>
        {
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

/// `Ctrl+C-only` shutdown policy.
///
/// Used by `cars-demo-zmq pipeline`: the pipeline runs forever in
/// principle; both the [`ZmqSource`] head and the [`ZmqSink`] tail
/// are inherently multi-stream so they keep running across wire EOS
/// events.  Therefore, a clean stage exit is **never** expected —
/// only fatal mid-pipeline errors and the user pressing `Ctrl+C`
/// should bring the pipeline down.
///
/// * `StageExit` — log at `info` and ignore.  The supervisor
///   already surfaces fatal stage panics via
///   [`StageReport`](savant_perception::system::StageReport)
///   regardless of the shutdown handler.
/// * `CtrlC` — broadcast immediately with no grace.
fn ctrlc_only_shutdown_handler(
    cause: ShutdownCause,
    _ctx: &mut ShutdownCtx<'_>,
) -> Result<ShutdownAction> {
    match cause {
        ShutdownCause::StageExit { stage } => {
            log::info!(
                "[supervisor] {stage} exited; ignoring (cars-demo-zmq pipeline runs until Ctrl+C)"
            );
            Ok(ShutdownAction::Ignore)
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

/// Short label for [`PipelineHead`] used in startup logs.
fn head_label(head: &PipelineHead) -> String {
    match head {
        PipelineHead::Mp4Demux { input, .. } => format!("mp4://{}", input.display()),
        PipelineHead::UriDemux { input, .. } => format!("uri://{input}"),
        PipelineHead::Zmq { config } => {
            format!("zmq://{}", config.endpoint())
        }
    }
}

/// Short label for [`PipelineTail`] used in startup logs.
fn tail_label(tail: &PipelineTail) -> String {
    match tail {
        PipelineTail::Mp4Mux { output } => format!("mp4://{}", output.display()),
        PipelineTail::Bitstream => "bitstream://drop".to_string(),
        PipelineTail::Function => "function://drop".to_string(),
        PipelineTail::Zmq {
            config,
            payload_carrier,
        } => {
            format!("zmq://{} carrier={payload_carrier:?}", config.endpoint())
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
            input: InputSource::Path(PathBuf::from("/definitely/does/not/exist.mp4")),
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
            stats_period_ms: 30_000,
        };
        let err = run(cli).unwrap_err();
        assert!(
            err.to_string().contains("input file does not exist"),
            "unexpected error: {err:#}"
        );
    }

    /// Natural exit of the URI demuxer stage must be treated the
    /// same as the MP4 demuxer: ignored so the terminus can drain
    /// the in-band `SourceEos` before a broadcast fires.
    #[test]
    fn shutdown_handler_ignores_uri_demux_stage_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::UriDemux),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        assert!(matches!(action, ShutdownAction::Ignore));
    }

    /// `ZmqSource` is a source like `Mp4Demux` / `UriDemux`: its
    /// natural exit (wire EOS for a single source) must not trigger
    /// a broadcast — wait for the terminus to drain in-band first.
    #[test]
    fn shutdown_handler_ignores_zmq_source_stage_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::ZmqSource),
        };
        let action = cars_shutdown_handler(cause, &mut ctx).unwrap();
        assert!(matches!(action, ShutdownAction::Ignore));
    }

    /// `ctrlc_only_shutdown_handler` must ignore any `StageExit`
    /// (including the terminus): the `cars-demo-zmq pipeline`
    /// subcommand only stops on Ctrl+C.
    #[test]
    fn ctrlc_only_handler_ignores_terminus_stage_exit() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let cause = ShutdownCause::StageExit {
            stage: StageName::unnamed(StageKind::ZmqSink),
        };
        let action = ctrlc_only_shutdown_handler(cause, &mut ctx).unwrap();
        assert!(matches!(action, ShutdownAction::Ignore));
    }

    /// `ctrlc_only_shutdown_handler` must broadcast on Ctrl+C —
    /// that is the *only* exit path for the `pipeline` subcommand.
    #[test]
    fn ctrlc_only_handler_broadcasts_on_ctrlc() {
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let action = ctrlc_only_shutdown_handler(ShutdownCause::CtrlC, &mut ctx).unwrap();
        match action {
            ShutdownAction::Broadcast { grace, reason } => {
                assert_eq!(grace, None);
                assert_eq!(reason, "ctrl-c");
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }
}
