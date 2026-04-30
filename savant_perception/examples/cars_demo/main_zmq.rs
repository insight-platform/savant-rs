//! `cars-demo-zmq` example: hybrid ZMQ producer / pipeline / consumer.
//!
//! Three-role demo binary that exercises the full ZMQ stack:
//!
//! ```text
//! producer:  (Mp4Demuxer | UriDemuxer)  -> ZmqSink
//! pipeline:  ZmqSource -> Decoder -> NvInfer -> NvTracker -> Picasso -> ZmqSink
//! consumer:  ZmqSource -> Mp4Muxer
//! ```
//!
//! Recommended end-to-end run (three shells; the default
//! `--zmq-in` / `--zmq-out` values form a compatible IPC chain so
//! the simple case needs no extra arguments):
//!
//! ```bash
//! # shell 1 — pipeline (binds both ends; start first)
//! cargo run -p savant-perception-framework --example cars-demo-zmq -- pipeline
//!
//! # shell 2 — consumer (subscribes to pipeline output)
//! cargo run -p savant-perception-framework --example cars-demo-zmq -- \
//!     consumer --output /tmp/cars_zmq.mp4
//!
//! # shell 3 — producer (pushes the source clip into the pipeline)
//! cargo run -p savant-perception-framework --example cars-demo-zmq -- \
//!     producer --input /path/to/clip.mp4
//! ```
//!
//! The defaults are:
//!
//! * `producer  --zmq-out  dealer+connect:ipc:///tmp/savant_demo_in`
//! * `pipeline  --zmq-in   router+bind:ipc:///tmp/savant_demo_in`
//! * `pipeline  --zmq-out  pub+bind:ipc:///tmp/savant_demo_out`
//! * `consumer  --zmq-in   sub+connect:ipc:///tmp/savant_demo_out`
//!
//! Override any of these flags to bridge over TCP, swap socket types,
//! or run multiple producers against the same pipeline.
//!
//! ## Lifecycle
//!
//! ZMQ stages default to multi-stream semantics — wire EOS is
//! forwarded but never terminates the source/sink on its own.
//! Subcommand-specific exit behaviour is layered on top:
//!
//! * `producer` — exits after sending the last frame + wire EOS.
//!   The demuxer source pushes a final [`EncodedMsg::SourceEos`]
//!   into its downstream channel and exits.  The producer overrides
//!   [`ZmqSink`]'s `on_source_eos` to return `Flow::Stop`, so once
//!   the sink has forwarded the wire EOS (and classified the
//!   writer-result) the actor exits naturally.  The
//!   `producer_consumer_shutdown_handler` then sees the sink exit
//!   and broadcasts Shutdown.
//! * `pipeline` — runs forever; only `Ctrl+C` stops it.  Both
//!   [`ZmqSource`] and [`ZmqSink`] keep their multi-stream defaults
//!   (per-stream EOS forwards downstream but the actors keep
//!   running); the supervisor uses [`ShutdownPolicy::CtrlCOnly`].
//! * `consumer` — exits when the first wire EOS arrives for the
//!   active source.  [`ZmqSource`]'s `on_source_eos` is overridden
//!   to return `Flow::Stop`; the legacy [`Mp4Muxer`] terminus then
//!   drains.

mod assets;
#[allow(dead_code)]
mod cars_demo;
#[allow(dead_code)]
mod cli;
#[allow(dead_code)]
mod cli_zmq;

use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use clap::Parser;
use savant_core::transport::zeromq::{ReaderConfig, WriterConfig};
use savant_perception::envelopes::EncodedMsg;
use savant_perception::shutdown::{ShutdownAction, ShutdownCause, ShutdownCtx};
use savant_perception::stages::decoder::make_decode_frame;
use savant_perception::stages::{
    Mp4DemuxerResults, Mp4DemuxerSource, Mp4Muxer, PayloadCarrier, UriDemuxerResults,
    UriDemuxerSource, ZmqSink, ZmqSinkInbox, ZmqSource, ZmqSourceResults,
};
use savant_perception::supervisor::{StageKind, StageName};
use savant_perception::Flow;
use savant_perception::System;

use cars_demo::pipeline::{
    run_pipeline, PipelineHead, PipelineKnobs, PipelineTail, ShutdownPolicy,
};
use cars_demo::stats::PipelineStats;
use cli::InputSource;
use cli_zmq::{CliZmq, ConsumerArgs, PipelineArgs, ProducerArgs, Subcommand};

fn main() -> Result<()> {
    let cli = CliZmq::parse();
    let default_filter = if cli.debug {
        "info,cars_demo=debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

    match cli.subcommand {
        Subcommand::Producer(args) => run_producer(args),
        Subcommand::Pipeline(args) => run_pipeline_subcommand(args),
        Subcommand::Consumer(args) => run_consumer(args),
    }
}

/// Build a [`ReaderConfig`] from a single URL (default sockopts).
fn reader_config_from_url(url: &str) -> Result<ReaderConfig> {
    ReaderConfig::new()
        .url(url)
        .with_context(|| format!("ReaderConfig::url({url})"))?
        .build()
        .with_context(|| format!("ReaderConfig::build (url={url})"))
}

/// Build a [`WriterConfig`] from a single URL (default sockopts).
fn writer_config_from_url(url: &str) -> Result<WriterConfig> {
    WriterConfig::new()
        .url(url)
        .with_context(|| format!("WriterConfig::url({url})"))?
        .build()
        .with_context(|| format!("WriterConfig::build (url={url})"))
}

/// Resolve the producer's `--input` to either a validated filesystem
/// path or a URI, sharing the [`cli::Cli`] resolver semantics.
fn resolve_input_source(input: &str) -> Result<InputSource> {
    let probe_cli = cli::Cli::try_parse_from([
        "cars-demo",
        "--input",
        input,
        "--no-picasso", // sidesteps the --output requirement of the legacy CLI
    ])
    .with_context(|| "cars-demo CLI did not accept --input for resolution")?;
    probe_cli.resolved_input()
}

// ── Producer ────────────────────────────────────────────────────────────

fn run_producer(args: ProducerArgs) -> Result<()> {
    args.validate()?;
    let input = resolve_input_source(&args.input)?;
    log::info!(
        "cars-demo-zmq producer: input={input} zmq-out={} source_id={} channel_cap={} no_eos={} loop={}",
        args.zmq_out,
        args.source_id,
        args.channel_cap,
        args.no_eos,
        args.loop_input,
    );

    gstreamer::init().context("gstreamer init")?;

    let writer_cfg = writer_config_from_url(&args.zmq_out)?;

    let head_name = match &input {
        InputSource::Path(_) | InputSource::Uri(_) => {
            StageName::unnamed(StageKind::BitstreamSource)
        }
    };
    let sink_name = StageName::unnamed(StageKind::BitstreamSink);

    // `--no-eos` rewires both the head (demuxer) and the supervisor:
    //
    // * The demuxer's `on_source_eos` becomes a no-op, so
    //   [`EncodedMsg::SourceEos`] is *not* forwarded downstream.
    //   Without that envelope, [`ZmqSink`] never enters its EOS
    //   handler and never issues a wire EOS via
    //   `writer.send_eos(...)`.
    // * The shutdown handler broadcasts as soon as the demuxer
    //   source exits (its loop completes naturally when the input
    //   is exhausted).  The default
    //   [`producer_consumer_shutdown_handler`] would have waited
    //   for the terminus to drain on a [`SourceEos`] that never
    //   arrives, so we install a `--no-eos`-aware variant when the
    //   flag is set.
    //
    // `--loop` is independent of `--no-eos`: by default each loop
    // iteration *does* forward its terminating EOS, so a downstream
    // consumer that exits on the first wire EOS will still terminate
    // after iteration #1.  Combine `--loop --no-eos` when running
    // against a long-lived consumer that should *not* see per-loop
    // EOS markers.
    let no_eos = args.no_eos;

    let mut sys = if no_eos {
        System::new().on_shutdown(producer_no_eos_shutdown_handler)
    } else {
        System::new().on_shutdown(producer_consumer_shutdown_handler)
    };

    let stats = PipelineStats::new();
    sys.insert_shared_arc::<PipelineStats>(stats.clone());

    let source_id = args.source_id.clone();
    let loop_input = args.loop_input;
    match input {
        InputSource::Path(path) => {
            let mut results = Mp4DemuxerResults::builder().on_packet(
                |_input, source_id, info, pkt, router, ctx| {
                    if let Some(stats) = ctx.shared::<PipelineStats>() {
                        stats.demux_packets.fetch_add(1, Ordering::Relaxed);
                    }
                    let frame = make_decode_frame(source_id, &pkt, info);
                    router.send(EncodedMsg::Frame {
                        frame,
                        payload: Some(pkt.data),
                    });
                    Ok(())
                },
            );
            if no_eos {
                results = results.on_source_eos(|source_id, _router, ctx| {
                    log::info!(
                        "[{}] SourceEos {source_id}: --no-eos set, suppressing downstream forward",
                        ctx.own_name()
                    );
                    Ok(Flow::Cont)
                });
            }
            let path_str = path.to_string_lossy().into_owned();
            let mut builder = Mp4DemuxerSource::builder(head_name.clone());
            builder = if loop_input {
                builder.looped(path_str, source_id.as_str())
            } else {
                builder.one_shot(path_str, source_id.as_str())
            };
            let demux_src = builder
                .downstream(sink_name.clone())
                .results(results.build())
                .build()?;
            sys.register_source(demux_src)?;
        }
        InputSource::Uri(uri) => {
            let mut results = UriDemuxerResults::builder().on_packet(
                |_uri, source_id, info, pkt, router, ctx| {
                    if let Some(stats) = ctx.shared::<PipelineStats>() {
                        stats.demux_packets.fetch_add(1, Ordering::Relaxed);
                    }
                    let frame = make_decode_frame(source_id, &pkt, info);
                    router.send(EncodedMsg::Frame {
                        frame,
                        payload: Some(pkt.data),
                    });
                    Ok(())
                },
            );
            if no_eos {
                results = results.on_source_eos(|source_id, _router, ctx| {
                    log::info!(
                        "[{}] SourceEos {source_id}: --no-eos set, suppressing downstream forward",
                        ctx.own_name()
                    );
                    Ok(Flow::Cont)
                });
            }
            let mut builder = UriDemuxerSource::builder(head_name.clone());
            builder = if loop_input {
                builder.looped(uri, source_id.as_str())
            } else {
                builder.one_shot(uri, source_id.as_str())
            };
            let demux_src = builder
                .downstream(sink_name.clone())
                .results(results.build())
                .build()?;
            sys.register_source(demux_src)?;
        }
    }

    // Single-shot producer: override the multi-stream default so
    // the sink terminates after the demuxer's last `SourceEos` has
    // been forwarded on the wire.  `Handler<SourceEosPayload>`
    // calls this hook first, then issues `writer.send_eos(...)`
    // and classifies the writer result; returning `Flow::Stop`
    // therefore terminates the actor *after* the wire EOS
    // round-trip succeeds.  The supervisor sees `ZmqSink` exit
    // (not in its wait-list) and broadcasts Shutdown.
    //
    // With `--no-eos` the demuxer never forwards
    // [`EncodedMsg::SourceEos`], so this hook never fires — the
    // sink keeps running until the supervisor broadcasts Shutdown
    // after the demuxer source exits.
    //
    // With `--loop` the demuxer forwards an `EncodedMsg::SourceEos`
    // at the end of *every* iteration; if we kept the
    // single-shot hook the sink would terminate after iteration #1
    // and the supervisor would broadcast Shutdown across the rest
    // of the pipeline, killing the loop on its first lap.  Skip the
    // override entirely when looping so the sink uses its
    // multi-stream `Flow::Cont` default and the producer only stops
    // on `Ctrl+C`.
    //
    // Default carrier = Multipart (matches savant adapters).
    let mut zmq_sink_inbox = ZmqSinkInbox::builder();
    if !loop_input {
        zmq_sink_inbox = zmq_sink_inbox.on_source_eos(|source_id, ctx| {
            log::info!(
                "[{}] SourceEos {source_id}: producer exiting after wire EOS",
                ctx.own_name()
            );
            Ok(Flow::Stop)
        });
    }
    let zmq_sink = ZmqSink::builder(sink_name.clone(), args.channel_cap)
        .config(writer_cfg)
        .payload_carrier(PayloadCarrier::Multipart)
        .inbox(zmq_sink_inbox.build())
        .build()?;
    sys.register_actor(zmq_sink)?;

    let report = sys.run()?;
    log::info!(
        "cars-demo-zmq producer done: demux_packets={}",
        stats.demux_packets.load(Ordering::Relaxed)
    );
    report.into_result()
}

// ── Consumer ────────────────────────────────────────────────────────────

fn run_consumer(args: ConsumerArgs) -> Result<()> {
    args.validate()?;
    log::info!(
        "cars-demo-zmq consumer: zmq-in={} output={} fps={}/{} channel_cap={}",
        args.zmq_in,
        args.output.display(),
        args.fps_num,
        args.fps_den,
        args.channel_cap,
    );

    gstreamer::init().context("gstreamer init")?;

    let reader_cfg = reader_config_from_url(&args.zmq_in)?;

    let source_name = StageName::unnamed(StageKind::BitstreamSource);
    let mux_name = StageName::unnamed(StageKind::BitstreamSink);

    let mut sys = System::new().on_shutdown(producer_consumer_shutdown_handler);

    let stats = PipelineStats::new();
    sys.insert_shared_arc::<PipelineStats>(stats.clone());

    // Consumer is single-source: override the multi-stream default
    // so the source forwards a final `EncodedMsg::SourceEos` and
    // exits its loop when the first wire EOS for the active source
    // lands.  The downstream `Mp4Muxer` then drains and the broadcast
    // shutdown handler kills the rest of the system.
    let zmq_src = ZmqSource::builder(source_name.clone())
        .config(reader_cfg)
        .downstream(mux_name.clone())
        .results(
            ZmqSourceResults::builder()
                .on_source_eos(|source_id, router, _ctx| {
                    log::info!("[zmq_source] EOS source_id={source_id}; consumer stopping");
                    router.send(EncodedMsg::SourceEos {
                        source_id: source_id.to_string(),
                    });
                    Ok(Flow::Stop)
                })
                .build(),
        )
        .build()?;
    sys.register_source(zmq_src)?;

    let muxer = Mp4Muxer::builder(mux_name.clone(), args.channel_cap)
        .output(args.output.to_string_lossy().into_owned())
        .framerate(args.fps_num, args.fps_den)
        .build()?;
    sys.register_actor(muxer)?;

    let report = sys.run()?;
    log::info!(
        "cars-demo-zmq consumer done: output={}",
        args.output.display()
    );
    report.into_result()
}

// ── Pipeline ────────────────────────────────────────────────────────────

fn run_pipeline_subcommand(args: PipelineArgs) -> Result<()> {
    args.validate()?;

    // `--no-sink` short-circuits the pipeline at the tracker output:
    // we drop Picasso (and therefore the encoder) entirely and feed
    // the trailing `BitstreamFunction` directly so the bench measures
    // raw decode → infer → track throughput, free of any encode /
    // overlay / GPU-transform cost.  The flag also implies
    // `draw_enabled = false` because the Skia overlay is part of the
    // Picasso stage that no longer exists in this mode.
    //
    // We only resolve the writer config when the sink is actually
    // used; this keeps the no-sink path independent of `--zmq-out`
    // reachability.
    let picasso_enabled = !args.no_sink;
    let draw_enabled = picasso_enabled && !args.no_draw;

    log::info!(
        "cars-demo-zmq pipeline: zmq-in={} zmq-out={} gpu={} conf={} iou={} fps={}/{} \
         channel_cap={} no_draw={} no_sink={} picasso_enabled={} draw_enabled={}",
        args.zmq_in,
        args.zmq_out,
        args.gpu,
        args.conf,
        args.iou,
        args.fps_num,
        args.fps_den,
        args.channel_cap,
        args.no_draw,
        args.no_sink,
        picasso_enabled,
        draw_enabled,
    );

    let reader_cfg = reader_config_from_url(&args.zmq_in)?;

    let head = PipelineHead::Zmq {
        config: Box::new(reader_cfg),
    };
    let tail = if args.no_sink {
        PipelineTail::Function
    } else {
        let writer_cfg = writer_config_from_url(&args.zmq_out)?;
        PipelineTail::Zmq {
            config: Box::new(writer_cfg),
            payload_carrier: PayloadCarrier::Multipart,
        }
    };
    let knobs = PipelineKnobs {
        gpu: args.gpu,
        conf: args.conf,
        iou: args.iou,
        channel_cap: args.channel_cap,
        fps_num: args.fps_num,
        fps_den: args.fps_den,
        draw_enabled,
        picasso_enabled,
        debug: false,
        stats_period_ms: i64::from(u32::try_from(args.stats_period).unwrap_or(u32::MAX)) * 1_000,
    };
    run_pipeline(head, tail, knobs, ShutdownPolicy::CtrlCOnly)
}

// ── Shared shutdown policy ──────────────────────────────────────────────

/// Producer / consumer shutdown policy.
///
/// Mirrors the `cars_shutdown_handler` legacy policy but recognises
/// every source kind that may exit naturally on EOS — `Mp4Demux`,
/// `UriDemux`, and `ZmqSource`.  The terminus exit (sink / muxer)
/// triggers a broadcast Shutdown.
fn producer_consumer_shutdown_handler(
    cause: ShutdownCause,
    _ctx: &mut ShutdownCtx<'_>,
) -> Result<ShutdownAction> {
    match cause {
        ShutdownCause::StageExit { stage }
            if matches!(
                stage.kind,
                StageKind::BitstreamSource
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

/// Producer shutdown policy for `--no-eos`.
///
/// The default [`producer_consumer_shutdown_handler`] waits for the
/// terminus to drain after the demuxer source exits, relying on the
/// trailing [`EncodedMsg::SourceEos`] to walk through the pipeline
/// and stop the [`ZmqSink`].  With `--no-eos` that envelope is never
/// forwarded, so we broadcast Shutdown as soon as the demuxer source
/// exits — the sink is then torn down by the supervisor, no wire EOS
/// is ever issued, and the producer process terminates cleanly.
fn producer_no_eos_shutdown_handler(
    cause: ShutdownCause,
    _ctx: &mut ShutdownCtx<'_>,
) -> Result<ShutdownAction> {
    match cause {
        ShutdownCause::StageExit { stage }
            if matches!(stage.kind, StageKind::BitstreamSource) =>
        {
            log::info!(
                "[supervisor] {stage} exited (--no-eos); broadcasting Shutdown without wire EOS"
            );
            Ok(ShutdownAction::Broadcast {
                grace: None,
                reason: std::borrow::Cow::Owned(format!("{stage} exited (--no-eos)")),
            })
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
