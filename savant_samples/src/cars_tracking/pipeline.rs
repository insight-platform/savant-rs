//! End-to-end streaming pipeline for the `cars_tracking` sample.
//!
//! ```text
//! Mp4Demuxer -> FlexibleDecoderPool -> NvInfer -> NvTracker -> Picasso -> Mp4Muxer
//! ```
//!
//! # Architecture invariants
//!
//! - Every stage processes frames one at a time — operators are configured
//!   with `max_batch_size = 1`, `max_batch_wait = 0`.
//! - Every stage boundary is a bounded [`crossbeam::channel::bounded`] whose
//!   capacity is controlled by `--channel-cap`, so total memory usage is
//!   `O(channel_cap)` regardless of input length.
//! - Operator result callbacks only forward the raw
//!   [`deepstream_nvinfer::SealedDeliveries`] /
//!   [`deepstream_nvtracker::SealedDeliveries`] through the channel — the
//!   consumer thread on the other side calls `unseal()` before submitting
//!   the `(frame, buffer)` pair to the next stage.  Unsealing is NEVER done
//!   on the callback thread because it would pin the operator's internal
//!   completion worker while the downstream stage processes the frame.
//! - Callbacks only log per-source EOS and operator errors; pipeline
//!   shutdown is driven by the orchestrator dropping upstream senders, each
//!   stage calling `graceful_shutdown()` and processing whatever residual
//!   outputs it returns.
//! - `NvInferBatchingOperator::graceful_shutdown` / its nvtracker
//!   counterpart drain pending work into a returned `Vec<…Output>` rather
//!   than firing the callback, so the pump threads reuse
//!   [`detections::process_infer_output`] / [`tracker::process_tracker_output`]
//!   to process those drained outputs with the exact same logic as the
//!   live callback.
//! - The muxer flushes the `moov` atom only after receiving
//!   [`crate::channels::EncodedMsg::Eos`] (forwarded by the Picasso
//!   `on_encoded_frame` callback when the engine emits
//!   [`picasso::prelude::OutputMessage::EndOfStream`]).

use anyhow::{anyhow, bail, Context, Result};
use crossbeam::channel::{bounded, Receiver, RecvError, Sender};
use deepstream_buffers::{cuda_init, SurfaceView};
use deepstream_encoders::{EncoderConfig, H264EncoderConfig, NvEncoderConfig};
use deepstream_inputs::prelude::{
    FlexibleDecoderOutput, FlexibleDecoderPool, FlexibleDecoderPoolConfig, SealedDelivery,
};
use deepstream_nvinfer::prelude::{
    NvInferBatchingOperator, SealedDeliveries as InferSealedDeliveries,
    VideoFormat as InferVideoFormat,
};
use deepstream_nvtracker::{
    NvTrackerBatchingOperator, SealedDeliveries as TrackerSealedDeliveries,
};
use parking_lot::Mutex as PlMutex;
use picasso::prelude::{
    Callbacks, CodecSpec, GeneralSpec, OnEncodedFrame, OutputMessage, PicassoEngine, SourceSpec,
    TransformConfig,
};
use savant_core::pipeline::stats::{StageLatencyStat, StageProcessingStat, StageStats, Stats};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, Mp4Demuxer, Mp4DemuxerOutput, VideoInfo};
use savant_gstreamer::mp4_muxer::Mp4Muxer;

#[cfg(not(target_arch = "aarch64"))]
use deepstream_encoders::properties::H264DgpuProps;
#[cfg(target_arch = "aarch64")]
use deepstream_encoders::properties::H264JetsonProps;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::channels::{
    demux_channel, encoded_channel, DemuxMsg, DemuxSender, EncodedMsg, EncodedSender,
};
use crate::cli::ResolvedCli;

use crate::cars_tracking::detections::{
    self, build_batch_formation as build_infer_batch_formation,
    build_result_callback as build_infer_callback, build_yolo_converter, InferResultReceiver,
    InferResultSender, InferStats,
};
use crate::cars_tracking::draw::{attach_frame_id_overlay, build_vehicle_draw_spec};
use crate::cars_tracking::model::{build_nvinfer_config, promote_yolo11n_engine};
use crate::cars_tracking::tracker::{
    self, build_batch_formation as build_tracker_batch_formation,
    build_result_callback as build_tracker_callback, build_tracker_config, TrackerResultReceiver,
    TrackerResultSender, TrackerStats,
};

/// Default source identifier used throughout the pipeline.  Every stage
/// uses the same id because the sample processes exactly one input file.
const SOURCE_ID: &str = "cars-demo";
/// Decoder buffer-pool size.  Enough to hide NVDEC queue depth without
/// holding on to memory across many frames.
const DECODER_POOL_SIZE: u32 = 4;
/// Eviction TTL for [`FlexibleDecoderPool`].  Large enough that our single
/// long-running file is never evicted mid-run; a shorter value is suitable
/// for multi-source live-stream pipelines.
const DECODER_EVICTION_TTL: Duration = Duration::from_secs(3600);
/// Maximum time [`NvInferBatchingOperator::graceful_shutdown`] waits for
/// in-flight batches to complete.  Large value because TensorRT engine
/// may still be processing the last frame when shutdown is signaled.
const INFER_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum time [`NvTrackerBatchingOperator::graceful_shutdown`] waits.
const TRACKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Period (ms) at which [`Stats`] logs timestamp-based FPS while the
/// pipeline runs.  One second matches the prior custom reporter.
const STATS_TIMESTAMP_PERIOD_MS: i64 = 1_000;
/// Retained history length for [`Stats`].
const STATS_HISTORY_LEN: usize = 100;

/// Sample-level shared counters that don't map onto a per-stage processing
/// counter in [`savant_core::pipeline::stats::Stats`].
///
/// Per-stage frame / object counts live inside [`StageStats`] instances
/// registered with [`Stats`] (see [`run`]).  This struct only carries the
/// counters whose semantics are specific to this sample: demuxed
/// access-unit counts and encoded-byte totals.  The domain-specific
/// [`InferStats`] and [`TrackerStats`] arcs live alongside and are shared
/// directly with the relevant stage threads.
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Demuxed access-unit count.
    pub demux_packets: AtomicU64,
    /// Total encoded bytes written by the muxer.
    pub encoded_bytes: AtomicU64,
}

impl PipelineStats {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// Build a fresh [`StageStats`] handle bound to `name`.
fn make_stage(name: &str) -> StageStats {
    Arc::new(PlMutex::new((
        StageProcessingStat::new(name.to_string()),
        StageLatencyStat::new(name.to_string()),
    )))
}

/// Increment the `frame_counter` on a stage, optionally adding to the
/// `object_counter`.  Called on every successfully processed frame.
#[inline]
fn tick_stage(stage: &StageStats, frames: usize, objects: usize) {
    let mut guard = stage.lock();
    guard.0.frame_counter += frames;
    guard.0.object_counter += objects;
}

/// Read the total `frame_counter` value from a [`StageStats`].
#[inline]
fn stage_frames(stage: &StageStats) -> usize {
    stage.lock().0.frame_counter
}

/// Read the total `object_counter` value from a [`StageStats`].
#[inline]
fn stage_objects(stage: &StageStats) -> usize {
    stage.lock().0.object_counter
}

/// Run the cars-tracking pipeline end-to-end.  Blocks until the input MP4
/// is fully processed and the output MP4 is finalized.
pub fn run(cli: ResolvedCli) -> Result<()> {
    if !cli.input.is_file() {
        bail!("input file does not exist: {}", cli.input.display());
    }
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            bail!("output directory does not exist: {}", parent.display());
        }
    }

    gstreamer::init().context("gstreamer init")?;
    cuda_init(cli.gpu).map_err(|e| anyhow!("cuda init (gpu={}): {e}", cli.gpu))?;

    log::info!(
        "cars-demo starting: input={} output={} gpu={} conf={} iou={} channel_cap={} fps={}/{} debug={}",
        cli.input.display(),
        cli.output.display(),
        cli.gpu,
        cli.conf,
        cli.iou,
        cli.channel_cap,
        cli.fps_num,
        cli.fps_den,
        cli.debug
    );

    let stats = PipelineStats::new();
    let fatal = Arc::new(AtomicBool::new(false));

    // ── Per-stage counters + periodic FPS reporter ──────────────────────
    // `savant_core::pipeline::stats::Stats` owns a background thread that
    // emits a timestamp-based FPS line every `STATS_TIMESTAMP_PERIOD_MS` ms
    // together with the per-stage processing stats we register below.
    let core_stats = Arc::new(Stats::new(
        STATS_HISTORY_LEN,
        None,
        Some(STATS_TIMESTAMP_PERIOD_MS),
    ));
    let decode_stage = make_stage("decode");
    let infer_stage = make_stage("infer");
    let track_stage = make_stage("track");
    let encode_stage = make_stage("encode");
    core_stats.add_stage_stats(decode_stage.clone());
    core_stats.add_stage_stats(infer_stage.clone());
    core_stats.add_stage_stats(track_stage.clone());
    core_stats.add_stage_stats(encode_stage.clone());
    core_stats.kick_off();

    // ── Channels ────────────────────────────────────────────────────────
    let (tx_demux, rx_demux) = demux_channel(cli.channel_cap);
    let (tx_decoded, rx_decoded) = bounded::<SealedDelivery>(cli.channel_cap);
    let (tx_infer, rx_infer) = bounded::<InferSealedDeliveries>(cli.channel_cap);
    let (tx_tracker, rx_tracker) = bounded::<TrackerSealedDeliveries>(cli.channel_cap);
    let (tx_encoded, rx_encoded) = encoded_channel(cli.channel_cap);

    // ── Build operators (fails fast; engine build happens here) ─────────
    let infer_cfg = build_nvinfer_config(cli.gpu)?;
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
    );
    let build_t0 = Instant::now();
    let infer_operator = NvInferBatchingOperator::new(infer_cfg, infer_batch_cb, infer_result_cb)
        .map_err(|e| anyhow!("build NvInferBatchingOperator: {e}"))?;
    log::info!(
        "nvinfer engine ready ({:.1}s)",
        build_t0.elapsed().as_secs_f64()
    );
    // Cache the engine for subsequent runs.
    if let Err(e) = promote_yolo11n_engine(cli.gpu) {
        log::warn!("engine promotion failed (will rebuild next run): {e:#}");
    }

    let tracker_cfg = build_tracker_config(cli.gpu)?;
    let tracker_batch_cb = build_tracker_batch_formation();
    let tracker_stats = Arc::new(TrackerStats::new());
    let tracker_result_tx_cb = tx_tracker.clone();
    let tracker_drain_tx = tx_tracker.clone();
    drop(tx_tracker);
    let tracker_result_cb =
        build_tracker_callback(tracker_result_tx_cb, Some(tracker_stats.clone()));
    let tracker_operator =
        NvTrackerBatchingOperator::new(tracker_cfg, tracker_batch_cb, tracker_result_cb)
            .map_err(|e| anyhow!("build NvTrackerBatchingOperator: {e}"))?;
    log::info!("nvtracker ready");

    // ── Build Picasso engine ────────────────────────────────────────────
    let picasso = Arc::new(build_picasso_engine(
        tx_encoded.clone(),
        stats.clone(),
        encode_stage.clone(),
        core_stats.clone(),
    ));
    drop(tx_encoded);
    log::info!("picasso engine ready");

    // ── Spawn threads ───────────────────────────────────────────────────
    let demux_handle = spawn_demux_thread(
        cli.input.to_string_lossy().into_owned(),
        tx_demux,
        fatal.clone(),
        stats.clone(),
    )?;

    let decode_handle = spawn_decode_thread(
        SOURCE_ID.to_string(),
        cli.gpu,
        rx_demux,
        tx_decoded,
        fatal.clone(),
        decode_stage.clone(),
    )?;

    // Clones kept for the end-of-run summary (the originals are moved
    // into the stage threads below).
    let infer_stats_summary = infer_stats.clone();
    let tracker_stats_summary = tracker_stats.clone();

    let infer_handle = spawn_infer_thread(InferThreadArgs {
        source_id: SOURCE_ID.to_string(),
        operator: infer_operator,
        rx: rx_decoded,
        drain_tx: infer_drain_tx,
        converter,
        stats: infer_stats,
        fatal: fatal.clone(),
        stage: infer_stage.clone(),
    })?;

    let tracker_handle = spawn_tracker_thread(
        SOURCE_ID.to_string(),
        tracker_operator,
        rx_infer,
        tracker_drain_tx,
        tracker_stats,
        fatal.clone(),
        track_stage.clone(),
    )?;

    let render_handle =
        spawn_render_thread(SOURCE_ID.to_string(), picasso, rx_tracker, fatal.clone())?;

    let mux_handle = spawn_mux_thread(
        cli.output.to_string_lossy().into_owned(),
        cli.fps_num,
        cli.fps_den,
        rx_encoded,
        fatal.clone(),
        stats.clone(),
    )?;

    // ── Join in stage order (downstream may still be draining) ──────────
    let demux_result = demux_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("demux panic")));
    let decode_result = decode_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("decode panic")));
    let infer_result = infer_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("infer panic")));
    let tracker_result = tracker_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("tracker panic")));
    let render_result = render_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("render panic")));
    let mux_result = mux_handle
        .join()
        .unwrap_or_else(|_| Err(anyhow!("mux panic")));

    // Flush a final timestamp-based FPS record (and drop `core_stats`
    // below so the background thread is joined before we exit).
    core_stats.log_final_fps();

    log::info!(
        "cars-demo done: demux={} decoded={} infer_frames={} detections={} track_frames={} unique_tracks={} unmatched_updates={} encoded={} bytes={}",
        stats.demux_packets.load(Ordering::Relaxed),
        stage_frames(&decode_stage),
        stage_frames(&infer_stage),
        stage_objects(&infer_stage),
        stage_frames(&track_stage),
        tracker_stats_summary.unique_tracks(),
        tracker_stats_summary.unmatched_updates(),
        stage_frames(&encode_stage),
        stats.encoded_bytes.load(Ordering::Relaxed),
    );
    // Surface the detailed infer counters (detections match the
    // object_counter on the infer stage; this extra line keeps the
    // InferStats API used and helps spot divergence).
    log::debug!(
        "final InferStats: frames={} detections={}",
        infer_stats_summary.frames(),
        infer_stats_summary.detections()
    );
    drop(core_stats);

    // First error wins, rest logged.
    for (name, r) in [
        ("demux", &demux_result),
        ("decode", &decode_result),
        ("infer", &infer_result),
        ("tracker", &tracker_result),
        ("render", &render_result),
        ("mux", &mux_result),
    ] {
        if let Err(e) = r {
            log::error!("[{name}] stage error: {e:#}");
        }
    }
    demux_result?;
    decode_result?;
    infer_result?;
    tracker_result?;
    render_result?;
    mux_result?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
//  Demux stage
// ═══════════════════════════════════════════════════════════════════════

fn spawn_demux_thread(
    input: String,
    tx: DemuxSender,
    fatal: Arc<AtomicBool>,
    stats: Arc<PipelineStats>,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-demux".into())
        .spawn(move || demux_thread(input, tx, fatal, stats))
        .context("spawn demux thread")
}

fn demux_thread(
    input: String,
    tx: DemuxSender,
    fatal: Arc<AtomicBool>,
    stats: Arc<PipelineStats>,
) -> Result<()> {
    log::info!("[demux] starting on {input}");
    let tx_cb = tx.clone();
    let fatal_cb = fatal.clone();
    let stats_cb = stats.clone();
    let demuxer = Mp4Demuxer::new_parsed(&input, move |output| {
        if fatal_cb.load(Ordering::Acquire) {
            return;
        }
        match output {
            Mp4DemuxerOutput::StreamInfo(info) => {
                log::info!(
                    "[demux] stream info: {}x{} @ {}/{} codec={:?}",
                    info.width,
                    info.height,
                    info.framerate_num,
                    info.framerate_den,
                    info.codec
                );
                if tx_cb.send(DemuxMsg::StreamInfo(info)).is_err() {
                    log::warn!("[demux] receiver closed; dropping stream info");
                    fatal_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Packet(pkt) => {
                stats_cb.demux_packets.fetch_add(1, Ordering::Relaxed);
                if tx_cb.send(DemuxMsg::Packet(pkt)).is_err() {
                    log::warn!("[demux] receiver closed; dropping packet");
                    fatal_cb.store(true, Ordering::Release);
                }
            }
            Mp4DemuxerOutput::Eos => {
                log::info!("[demux] EOS");
                let _ = tx_cb.send(DemuxMsg::Eos);
            }
            Mp4DemuxerOutput::Error(e) => {
                log::error!("[demux] pipeline error: {e}");
                let _ = tx_cb.send(DemuxMsg::Error(e.to_string()));
                fatal_cb.store(true, Ordering::Release);
            }
        }
    })
    .map_err(|e| anyhow!("Mp4Demuxer::new_parsed: {e}"))?;

    demuxer.wait();
    let codec = demuxer.detected_codec();
    log::info!("[demux] finished, detected_codec={codec:?}");
    drop(tx);
    if codec.is_none() {
        bail!("demuxer did not detect a video codec (empty stream?)");
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
//  Decode stage
// ═══════════════════════════════════════════════════════════════════════

type DecodedSender = Sender<SealedDelivery>;
type DecodedReceiver = Receiver<SealedDelivery>;

fn spawn_decode_thread(
    source_id: String,
    gpu_id: u32,
    rx: Receiver<DemuxMsg>,
    tx: DecodedSender,
    fatal: Arc<AtomicBool>,
    stage: StageStats,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-decode".into())
        .spawn(move || decode_thread(source_id, gpu_id, rx, tx, fatal, stage))
        .context("spawn decode thread")
}

fn decode_thread(
    source_id: String,
    gpu_id: u32,
    rx: Receiver<DemuxMsg>,
    tx: DecodedSender,
    fatal: Arc<AtomicBool>,
    stage: StageStats,
) -> Result<()> {
    log::info!("[decode] starting source_id={source_id}");
    let dec_cfg = FlexibleDecoderPoolConfig::new(gpu_id, DECODER_POOL_SIZE, DECODER_EVICTION_TTL)
        .idle_timeout(Duration::from_secs(5))
        .detect_buffer_limit(60);

    let fatal_cb = fatal.clone();
    let tx_cb = tx.clone();
    let stage_cb = stage.clone();
    let mut decoder = FlexibleDecoderPool::new(dec_cfg, move |mut out| {
        if fatal_cb.load(Ordering::Acquire) {
            return;
        }
        match &out {
            FlexibleDecoderOutput::Frame { frame, .. } => {
                log::debug!(
                    "[decode] frame uuid={:032x} pts={}ms",
                    frame.get_uuid_u128(),
                    frame.get_pts() / 1_000_000
                );
                if let Some(sealed) = out.take_delivery() {
                    tick_stage(&stage_cb, 1, 0);
                    if tx_cb.send(sealed).is_err() {
                        log::warn!("[decode] downstream closed; dropping decoded frame");
                        fatal_cb.store(true, Ordering::Release);
                    }
                }
            }
            FlexibleDecoderOutput::ParameterChange { old, new } => {
                log::info!("[decode] parameter change: {old:?} -> {new:?}");
            }
            FlexibleDecoderOutput::Skipped { reason, .. } => {
                log::debug!("[decode] skipped: {reason:?}");
            }
            FlexibleDecoderOutput::OrphanFrame { .. } => {
                log::debug!("[decode] orphan frame (source id mismatch?)");
            }
            FlexibleDecoderOutput::SourceEos { source_id } => {
                log::info!("[decode] source EOS: {source_id}");
            }
            FlexibleDecoderOutput::Event(_) => {}
            FlexibleDecoderOutput::Error(err) => {
                log::error!("[decode] decoder error: {err}");
                fatal_cb.store(true, Ordering::Release);
            }
        }
    });

    // Drop our callback-outside copy of tx so the callback's clone is the
    // only remaining strong producer.
    drop(tx);

    let mut saw_eos = false;
    let mut video_info: Option<VideoInfo> = None;
    loop {
        match rx.recv() {
            Ok(DemuxMsg::StreamInfo(info)) => {
                if video_info.is_some() {
                    log::warn!("[decode] duplicate StreamInfo; ignoring");
                } else {
                    log::info!(
                        "[decode] stream info: {}x{} @ {}/{}",
                        info.width,
                        info.height,
                        info.framerate_num,
                        info.framerate_den
                    );
                    video_info = Some(info);
                }
            }
            Ok(DemuxMsg::Packet(pkt)) => {
                let Some(info) = video_info.as_ref() else {
                    log::error!("[decode] packet received before StreamInfo — aborting");
                    fatal.store(true, Ordering::Release);
                    break;
                };
                let frame = make_decode_frame(&source_id, &pkt, info);
                if let Err(e) = decoder.submit(&frame, Some(&pkt.data)) {
                    log::error!("[decode] submit failed: {e}");
                    fatal.store(true, Ordering::Release);
                    break;
                }
            }
            Ok(DemuxMsg::Eos) => {
                log::info!("[decode] received EOS from demux");
                saw_eos = true;
                break;
            }
            Ok(DemuxMsg::Error(msg)) => {
                log::error!("[decode] demux error: {msg}");
                fatal.store(true, Ordering::Release);
                break;
            }
            Err(RecvError) => {
                log::warn!("[decode] demux channel closed without EOS");
                break;
            }
        }
    }

    if saw_eos {
        if let Err(e) = decoder.source_eos(&source_id) {
            log::warn!("[decode] source_eos failed: {e}");
        }
    }
    if let Err(e) = decoder.graceful_shutdown() {
        log::warn!("[decode] graceful_shutdown failed: {e}");
    }
    drop(decoder);
    log::info!("[decode] finished ({} frames)", stage_frames(&stage));
    Ok(())
}

/// Fallback framerate used only when the container does not advertise one
/// (per [`VideoInfo`] contract: `framerate_num == 0`).
const FALLBACK_FPS_NUM: i64 = 30;
const FALLBACK_FPS_DEN: i64 = 1;

fn make_decode_frame(source_id: &str, pkt: &DemuxedPacket, info: &VideoInfo) -> VideoFrameProxy {
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

// ═══════════════════════════════════════════════════════════════════════
//  Inference stage
// ═══════════════════════════════════════════════════════════════════════

struct InferThreadArgs {
    source_id: String,
    operator: NvInferBatchingOperator,
    rx: DecodedReceiver,
    drain_tx: InferResultSender,
    converter: Arc<savant_core::converters::YoloDetectionConverter>,
    stats: Arc<InferStats>,
    fatal: Arc<AtomicBool>,
    stage: StageStats,
}

fn spawn_infer_thread(args: InferThreadArgs) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-infer".into())
        .spawn(move || infer_thread(args))
        .context("spawn infer thread")
}

fn infer_thread(args: InferThreadArgs) -> Result<()> {
    let InferThreadArgs {
        source_id,
        mut operator,
        rx,
        drain_tx,
        converter,
        stats,
        fatal,
        stage,
    } = args;
    log::info!("[infer] starting source_id={source_id}");
    // Baseline for per-frame detection deltas (the operator callback
    // increments `stats.detections()` asynchronously, so we snapshot the
    // cumulative count before each `add_frame` to derive the per-call
    // object delta for the pipeline stage counters).
    let mut det_baseline = stats.detections();
    loop {
        match rx.recv() {
            Ok(sealed) => {
                // Decoder SealedDelivery carries a single frame/buffer pair.
                if let Some((frame, buffer)) = sealed.unseal() {
                    log::debug!(
                        "[infer] submit frame uuid={:032x} pts={}ms",
                        frame.get_uuid_u128(),
                        frame.get_pts() / 1_000_000
                    );
                    if let Err(e) = operator.add_frame(frame, buffer) {
                        log::error!("[infer] add_frame failed: {e}");
                        fatal.store(true, Ordering::Release);
                        break;
                    }
                    let now = stats.detections();
                    tick_stage(&stage, 1, (now - det_baseline) as usize);
                    det_baseline = now;
                }
                if fatal.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(RecvError) => {
                log::info!("[infer] upstream channel closed");
                break;
            }
        }
    }

    if let Err(e) = operator.send_eos(&source_id) {
        log::warn!("[infer] send_eos failed: {e}");
    }
    match operator.graceful_shutdown(INFER_DRAIN_TIMEOUT) {
        Ok(drained) => {
            log::info!("[infer] drained {} remaining outputs", drained.len());
            for out in drained {
                detections::process_infer_output(out, converter.as_ref(), &drain_tx, Some(&stats));
            }
        }
        Err(e) => {
            log::error!("[infer] graceful_shutdown failed: {e}");
            fatal.store(true, Ordering::Release);
        }
    }
    // Absorb any detections registered after the last tick above (including
    // results produced by `graceful_shutdown`) so the stage object counter
    // matches the sample-level InferStats.
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

// ═══════════════════════════════════════════════════════════════════════
//  Tracker stage
// ═══════════════════════════════════════════════════════════════════════

fn spawn_tracker_thread(
    source_id: String,
    operator: NvTrackerBatchingOperator,
    rx: InferResultReceiver,
    drain_tx: TrackerResultSender,
    stats: Arc<TrackerStats>,
    fatal: Arc<AtomicBool>,
    stage: StageStats,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-tracker".into())
        .spawn(move || tracker_thread(source_id, operator, rx, drain_tx, stats, fatal, stage))
        .context("spawn tracker thread")
}

fn tracker_thread(
    source_id: String,
    mut operator: NvTrackerBatchingOperator,
    rx: InferResultReceiver,
    drain_tx: TrackerResultSender,
    stats: Arc<TrackerStats>,
    fatal: Arc<AtomicBool>,
    stage: StageStats,
) -> Result<()> {
    log::info!("[track] starting source_id={source_id}");
    loop {
        match rx.recv() {
            Ok(sealed) => {
                let pairs = sealed.unseal();
                for (frame, buffer) in pairs {
                    log::debug!("[track] submit frame pts={}ms", frame.get_pts() / 1_000_000);
                    if let Err(e) = operator.add_frame(frame, buffer) {
                        log::error!("[track] add_frame failed: {e}");
                        fatal.store(true, Ordering::Release);
                        break;
                    }
                    tick_stage(&stage, 1, 0);
                }
                if fatal.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(RecvError) => {
                log::info!("[track] upstream channel closed");
                break;
            }
        }
    }

    if let Err(e) = operator.send_eos(&source_id) {
        log::warn!("[track] send_eos failed: {e}");
    }
    match operator.graceful_shutdown(TRACKER_DRAIN_TIMEOUT) {
        Ok(drained) => {
            log::info!("[track] drained {} remaining outputs", drained.len());
            for out in drained {
                tracker::process_tracker_output(out, &drain_tx, Some(&stats));
            }
        }
        Err(e) => {
            log::error!("[track] graceful_shutdown failed: {e}");
            fatal.store(true, Ordering::Release);
        }
    }
    drop(drain_tx);
    drop(operator);
    log::info!(
        "[track] finished: frames={} unique_tracks={}",
        stage_frames(&stage),
        stats.unique_tracks()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
//  Render stage (Picasso)
// ═══════════════════════════════════════════════════════════════════════

fn build_picasso_engine(
    tx_encoded: EncodedSender,
    stats: Arc<PipelineStats>,
    stage: StageStats,
    core_stats: Arc<Stats>,
) -> PicassoEngine {
    let callbacks = Callbacks::builder()
        .on_encoded_frame(EncodedSink {
            tx: tx_encoded,
            stats,
            stage,
            core_stats,
        })
        .build();
    let general = GeneralSpec::builder()
        .name("cars-demo")
        .idle_timeout_secs(600)
        .inflight_queue_size(8)
        .build();
    PicassoEngine::new(general, callbacks)
}

struct EncodedSink {
    tx: EncodedSender,
    stats: Arc<PipelineStats>,
    stage: StageStats,
    /// Pipeline-tail counter — the timestamp-based FPS emitted by
    /// [`Stats`] is driven by the rate at which we call
    /// [`Stats::register_frame`] here on the encoded-frame callback.
    core_stats: Arc<Stats>,
}

impl OnEncodedFrame for EncodedSink {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                let content = frame.get_content();
                let data = match content.as_ref() {
                    VideoFrameContent::Internal(d) => d.clone(),
                    other => {
                        log::error!(
                            "[encode-cb] unexpected content variant: {:?}",
                            std::mem::discriminant(other)
                        );
                        return;
                    }
                };
                let pts_ns = frame.get_pts().max(0) as u64;
                let dts_ns = frame.get_dts().map(|v| v.max(0) as u64);
                let duration_ns = frame.get_duration().map(|v| v.max(0) as u64);
                tick_stage(&self.stage, 1, 0);
                self.core_stats.register_frame(0);
                self.stats
                    .encoded_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                log::debug!(
                    "[encode-cb] frame pts={}ms bytes={}",
                    pts_ns / 1_000_000,
                    data.len()
                );
                if self
                    .tx
                    .send(EncodedMsg::AccessUnit {
                        data,
                        pts_ns,
                        dts_ns,
                        duration_ns,
                    })
                    .is_err()
                {
                    log::warn!("[encode-cb] muxer closed; dropping encoded frame");
                }
            }
            OutputMessage::EndOfStream(_) => {
                log::info!("[encode-cb] EndOfStream");
                let _ = self.tx.send(EncodedMsg::Eos);
            }
        }
    }
}

fn spawn_render_thread(
    source_id: String,
    picasso: Arc<PicassoEngine>,
    rx: TrackerResultReceiver,
    fatal: Arc<AtomicBool>,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-render".into())
        .spawn(move || render_thread(source_id, picasso, rx, fatal))
        .context("spawn render thread")
}

fn render_thread(
    source_id: String,
    picasso: Arc<PicassoEngine>,
    rx: TrackerResultReceiver,
    fatal: Arc<AtomicBool>,
) -> Result<()> {
    log::info!("[render] starting source_id={source_id}");
    let mut source_spec_set = false;
    // Monotonically increasing per-frame counter rendered as the
    // top-left `frame #N` badge.  Attaching the overlay here — after
    // inference and tracking — keeps the synthetic object invisible to
    // both the inference batch formation (`RoiKind::FullFrame`) and the
    // tracker batch formation (which filters by
    // `DETECTION_NAMESPACE`); see
    // [`crate::cars_tracking::draw::attach_frame_id_overlay`].
    let mut frame_counter: u64 = 0;
    loop {
        match rx.recv() {
            Ok(sealed) => {
                let pairs = sealed.unseal();
                for (frame, buffer) in pairs {
                    if !source_spec_set {
                        let w = frame.get_width().max(1) as u32;
                        let h = frame.get_height().max(1) as u32;
                        let fps = frame.get_fps();
                        log::info!("[render] first frame: {w}x{h} fps={}/{}", fps.0, fps.1);
                        let spec = build_source_spec(w, h, fps.0 as i32, fps.1 as i32)?;
                        picasso
                            .set_source_spec(&source_id, spec)
                            .map_err(|e| anyhow!("set_source_spec: {e}"))?;
                        source_spec_set = true;
                    }

                    if let Err(e) = attach_frame_id_overlay(&frame, frame_counter) {
                        log::warn!("[render] attach_frame_id_overlay failed: {e}");
                    }
                    frame_counter = frame_counter.wrapping_add(1);

                    let view = SurfaceView::from_buffer(&buffer, 0)
                        .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
                    log::debug!("[render] send frame pts={}ms", frame.get_pts() / 1_000_000);
                    if let Err(e) = picasso.send_frame(&source_id, frame, view, None) {
                        log::error!("[render] send_frame failed: {e}");
                        fatal.store(true, Ordering::Release);
                        break;
                    }
                }
                if fatal.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(RecvError) => {
                log::info!("[render] upstream channel closed");
                break;
            }
        }
    }

    // Signal end-of-source, then drain workers.  `shutdown` blocks until
    // all workers have finished; the OnEncodedFrame callback forwards the
    // resulting `OutputMessage::EndOfStream` onto the muxer channel.
    if let Err(e) = picasso.send_eos(&source_id) {
        log::warn!("[render] send_eos: {e}");
    }
    picasso.shutdown();
    // Drop our strong ref — the only remaining ref is the Arc passed to
    // workers, which are already joined by `shutdown()`.  After this drop
    // the OnEncodedFrame callback's tx_encoded is released.
    drop(picasso);
    log::info!("[render] finished");
    Ok(())
}

fn build_source_spec(width: u32, height: u32, fps_num: i32, fps_den: i32) -> Result<SourceSpec> {
    let encoder = build_encoder_config(width, height, fps_num, fps_den);
    let draw = build_vehicle_draw_spec().context("build vehicle draw spec")?;
    Ok(SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(encoder),
        },
        draw,
        font_family: "monospace".to_string(),
        use_on_render: false,
        use_on_gpumat: false,
        ..Default::default()
    })
}

fn build_encoder_config(width: u32, height: u32, fps_num: i32, fps_den: i32) -> NvEncoderConfig {
    let cfg = H264EncoderConfig::new(width, height)
        .format(InferVideoFormat::RGBA)
        .fps(fps_num, fps_den);
    #[cfg(target_arch = "aarch64")]
    let cfg = cfg.props(H264JetsonProps {
        bitrate: Some(6_000_000),
        iframeinterval: Some(fps_num.max(1) as u32),
        ..Default::default()
    });
    #[cfg(not(target_arch = "aarch64"))]
    let cfg = cfg.props(H264DgpuProps {
        bitrate: Some(6_000_000),
        iframeinterval: Some(fps_num.max(1) as u32),
        ..Default::default()
    });
    NvEncoderConfig::new(0, EncoderConfig::H264(cfg)).name("cars-demo/enc")
}

// ═══════════════════════════════════════════════════════════════════════
//  Mux stage
// ═══════════════════════════════════════════════════════════════════════

fn spawn_mux_thread(
    output: String,
    fps_num: i32,
    fps_den: i32,
    rx: Receiver<EncodedMsg>,
    fatal: Arc<AtomicBool>,
    stats: Arc<PipelineStats>,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-mux".into())
        .spawn(move || mux_thread(output, fps_num, fps_den, rx, fatal, stats))
        .context("spawn mux thread")
}

fn mux_thread(
    output: String,
    fps_num: i32,
    fps_den: i32,
    rx: Receiver<EncodedMsg>,
    fatal: Arc<AtomicBool>,
    _stats: Arc<PipelineStats>,
) -> Result<()> {
    log::info!("[mux] starting output={output} fps={fps_num}/{fps_den}");
    let mut muxer = Mp4Muxer::new(VideoCodec::H264, &output, fps_num, fps_den)
        .map_err(|e| anyhow!("Mp4Muxer::new: {e}"))?;

    loop {
        match rx.recv() {
            Ok(EncodedMsg::AccessUnit {
                data,
                pts_ns,
                dts_ns,
                duration_ns,
            }) => {
                if let Err(e) = muxer.push(&data, pts_ns, dts_ns, duration_ns) {
                    log::error!("[mux] push failed: {e}");
                    fatal.store(true, Ordering::Release);
                    break;
                }
            }
            Ok(EncodedMsg::Eos) => {
                log::info!("[mux] EOS");
                break;
            }
            Err(RecvError) => {
                log::info!("[mux] upstream channel closed without EOS");
                break;
            }
        }
    }

    muxer
        .finish()
        .map_err(|e| anyhow!("Mp4Muxer::finish: {e}"))?;
    log::info!("[mux] finished: {}", output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    /// Integration guard: the orchestrator must fail fast when the input
    /// file is missing, before touching GStreamer / CUDA.
    #[test]
    fn run_rejects_missing_input() {
        let cli = ResolvedCli {
            input: PathBuf::from("/definitely/does/not/exist.mp4"),
            output: std::env::temp_dir().join("savant_samples_pipeline_out.mp4"),
            gpu: 0,
            conf: 0.25,
            iou: 0.45,
            channel_cap: 4,
            debug: false,
            fps_num: 25,
            fps_den: 1,
        };
        let err = run(cli).unwrap_err();
        assert!(
            err.to_string().contains("input file does not exist"),
            "unexpected error: {err:#}"
        );
    }
}
