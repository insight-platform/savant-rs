//! Picasso actor(s) for the `cars_tracking` sample.
//!
//! Two actors share this module because they occupy the same slot in
//! the pipeline topology (pipeline-tail, consuming the tracker stage's
//! output) and only one of them runs for any given invocation:
//!
//! * [`spawn_picasso_thread`] — the full Picasso-backed actor.  Owns
//!   the [`PicassoEngine`], the [`EncodedSink`] (
//!   [`OnEncodedFrame`] callback that pushes [`EncodedMsg`] onto the
//!   mux channel), and the overlay/draw-spec wiring.  Drives the
//!   encode + mux tail of the pipeline.
//! * [`spawn_drain_thread`] — the `--no-picasso` actor.  Unseals
//!   every [`PipelineMsg::Delivery`] and drops it, keeping GPU slots
//!   recycled.  Also ticks the pipeline-tail stage + registers frames
//!   with [`Stats`] so the time-based FPS counter works in
//!   `--no-picasso` mode.
//!
//! Both actors terminate on the in-band
//! [`PipelineMsg::SourceEos`](crate::cars_tracking::message::PipelineMsg::SourceEos)
//! sentinel — the Picasso actor forwards an [`EncodedMsg::Eos`]
//! downstream after [`PicassoEngine::send_eos`] drains and fires
//! [`OutputMessage::EndOfStream`].

pub mod draw_spec;

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::{bounded, Receiver, RecvTimeoutError, Sender};
use deepstream_buffers::SurfaceView;
use deepstream_encoders::{EncoderConfig, H264EncoderConfig, NvEncoderConfig};
use deepstream_nvinfer::prelude::VideoFormat as InferVideoFormat;
use hashbrown::HashSet;
use picasso::prelude::{
    Callbacks, CodecSpec, GeneralSpec, ObjectDrawSpec, OnEncodedFrame, OutputMessage,
    PicassoEngine, SourceSpec, TransformConfig,
};
use savant_core::pipeline::stats::{StageStats, Stats};
use savant_core::primitives::frame::VideoFrameContent;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "aarch64"))]
use deepstream_encoders::properties::H264DgpuProps;
#[cfg(target_arch = "aarch64")]
use deepstream_encoders::properties::H264JetsonProps;

use self::draw_spec::{attach_frame_id_overlay, build_vehicle_draw_spec};
use super::decoder::handle_shutdown;
use super::tracker::TrackerResultReceiver;
use crate::cars_tracking::message::PipelineMsg;
use crate::cars_tracking::stats::{tick_stage, PipelineStats};
// `name` is passed in per spawn so picasso and drain actors carry
// disambiguating instance tags when multiple are live.
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};

/// Receive poll timeout for both [`picasso_thread`] and
/// [`drain_thread`].  Drives the grace-deadline check and keeps the
/// loop responsive while there is no upstream traffic.
const PICASSO_RECV_POLL: Duration = Duration::from_millis(100);

/// Picasso -> mux boundary message type.
///
/// Emitted exclusively by the [`EncodedSink`] on the
/// [`OnEncodedFrame`] callback — that callback is the carrier-side
/// boundary between Picasso's encoder thread and the sample's muxer
/// actor.
#[derive(Debug)]
pub enum EncodedMsg {
    /// A single H.264 access unit plus its timing metadata.
    AccessUnit {
        /// Encoded access unit payload.
        data: Vec<u8>,
        /// Presentation timestamp in nanoseconds.
        pts_ns: u64,
        /// Decode timestamp in nanoseconds (optional — mirrors the
        /// GStreamer buffer convention).
        dts_ns: Option<u64>,
        /// Duration in nanoseconds (optional).
        duration_ns: Option<u64>,
    },
    /// End-of-stream sentinel forwarded after
    /// [`OutputMessage::EndOfStream`].  The muxer flushes its `moov`
    /// atom on receipt.
    Eos,
}

/// Alias for the picasso -> mux channel sender.
pub type EncodedSender = Sender<EncodedMsg>;
/// Alias for the picasso -> mux channel receiver.
pub type EncodedReceiver = Receiver<EncodedMsg>;

/// Create a bounded picasso -> mux channel.
pub fn encoded_channel(cap: usize) -> (EncodedSender, EncodedReceiver) {
    bounded(cap)
}

/// Build the Picasso rendering engine wired to the `cars-demo`
/// encoded-frame sink.
pub fn build_picasso_engine(
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

/// `on_encoded_frame` callback — forwards encoded access units /
/// EOS onto the mux channel and drives the pipeline-tail counters.
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

// ═══════════════════════════════════════════════════════════════════════
//  Picasso actor
// ═══════════════════════════════════════════════════════════════════════

/// Spawn the Picasso-backed rendering actor.
///
/// The actor consumes [`PipelineMsg`] from the tracker stage: deliveries
/// are fed to [`PicassoEngine::send_frame`], per-source EOS initiates
/// `picasso.send_eos(&sid)` (whose completion is surfaced downstream by
/// the [`OnEncodedFrame`] callback), and [`PipelineMsg::Shutdown`] drives
/// cooperative exit.  Fatal errors surface as `Err(_)`.
pub fn spawn_picasso_thread(
    picasso: Arc<PicassoEngine>,
    rx: TrackerResultReceiver,
    draw_enabled: bool,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-picasso".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            picasso_thread(picasso, rx, draw_enabled)
        })
        .context("spawn picasso thread")
}

fn picasso_thread(
    picasso: Arc<PicassoEngine>,
    rx: TrackerResultReceiver,
    draw_enabled: bool,
) -> Result<()> {
    log::info!("[picasso] starting draw_enabled={draw_enabled}");
    // Per-source tracking: Picasso is multiplexed (each source_id
    // gets its own SourceSpec / encoder slot).  We lazily register
    // a `SourceSpec` the first time we see a given source_id, and
    // issue `send_eos(sid)` as soon as the upstream SourceEos
    // sentinel arrives.  The loop exits on upstream channel closure
    // or `PipelineMsg::Shutdown` (after any grace period).
    let mut registered: HashSet<String> = HashSet::new();
    // Monotonically increasing per-frame counter rendered as the
    // top-left `frame #N` badge when drawing is enabled.  Attaching
    // the overlay here — after inference and tracking — keeps the
    // synthetic object invisible to both the inference batch
    // formation (`RoiKind::FullFrame`) and the tracker batch
    // formation (which filters by `DETECTION_NAMESPACE`); see
    // [`draw_spec::attach_frame_id_overlay`].  When
    // `draw_enabled == false` we skip the attach entirely and install
    // an empty [`ObjectDrawSpec`] on the source spec so Picasso
    // renders no overlays at all — decode → infer → track →
    // transform → encode still run as normal.
    let mut frame_counter: u64 = 0;
    let mut deadline: Option<Instant> = None;
    let mut break_now = false;
    loop {
        match rx.recv_timeout(PICASSO_RECV_POLL) {
            Ok(msg @ (PipelineMsg::Delivery(_) | PipelineMsg::Deliveries(_))) => {
                // Generalized ingress: both delivery shapes are
                // normalized via [`PipelineMsg::into_pairs`] and
                // forwarded to Picasso one-by-one.
                for (frame, buffer) in msg.into_pairs() {
                    let sid = frame.get_source_id();
                    if registered.insert(sid.clone()) {
                        let w = frame.get_width().max(1) as u32;
                        let h = frame.get_height().max(1) as u32;
                        let fps = frame.get_fps();
                        log::info!(
                            "[picasso] registering source_id={sid} {w}x{h} fps={}/{} draw_enabled={draw_enabled}",
                            fps.0,
                            fps.1,
                        );
                        let spec =
                            build_source_spec(w, h, fps.0 as i32, fps.1 as i32, draw_enabled)?;
                        picasso
                            .set_source_spec(&sid, spec)
                            .map_err(|e| anyhow!("set_source_spec({sid}): {e}"))?;
                    }

                    if draw_enabled {
                        if let Err(e) = attach_frame_id_overlay(&frame, frame_counter) {
                            log::warn!("[picasso] attach_frame_id_overlay failed: {e}");
                        }
                    }
                    frame_counter = frame_counter.wrapping_add(1);

                    let view = SurfaceView::from_buffer(&buffer, 0)
                        .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
                    if let Err(e) = picasso.send_frame(&sid, frame, view, None) {
                        log::error!("[picasso] send_frame failed: {e}");
                        return Err(anyhow!("picasso send_frame: {e}"));
                    }
                }
            }
            Ok(PipelineMsg::SourceEos { source_id: sid }) => {
                // Per-source flush only — Picasso's
                // [`OnEncodedFrame`] callback asynchronously emits
                // [`EncodedMsg::Eos`] on
                // [`OutputMessage::EndOfStream`] once the last
                // frame for this source has drained.  The loop
                // does NOT break; other sources may still stream.
                log::info!("[picasso] SourceEos {sid}: flushing picasso");
                if let Err(e) = picasso.send_eos(&sid) {
                    log::warn!("[picasso] send_eos({sid}): {e}");
                }
            }
            Ok(PipelineMsg::Shutdown { grace, reason }) => {
                handle_shutdown("picasso", grace, &reason, &mut deadline, &mut break_now);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[picasso] upstream channel disconnected; exiting receive loop");
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
                log::info!("[picasso] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }

    picasso.shutdown();
    // Drop our strong ref — the only remaining ref is the Arc passed
    // to workers, which are already joined by `shutdown()`.  After
    // this drop the OnEncodedFrame callback's tx_encoded is released.
    drop(picasso);
    log::info!("[picasso] finished");
    Ok(())
}

fn build_source_spec(
    width: u32,
    height: u32,
    fps_num: i32,
    fps_den: i32,
    draw_enabled: bool,
) -> Result<SourceSpec> {
    let encoder = build_encoder_config(width, height, fps_num, fps_den);
    // Empty [`ObjectDrawSpec`] is the `--no-draw` escape hatch:
    // Picasso's per-object lookup returns `None` for every
    // `(namespace, label)` pair, so no bboxes/labels/badges are
    // composited onto the frame.  The transform + encode stages
    // still execute so the output MP4 is a clean re-encoded copy of
    // the source.
    let draw = if draw_enabled {
        build_vehicle_draw_spec().context("build vehicle draw spec")?
    } else {
        ObjectDrawSpec::default()
    };
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
