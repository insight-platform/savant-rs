//! Picasso actor(s) for the `cars_tracking` sample.
//!
//! Two actors share this module because they occupy the same slot in
//! the pipeline topology (pipeline-tail, consuming the tracker stage's
//! output) and only one of them runs for any given invocation:
//!
//! * [`spawn_picasso_thread`] — the full Picasso-backed actor.  Owns
//!   the [`PicassoEngine`], the [`EncodedSink`]
//!   ([`OnEncodedFrame`] callback that pushes [`EncodedMsg`] onto the
//!   downstream channel), and the overlay/draw-spec wiring.  Drives
//!   the encode + mux tail of the pipeline.
//!
//! The actor does not know *what* the downstream consumer of its
//! encoded output will do — that is strictly the downstream actor's
//! concern.  Picasso forwards each encoded
//! [`OutputMessage::VideoFrame`] as [`EncodedMsg::Frame`] (the
//! pre-built `VideoFrameProxy` already carries source id, pts/dts,
//! duration, keyframe, and the encoded bitstream in its
//! [`VideoFrameContent::Internal`]) and
//! [`OutputMessage::EndOfStream`] as [`EncodedMsg::SourceEos`] (with
//! the source id) so the downstream terminus can decide for itself
//! when / how to finalise.

pub mod draw_spec;

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};
use deepstream_buffers::SurfaceView;
use deepstream_encoders::{EncoderConfig, H264EncoderConfig, NvEncoderConfig};
use deepstream_nvinfer::prelude::VideoFormat as InferVideoFormat;
use hashbrown::HashSet;
use picasso::prelude::{
    Callbacks, CodecSpec, GeneralSpec, ObjectDrawSpec, OnEncodedFrame, OutputMessage,
    PicassoEngine, SourceSpec, TransformConfig,
};
use savant_core::pipeline::stats::{StageStats, Stats};
use savant_core::primitives::frame::{VideoFrameContent, VideoFrameProxy};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "aarch64"))]
use deepstream_encoders::properties::H264DgpuProps;
#[cfg(target_arch = "aarch64")]
use deepstream_encoders::properties::H264JetsonProps;

use self::draw_spec::{attach_frame_id_overlay, build_vehicle_draw_spec};
use crate::cars_tracking::message::{apply_shutdown_signal, EncodedMsg, PipelineMsg};
use crate::cars_tracking::stats::{tick_stage, PipelineStats};
// `name` is passed in per spawn so picasso and drain actors carry
// disambiguating instance tags when multiple are live.
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};

/// Receive poll timeout for both [`picasso_thread`] and
/// [`drain_thread`].  Drives the grace-deadline check and keeps the
/// loop responsive while there is no upstream traffic.
const PICASSO_RECV_POLL: Duration = Duration::from_millis(100);

/// Build the Picasso rendering engine wired to the `cars-demo`
/// encoded-frame sink.
///
/// `name` is stored on the [`EncodedSink`] so every log record
/// emitted from the `on_encoded_frame` callback carries the actor's
/// configured [`StageName`] (kind + instance tag) — matching the
/// main actor thread's logs even when multiple Picasso actors run
/// concurrently.
pub fn build_picasso_engine(
    tx_encoded: Sender<EncodedMsg>,
    stats: Arc<PipelineStats>,
    stage: StageStats,
    core_stats: Arc<Stats>,
    name: StageName,
) -> PicassoEngine {
    let callbacks = Callbacks::builder()
        .on_encoded_frame(EncodedSink {
            tx: tx_encoded,
            stats,
            stage,
            core_stats,
            name,
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
/// source-EOS onto the downstream channel and drives the
/// pipeline-tail counters.
struct EncodedSink {
    tx: Sender<EncodedMsg>,
    stats: Arc<PipelineStats>,
    stage: StageStats,
    /// Pipeline-tail counter — the timestamp-based FPS emitted by
    /// [`Stats`] is driven by the rate at which we call
    /// [`Stats::register_frame`] here on the encoded-frame callback.
    core_stats: Arc<Stats>,
    /// Picasso actor's [`StageName`] — used verbatim as the log
    /// prefix on every record the sink emits so the callback's
    /// output is indistinguishable from the picasso thread's.
    name: StageName,
}

impl EncodedSink {
    /// Validate + trace an encoded [`VideoFrameProxy`].
    ///
    /// Ensures the frame's content is a [`VideoFrameContent::Internal`]
    /// blob (the only variant an encoder callback may legitimately
    /// produce in this pipeline), then updates per-stage +
    /// pipeline-tail counters and emits a single `debug!` line.
    /// PTS is logged verbatim — every stage in this pipeline
    /// treats PTS as an opaque, codec-supplied tick value and
    /// never rescales it to wall-clock units.
    ///
    /// Returns `true` when the frame is forwardable; `false`
    /// means the content variant was unexpected (logged at
    /// `error!`) and the caller must drop the frame.
    fn trace_encoded_frame(&self, frame: &VideoFrameProxy) -> bool {
        let bytes = match frame.get_content().as_ref() {
            VideoFrameContent::Internal(d) => d.len(),
            other => {
                log::error!(
                    "[{}/encode-cb] unexpected content variant: {:?}",
                    self.name,
                    std::mem::discriminant(other)
                );
                return false;
            }
        };
        tick_stage(&self.stage, 1, 0);
        self.core_stats.register_frame(0);
        self.stats
            .encoded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        log::debug!(
            "[{}/encode-cb] frame source_id={} pts={} bytes={bytes}",
            self.name,
            frame.get_source_id(),
            frame.get_pts(),
        );
        true
    }
}

impl OnEncodedFrame for EncodedSink {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                // Forward the pre-built frame verbatim — the frame
                // already carries everything downstream needs
                // (source id, pts/dts, duration, keyframe, encoded
                // bitstream in `VideoFrameContent::Internal`).  The
                // trace hook validates content and updates stats;
                // if it rejects the frame we simply drop it.
                if !self.trace_encoded_frame(&frame) {
                    return;
                }
                if self
                    .tx
                    .send(EncodedMsg::Frame {
                        frame,
                        payload: None,
                    })
                    .is_err()
                {
                    log::warn!(
                        "[{}/encode-cb] downstream closed; dropping encoded frame",
                        self.name
                    );
                }
            }
            OutputMessage::EndOfStream(eos) => {
                let source_id = eos.source_id;
                log::info!(
                    "[{}/encode-cb] EndOfStream source_id={source_id}",
                    self.name
                );
                let _ = self.tx.send(EncodedMsg::SourceEos { source_id });
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
    rx: Receiver<PipelineMsg>,
    draw_enabled: bool,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    let guard_name = name.clone();
    thread::Builder::new()
        .name("cars-picasso".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(guard_name, exit_tx);
            picasso_thread(picasso, rx, draw_enabled, name)
        })
        .context("spawn picasso thread")
}

fn picasso_thread(
    picasso: Arc<PicassoEngine>,
    rx: Receiver<PipelineMsg>,
    draw_enabled: bool,
    name: StageName,
) -> Result<()> {
    log::info!("[{name}] starting draw_enabled={draw_enabled}");
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
                            "[{name}] registering source_id={sid} {w}x{h} fps={}/{} draw_enabled={draw_enabled}",
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
                            log::warn!("[{name}] attach_frame_id_overlay failed: {e}");
                        }
                    }
                    frame_counter = frame_counter.wrapping_add(1);

                    let view = SurfaceView::from_buffer(&buffer, 0)
                        .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
                    if let Err(e) = picasso.send_frame(&sid, frame, view, None) {
                        log::error!("[{name}] send_frame failed: {e}");
                        return Err(anyhow!("picasso send_frame: {e}"));
                    }
                }
            }
            Ok(PipelineMsg::SourceEos { source_id: sid }) => {
                // Per-source flush only — Picasso's
                // [`OnEncodedFrame`] callback asynchronously emits
                // [`EncodedMsg::SourceEos { source_id }`] on
                // [`OutputMessage::EndOfStream`] once the last
                // frame for this source has drained.  The loop
                // does NOT break; other sources may still stream.
                log::info!("[{name}] SourceEos {sid}: flushing picasso");
                if let Err(e) = picasso.send_eos(&sid) {
                    log::warn!("[{name}] send_eos({sid}): {e}");
                }
            }
            Ok(PipelineMsg::Shutdown { grace, reason }) => {
                apply_shutdown_signal(&name, grace, &reason, &mut deadline, &mut break_now);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[{name}] upstream channel disconnected; exiting receive loop");
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
                log::info!("[{name}] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }

    picasso.shutdown();
    // Drop our strong ref — the only remaining ref is the Arc passed
    // to workers, which are already joined by `shutdown()`.  After
    // this drop the OnEncodedFrame callback's tx_encoded is released.
    drop(picasso);
    log::info!("[{name}] finished");
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
