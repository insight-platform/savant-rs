use crate::callbacks::Callbacks;
use crate::message::{EncodedOutput, WorkerMessage};
use crate::pipeline::encode::RenderOpts;
use crate::pipeline::{bypass, encode, FrameInput};
use crate::skia::context::DrawContext;
use crate::spec::{CodecSpec, SourceSpec};
use crossbeam::channel::{Receiver, Sender};
use deepstream_encoders::NvEncoder;
use deepstream_nvbufsurface::SkiaRenderer;
use log::{debug, error, info, warn};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::WithAttributes;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Timeout (ms) given to the hardware encoder to flush remaining frames on
/// EOS / shutdown / codec hot-swap.
const ENCODER_DRAIN_TIMEOUT_MS: u64 = 3000;

/// Handle to a running source worker thread.
pub struct SourceWorker {
    tx: Sender<WorkerMessage>,
    pub(crate) last_activity: Arc<parking_lot::Mutex<Instant>>,
    alive: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SourceWorker {
    /// Spawn a new worker thread for the given source.
    pub fn spawn(
        source_id: String,
        spec: SourceSpec,
        callbacks: Arc<Callbacks>,
        idle_timeout: Duration,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::bounded::<WorkerMessage>(16);
        let last_activity = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let alive = Arc::new(AtomicBool::new(true));

        let la = last_activity.clone();
        let al = alive.clone();

        let thread = std::thread::Builder::new()
            .name(format!("picasso-{source_id}"))
            .spawn(move || {
                worker_loop(source_id, rx, spec, callbacks, idle_timeout, la, al);
            })
            .expect("failed to spawn picasso worker thread");

        Self {
            tx,
            last_activity,
            alive,
            thread: Some(thread),
        }
    }

    /// Send a message to this worker.
    #[allow(clippy::result_large_err)]
    pub fn send(
        &self,
        msg: WorkerMessage,
    ) -> Result<(), crossbeam::channel::SendError<WorkerMessage>> {
        self.tx.send(msg)
    }

    /// Whether the worker thread is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// Update the last-activity timestamp (called on eviction keep-for).
    pub fn touch(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    /// Get the elapsed time since last activity.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.lock().elapsed()
    }
}

impl Drop for SourceWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerMessage::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Per-worker mutable state used during frame processing.
struct WorkerState {
    source_id: String,
    spec: SourceSpec,
    encoder: Option<NvEncoder>,
    renderer: Option<SkiaRenderer>,
    callbacks: Arc<Callbacks>,
    frame_counter: u128,
    draw_ctx: DrawContext,
    /// Frames submitted to the encoder but not yet drained as encoded output.
    /// Keyed by `frame_id` (monotonic counter) so that encoded output can be
    /// matched back to the original [`VideoFrameProxy`].
    pending_frames: HashMap<u128, VideoFrameProxy>,
}

impl WorkerState {
    fn process_frame(&mut self, frame: VideoFrameProxy, buffer: gstreamer::Buffer) {
        if let Some((ns, name)) = &self.spec.conditional.encode_attribute {
            if frame.get_attribute(ns, name).is_none() {
                debug!("conditional skip (frame attr): source={}", self.source_id);
                return;
            }
        }

        self.frame_counter += 1;
        let frame_id = self.frame_counter;
        let input = FrameInput {
            frame,
            buffer,
            frame_id,
        };

        match &self.spec.codec {
            CodecSpec::Drop => {
                debug!("drop: source={}", self.source_id);
            }
            CodecSpec::Bypass => {
                bypass::process_bypass(&self.source_id, input.frame, input.buffer, &self.callbacks);
            }
            CodecSpec::Encode {
                transform,
                encoder: encoder_config,
            } => {
                let should_render =
                    if let Some((ns, name)) = &self.spec.conditional.render_attribute {
                        input.frame.get_attribute(ns, name).is_some()
                    } else {
                        self.spec.use_on_render || !self.spec.draw.is_empty()
                    };

                let gpu_id = encoder_config.gpu_id;
                let enc = ensure_encoder(&self.source_id, &mut self.encoder, encoder_config);
                if let Some(enc) = enc {
                    let mut render_opts = if should_render {
                        Some(RenderOpts {
                            draw_spec: &self.spec.draw,
                            use_on_render: self.spec.use_on_render,
                            gpu_id,
                            renderer: &mut self.renderer,
                            draw_ctx: &mut self.draw_ctx,
                        })
                    } else {
                        None
                    };

                    if let Err(e) = encode::process_encode(
                        &self.source_id,
                        input,
                        transform,
                        enc,
                        &self.callbacks,
                        self.spec.use_on_gpumat,
                        render_opts.as_mut(),
                        &mut self.pending_frames,
                    ) {
                        error!("encode error: source={}, err={e}", self.source_id);
                    }
                }
            }
        }
    }

    fn handle_eos(&mut self) {
        match &self.spec.codec {
            CodecSpec::Drop | CodecSpec::Bypass => {
                fire_eos_sentinel(&self.source_id, &self.callbacks);
            }
            CodecSpec::Encode { .. } => {
                if let Some(ref mut enc) = self.encoder {
                    drain_and_finish(
                        &self.source_id,
                        enc,
                        &self.callbacks,
                        &mut self.pending_frames,
                    );
                }
                fire_eos_sentinel(&self.source_id, &self.callbacks);
                self.encoder = None;
            }
        }
    }

    fn update_spec(&mut self, new_spec: SourceSpec) -> Option<Duration> {
        let codec_changed = codec_differs(&self.spec.codec, &new_spec.codec);
        if codec_changed {
            if let Some(ref mut enc) = self.encoder {
                drain_and_finish(
                    &self.source_id,
                    enc,
                    &self.callbacks,
                    &mut self.pending_frames,
                );
            }
            self.encoder = None;
            self.renderer = None;
        }
        if new_spec.font_family != self.draw_ctx.font_family() {
            self.draw_ctx = DrawContext::new(&new_spec.font_family);
        }
        self.draw_ctx.rebuild_template_cache(&new_spec.draw);
        let new_timeout = new_spec.idle_timeout_secs.map(Duration::from_secs);
        self.spec = new_spec;
        new_timeout
    }

    fn shutdown(&mut self) {
        if let Some(ref mut enc) = self.encoder {
            drain_and_finish(
                &self.source_id,
                enc,
                &self.callbacks,
                &mut self.pending_frames,
            );
        }
    }
}

fn worker_loop(
    source_id: String,
    rx: Receiver<WorkerMessage>,
    spec: SourceSpec,
    callbacks: Arc<Callbacks>,
    mut idle_timeout: Duration,
    last_activity: Arc<parking_lot::Mutex<Instant>>,
    alive: Arc<AtomicBool>,
) {
    let mut draw_ctx = DrawContext::new(&spec.font_family);
    draw_ctx.rebuild_template_cache(&spec.draw);
    let mut state = WorkerState {
        source_id: source_id.clone(),
        spec,
        encoder: None,
        renderer: None,
        callbacks,
        frame_counter: 0,
        draw_ctx,
        pending_frames: HashMap::new(),
    };

    info!("worker started: source={source_id}");

    loop {
        match rx.recv_timeout(idle_timeout) {
            Ok(WorkerMessage::Frame(frame, buffer)) => {
                *last_activity.lock() = Instant::now();
                state.process_frame(frame, buffer);
            }
            Ok(WorkerMessage::Eos) => {
                info!("EOS received: source={source_id}");
                state.handle_eos();
                state.renderer = None;
            }
            Ok(WorkerMessage::UpdateSpec(boxed_spec)) => {
                debug!("spec update: source={source_id}");
                if let Some(t) = state.update_spec(*boxed_spec) {
                    idle_timeout = t;
                }
            }
            Ok(WorkerMessage::Shutdown) => {
                info!("shutdown: source={source_id}");
                state.shutdown();
                break;
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                debug!("idle timeout: source={source_id}");
                if let Some(cb) = &state.callbacks.on_eviction {
                    let decision = cb.call(&source_id);
                    match decision {
                        crate::spec::EvictionDecision::KeepFor(secs) => {
                            idle_timeout = Duration::from_secs(secs);
                            *last_activity.lock() = Instant::now();
                            debug!("eviction: keep for {secs}s, source={source_id}");
                        }
                        crate::spec::EvictionDecision::Terminate => {
                            info!("eviction: terminate, source={source_id}");
                            state.handle_eos();
                            break;
                        }
                        crate::spec::EvictionDecision::TerminateImmediately => {
                            info!("eviction: terminate immediately, source={source_id}");
                            break;
                        }
                    }
                } else {
                    info!("eviction: no callback, terminating source={source_id}");
                    state.handle_eos();
                    break;
                }
            }
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                info!("channel disconnected: source={source_id}");
                break;
            }
        }
    }

    alive.store(false, Ordering::Relaxed);
    info!("worker exited: source={source_id}");
}

fn ensure_encoder<'a>(
    source_id: &str,
    encoder: &'a mut Option<NvEncoder>,
    config: &deepstream_encoders::EncoderConfig,
) -> Option<&'a mut NvEncoder> {
    if encoder.is_none() {
        match NvEncoder::new(config) {
            Ok(enc) => {
                info!("encoder created: source={source_id}");
                *encoder = Some(enc);
            }
            Err(e) => {
                error!("encoder creation failed: source={source_id}, err={e}");
                return None;
            }
        }
    }
    encoder.as_mut()
}

fn drain_and_finish(
    source_id: &str,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
    pending_frames: &mut HashMap<u128, VideoFrameProxy>,
) {
    match encoder.finish(Some(ENCODER_DRAIN_TIMEOUT_MS)) {
        Ok(remaining) => {
            for encoded in remaining {
                if let Some(cb) = &callbacks.on_encoded_frame {
                    if let Some(frame) = pending_frames.remove(&encoded.frame_id) {
                        encode::fill_encoded_frame(frame, encoded, cb);
                    } else {
                        warn!(
                            "drain: no pending frame for frame_id={}, source={source_id}",
                            encoded.frame_id
                        );
                    }
                }
            }
        }
        Err(e) => {
            error!("encoder finish error: source={source_id}, err={e}");
        }
    }
}

fn fire_eos_sentinel(source_id: &str, callbacks: &Arc<Callbacks>) {
    if let Some(cb) = &callbacks.on_encoded_frame {
        cb.call(EncodedOutput::EndOfStream(EndOfStream::new(source_id)));
    }
}

fn codec_differs(a: &CodecSpec, b: &CodecSpec) -> bool {
    match (a, b) {
        (CodecSpec::Drop, CodecSpec::Drop) => false,
        (CodecSpec::Bypass, CodecSpec::Bypass) => false,
        (CodecSpec::Encode { encoder: ea, .. }, CodecSpec::Encode { encoder: eb, .. }) => {
            ea.width != eb.width || ea.height != eb.height || ea.codec != eb.codec
        }
        _ => true,
    }
}
