use crate::callbacks::Callbacks;
use crate::message::{OutputMessage, WorkerMessage};
use crate::pipeline::encode::{DrainHandle, RenderOpts, SharedEncoder, SharedPendingFrames};
use crate::pipeline::{bypass, encode, FrameInput};
use crate::skia::context::DrawContext;
use crate::spec::{CodecSpec, SourceSpec};
use crossbeam::channel::{Receiver, Sender};
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::{create_cuda_stream, destroy_cuda_stream, SkiaRenderer};
use log::{debug, error, info, warn};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::WithAttributes;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Timeout (ms) given to the hardware encoder to flush remaining frames on
/// EOS / shutdown / codec hot-swap.
const ENCODER_DRAIN_TIMEOUT_MS: u64 = 5000;

/// Handle to a running source worker thread.
pub struct SourceWorker {
    tx: Sender<WorkerMessage>,
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
        queue_size: usize,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::bounded::<WorkerMessage>(queue_size);
        let alive = Arc::new(AtomicBool::new(true));
        let al = alive.clone();

        let thread = std::thread::Builder::new()
            .name(format!("picasso-{source_id}"))
            .spawn(move || {
                worker_loop(source_id, rx, spec, callbacks, idle_timeout, al);
            })
            .expect("failed to spawn picasso worker thread");

        Self {
            tx,
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
    encoder: Option<SharedEncoder>,
    renderer: Option<SkiaRenderer>,
    callbacks: Arc<Callbacks>,
    frame_counter: u128,
    draw_ctx: DrawContext,
    /// Frames submitted to the encoder but not yet drained as encoded output.
    /// Shared with the background [`DrainHandle`] thread.
    pending_frames: SharedPendingFrames,
    /// Background thread that pulls encoded output from the encoder.
    drain_handle: Option<DrainHandle>,
    /// Non-blocking CUDA stream owned by this worker, created alongside
    /// the encoder and destroyed when the encoder is stopped.
    cuda_stream: *mut std::ffi::c_void,
}

// Safety: cuda_stream is only used within the worker thread that created it
// and the same GPU context. The raw pointer is never shared across threads.
unsafe impl Send for WorkerState {}

impl WorkerState {
    fn process_frame(
        &mut self,
        frame: VideoFrameProxy,
        mut view: deepstream_nvbufsurface::SurfaceView,
        src_rect: Option<deepstream_nvbufsurface::Rect>,
    ) {
        if let Some((ns, name)) = &self.spec.conditional.encode_attribute {
            if frame.get_attribute(ns, name).is_none() {
                debug!("conditional skip (frame attr): source={}", self.source_id);
                return;
            }
        }

        {
            let buf_ref = view.buffer_mut().make_mut();
            crate::pipeline::apply_frame_timestamps_to_buffer(&frame, buf_ref);
        }

        self.frame_counter += 1;
        let frame_id = self.frame_counter;
        let input = FrameInput {
            frame,
            view,
            frame_id,
        };

        match &self.spec.codec {
            CodecSpec::Drop => {
                debug!("drop: source={}", self.source_id);
                return;
            }
            CodecSpec::Bypass => {
                bypass::process_bypass(&self.source_id, input.frame, input.view, &self.callbacks);
                return;
            }
            CodecSpec::Encode { .. } => {}
        }

        let CodecSpec::Encode {
            transform,
            encoder: encoder_config,
        } = &self.spec.codec
        else {
            unreachable!()
        };

        let should_render = if let Some((ns, name)) = &self.spec.conditional.render_attribute {
            input.frame.get_attribute(ns, name).is_some()
        } else {
            self.spec.use_on_render || !self.spec.draw.is_empty()
        };

        let gpu_id = encoder_config.gpu_id;
        let mut transform = transform.clone();
        let encoder_config = encoder_config.clone();
        self.ensure_encoder(&encoder_config);

        // Override the TransformConfig's CUDA stream with the worker's own stream.
        transform.cuda_stream = self.cuda_stream;

        if let Some(enc) = &self.encoder {
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
                &transform,
                enc,
                &self.callbacks,
                self.spec.use_on_gpumat,
                render_opts.as_mut(),
                &self.pending_frames,
                src_rect.as_ref(),
                self.spec.callback_order,
                self.cuda_stream,
            ) {
                error!("encode error: source={}, err={e}", self.source_id);
            }
        }
    }

    /// Create the encoder (and its drain thread) if not already present.
    fn ensure_encoder(&mut self, config: &deepstream_encoders::EncoderConfig) {
        if self.encoder.is_some() {
            return;
        }

        if self.cuda_stream.is_null() {
            match create_cuda_stream() {
                Ok(stream) => {
                    info!("CUDA stream created: source={}", self.source_id);
                    self.cuda_stream = stream;
                }
                Err(e) => {
                    error!(
                        "CUDA stream creation failed: source={}, err={e}",
                        self.source_id
                    );
                    return;
                }
            }
        }

        match NvEncoder::new(config) {
            Ok(enc) => {
                info!("encoder created: source={}", self.source_id);
                let shared: SharedEncoder = Arc::new(parking_lot::Mutex::new(enc));
                let drain = DrainHandle::spawn(
                    self.source_id.clone(),
                    shared.clone(),
                    self.callbacks.clone(),
                    self.pending_frames.clone(),
                );
                self.encoder = Some(shared);
                self.drain_handle = Some(drain);
            }
            Err(e) => {
                error!(
                    "encoder creation failed: source={}, err={e}",
                    self.source_id
                );
            }
        }
    }

    /// Stop the drain thread and flush remaining encoder output.
    fn stop_encoder(&mut self) {
        if let Some(mut drain) = self.drain_handle.take() {
            drain.stop();
        }
        if let Some(shared_enc) = self.encoder.take() {
            let mut enc = shared_enc.lock();
            let mut pending = self.pending_frames.lock();
            drain_and_finish(&self.source_id, &mut enc, &self.callbacks, &mut pending);
        }
        if !self.cuda_stream.is_null() {
            if let Err(e) = unsafe { destroy_cuda_stream(self.cuda_stream) } {
                warn!(
                    "Failed to destroy CUDA stream: source={}, err={e}",
                    self.source_id
                );
            } else {
                debug!("CUDA stream destroyed: source={}", self.source_id);
            }
            self.cuda_stream = std::ptr::null_mut();
        }
    }

    fn handle_eos(&mut self) {
        match &self.spec.codec {
            CodecSpec::Drop => {
                fire_eos_sentinel(&self.source_id, &self.callbacks);
            }
            CodecSpec::Bypass => {
                fire_bypass_eos_sentinel(&self.source_id, &self.callbacks);
            }
            CodecSpec::Encode { .. } => {
                self.stop_encoder();
                fire_eos_sentinel(&self.source_id, &self.callbacks);
            }
        }
    }

    fn update_spec(&mut self, new_spec: SourceSpec) -> Option<Duration> {
        let codec_changed = codec_differs(&self.spec.codec, &new_spec.codec);
        if codec_changed {
            self.stop_encoder();
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
        self.stop_encoder();
    }
}

fn worker_loop(
    source_id: String,
    rx: Receiver<WorkerMessage>,
    spec: SourceSpec,
    callbacks: Arc<Callbacks>,
    mut idle_timeout: Duration,
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
        pending_frames: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        drain_handle: None,
        cuda_stream: std::ptr::null_mut(),
    };

    info!("worker started: source={source_id}");

    loop {
        match rx.recv_timeout(idle_timeout) {
            Ok(WorkerMessage::Frame(frame, view, src_rect)) => {
                state.process_frame(frame, view, src_rect);
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

fn drain_and_finish(
    source_id: &str,
    encoder: &mut NvEncoder,
    callbacks: &Arc<Callbacks>,
    pending_frames: &mut HashMap<u128, VideoFrameProxy>,
) {
    encode::drain_remaining(source_id, encoder, callbacks, pending_frames);
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
        cb.call(OutputMessage::EndOfStream(EndOfStream::new(source_id)));
    }
}

fn fire_bypass_eos_sentinel(source_id: &str, callbacks: &Arc<Callbacks>) {
    if let Some(cb) = &callbacks.on_bypass_frame {
        cb.call(OutputMessage::EndOfStream(EndOfStream::new(source_id)));
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
