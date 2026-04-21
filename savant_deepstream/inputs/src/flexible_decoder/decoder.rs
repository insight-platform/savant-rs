//! [`FlexibleDecoder`] — single-stream adaptive decoder that wraps [`NvDecoder`]
//! and automatically handles codec/resolution changes.

use crate::codec_resolve::resolve_video_codec;
use deepstream_buffers::{
    BufferGenerator, CudaStream, Interpolation, NvBufSurfaceMemType, TransformConfig, VideoFormat,
};
use deepstream_decoders::{
    DecoderConfig, JpegBackend, NvDecoder, NvDecoderConfig, NvDecoderOutput,
};
use parking_lot::Mutex;
use savant_core::primitives::frame::{VideoFrameContent, VideoFrameProxy};
use savant_core::primitives::gstreamer_frame_time::frame_clock_ns;
use savant_core::primitives::video_codec::VideoCodec;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::config::FlexibleDecoderConfig;
use super::error::FlexibleDecoderError;
use super::handle_active::{handle_active, ActiveResult};
use super::handle_detecting::handle_detecting;
use super::handle_idle::handle_idle;
use super::output::{FlexibleDecoderOutput, SkipReason};
use super::state::{
    new_frame_map, ActivatedDecoder, DecoderState, FrameMap, StateGuard, SubmitContext,
};
use savant_core::utils::release_seal::ReleaseSeal;

/// Worker polls NvDecoder output at this interval.
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Single-stream adaptive GPU decoder.
///
/// Wraps [`NvDecoder`] and automatically manages codec/resolution changes:
/// when a parameter change is detected in incoming [`VideoFrameProxy`] metadata,
/// the old decoder is gracefully drained and a new one is created.
///
/// All output is delivered through a callback: decoded frames (paired with their
/// original [`VideoFrameProxy`]), parameter changes, skipped frames, source EOS,
/// and errors.
///
/// **Callback safety**: the user callback is never invoked while an internal
/// lock is held, so it is safe to call other `FlexibleDecoder` methods (or
/// shared state protected by user mutexes) from within the callback.
pub struct FlexibleDecoder {
    config: FlexibleDecoderConfig,
    state: Mutex<DecoderState>,
    on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync + 'static>,
    frame_map: FrameMap,
}

impl FlexibleDecoder {
    /// Construct a new decoder bound to `config.source_id`.
    ///
    /// No internal [`NvDecoder`] is created until the first successful
    /// [`submit`](Self::submit).
    pub fn new<F>(config: FlexibleDecoderConfig, on_output: F) -> Self
    where
        F: Fn(FlexibleDecoderOutput) + Send + Sync + 'static,
    {
        Self {
            config,
            state: Mutex::new(DecoderState::Idle),
            on_output: Arc::new(on_output),
            frame_map: new_frame_map(),
        }
    }

    /// Submit an encoded frame for decoding.
    ///
    /// Payload priority: `data` argument if `Some`, else
    /// `frame.get_content()` if `Internal(bytes)`. Missing payload emits
    /// `Skipped { NoPayload }`.
    ///
    /// **Blocks** when a parameter change requires draining the old decoder
    /// (bounded by `config.idle_timeout`).
    ///
    /// Returns `Ok(())` for soft rejections (source_id mismatch, unsupported
    /// codec, no payload, detection overflow) — those are delivered via
    /// callback. Returns `Err` only for infrastructure failures.
    pub fn submit(
        &self,
        frame: &VideoFrameProxy,
        data: Option<&[u8]>,
    ) -> Result<(), FlexibleDecoderError> {
        {
            let state = self.state.lock();
            if matches!(*state, DecoderState::ShutDown) {
                return Err(FlexibleDecoderError::ShutDown);
            }
        }

        let frame_source_id = frame.get_source_id();
        if frame_source_id != self.config.source_id {
            self.emit(FlexibleDecoderOutput::Skipped {
                frame: frame.clone(),
                data: None,
                reason: SkipReason::SourceIdMismatch {
                    expected: self.config.source_id.clone(),
                    actual: frame_source_id,
                },
            });
            return Ok(());
        }

        let video_codec = frame.get_codec();
        let width = frame.get_width();
        let height = frame.get_height();
        let frame_id = frame.get_uuid_u128();

        let clk = frame_clock_ns(frame);

        let payload = self.extract_payload(frame, data);
        let payload = match payload {
            Some(p) => p,
            None => {
                self.emit(FlexibleDecoderOutput::Skipped {
                    frame: frame.clone(),
                    data: None,
                    reason: SkipReason::NoPayload,
                });
                return Ok(());
            }
        };

        let fw = width.max(0) as u32;
        let fh = height.max(0) as u32;
        let resolve = match resolve_video_codec(video_codec, fw, fh) {
            Ok(r) => r,
            Err(None) => {
                self.emit(FlexibleDecoderOutput::Skipped {
                    frame: frame.clone(),
                    data: Some(payload),
                    reason: SkipReason::UnsupportedCodec(None),
                });
                return Ok(());
            }
            Err(Some(name)) => {
                self.emit(FlexibleDecoderOutput::Skipped {
                    frame: frame.clone(),
                    data: Some(payload),
                    reason: SkipReason::UnsupportedCodec(Some(name)),
                });
                return Ok(());
            }
        };

        if let Some(vc) = video_codec {
            if let Err(reason) = super::payload_validate::validate_payload(vc, &payload) {
                self.emit(FlexibleDecoderOutput::Skipped {
                    frame: frame.clone(),
                    data: Some(payload),
                    reason: SkipReason::InvalidPayload(reason),
                });
                return Ok(());
            }
        }

        // Locked section: lightweight outputs (Skipped, SourceEos, etc.) are
        // collected into `pending` and emitted after the state lock is released.
        //
        // Drain operations (graceful_shutdown on the NvDecoder) MUST happen
        // outside the lock because the drain callback delivers
        // FlexibleDecoderOutput::Frame which holds a GPU buffer from the pool.
        // Holding those buffers in a vec while the pipeline tries to decode more
        // frames would exhaust the fixed-size pool and deadlock.
        let mut pending = Vec::new();

        let mut resolve_opt = Some(resolve);

        let activate_fn =
            |cfg, codec, w, h, f: &VideoFrameProxy| self.activate(cfg, codec, w, h, f);
        let ctx = SubmitContext {
            video_codec,
            width,
            height,
            frame,
            payload: &payload,
            frame_id,
            clk: &clk,
        };

        // Phase 1 — may need to tear down an old Active session outside the lock.
        #[allow(clippy::type_complexity)]
        let drain_job: Option<(
            Arc<NvDecoder>,
            super::output::DecoderParameters,
            super::output::DecoderParameters,
        )>;

        let result = {
            let mut state = self.state.lock();
            let (guard, taken) = StateGuard::take(&mut state);

            match taken {
                DecoderState::ShutDown => {
                    guard.commit(DecoderState::ShutDown);
                    drain_job = None;
                    Err(FlexibleDecoderError::ShutDown)
                }

                DecoderState::Idle => {
                    drain_job = None;
                    handle_idle(
                        guard,
                        &mut pending,
                        &self.frame_map,
                        resolve_opt.take().unwrap(),
                        &ctx,
                        &activate_fn,
                    )
                }

                DecoderState::Detecting {
                    strategy: det_strategy,
                    video_codec: det_video_codec,
                    width: det_width,
                    height: det_height,
                    mut buffered,
                } => {
                    drain_job = None;
                    let codec_changed = video_codec != Some(det_video_codec);
                    let dims_changed = width != det_width || height != det_height;

                    if codec_changed || dims_changed {
                        for pkt in buffered.drain(..) {
                            pending.push(FlexibleDecoderOutput::Skipped {
                                frame: pkt.frame,
                                data: Some(pkt.data),
                                reason: SkipReason::WaitingForKeyframe,
                            });
                        }
                        handle_idle(
                            guard,
                            &mut pending,
                            &self.frame_map,
                            resolve_opt.take().unwrap(),
                            &ctx,
                            &activate_fn,
                        )
                    } else {
                        handle_detecting(
                            guard,
                            &mut pending,
                            &self.frame_map,
                            self.config.detect_buffer_limit,
                            det_strategy,
                            det_video_codec,
                            det_width,
                            det_height,
                            buffered,
                            &ctx,
                            &activate_fn,
                        )
                    }
                }

                DecoderState::Active {
                    decoder,
                    worker_join,
                    worker_stop,
                    gst_codec: active_gst_codec,
                    video_codec: active_video_codec,
                    width: active_width,
                    height: active_height,
                } => {
                    let resolve_ref = resolve_opt.as_ref().unwrap();
                    match handle_active(
                        guard,
                        &self.frame_map,
                        decoder,
                        worker_join,
                        worker_stop,
                        active_gst_codec,
                        active_video_codec,
                        active_width,
                        active_height,
                        resolve_ref,
                        &ctx,
                    ) {
                        ActiveResult::SteadyState(r) => {
                            drain_job = None;
                            r
                        }
                        ActiveResult::NeedDrain {
                            old_decoder,
                            old_params,
                            new_params,
                        } => {
                            drain_job = Some((old_decoder, old_params, new_params));
                            Ok(())
                        }
                    }
                }
            }
        };
        // State lock released — guard consumed by commit in every arm.

        // Phase 2 — drain the old decoder *outside* the lock so that the
        // callback immediately frees GPU buffers back to the pool.
        let result = if let Some((old_decoder, old_params, new_params)) = drain_job {
            let on_output = Arc::clone(&self.on_output);
            let fm = Arc::clone(&self.frame_map);
            let _ = old_decoder.graceful_shutdown(Some(self.config.idle_timeout), |out| {
                (on_output)(convert_output(&fm, out));
            });

            let mut state = self.state.lock();
            let (guard2, _taken) = StateGuard::take(&mut state);
            let handle_result = handle_idle(
                guard2,
                &mut pending,
                &self.frame_map,
                resolve_opt.take().unwrap(),
                &ctx,
                &activate_fn,
            );

            if handle_result.is_ok() {
                pending.push(FlexibleDecoderOutput::ParameterChange {
                    old: old_params,
                    new: new_params,
                });
            }
            handle_result
        } else {
            result
        };

        self.emit_all(pending);
        result
    }

    /// Inject a logical per-source EOS ordered with decoded frames.
    ///
    /// Maps to [`NvDecoder::send_source_eos`]. The decoder remains alive.
    /// If no decoder is active (Idle/Detecting), emits `SourceEos` directly
    /// through the callback.
    pub fn source_eos(&self, source_id: &str) -> Result<(), FlexibleDecoderError> {
        let mut pending = Vec::new();
        {
            let state = self.state.lock();
            match &*state {
                DecoderState::ShutDown => return Err(FlexibleDecoderError::ShutDown),
                DecoderState::Active { decoder, .. } => {
                    decoder.send_source_eos(source_id)?;
                }
                DecoderState::Idle | DecoderState::Detecting { .. } => {
                    pending.push(FlexibleDecoderOutput::SourceEos {
                        source_id: source_id.to_string(),
                    });
                }
            }
        }
        self.emit_all(pending);
        Ok(())
    }

    /// Force-flush pending rescue-eligible custom-downstream events
    /// inside the internal [`NvDecoder`] when it is idle (no buffers in
    /// flight).
    ///
    /// Typical use: call from the consumer's `recv_timeout(Timeout)`
    /// branch so a trailing [`source_eos`](Self::source_eos) marker
    /// submitted after the last frame is delivered to the output
    /// callback without requiring a full
    /// [`graceful_shutdown`](Self::graceful_shutdown).
    ///
    /// Returns the number of events actually flushed.  When the decoder
    /// is in `Idle` / `Detecting` state, there is no inner decoder to
    /// flush (the [`source_eos`](Self::source_eos) fast-path already
    /// emits `SourceEos` directly through the callback in those states),
    /// so this returns `Ok(0)`.
    ///
    /// See [`savant_gstreamer::pipeline::GstPipeline::flush_idle`] for
    /// full semantics and the race note.
    pub fn flush_idle(&self) -> Result<usize, FlexibleDecoderError> {
        let state = self.state.lock();
        match &*state {
            DecoderState::ShutDown => Err(FlexibleDecoderError::ShutDown),
            DecoderState::Active { decoder, .. } => Ok(decoder.flush_idle()?),
            DecoderState::Idle | DecoderState::Detecting { .. } => Ok(0),
        }
    }

    /// Drain the current internal decoder, forwarding all remaining frames
    /// to the callback (bounded by `config.idle_timeout`), then tear down.
    ///
    /// Terminal — subsequent [`submit`](Self::submit) / [`source_eos`](Self::source_eos)
    /// calls return `Err(ShutDown)`.
    pub fn graceful_shutdown(&self) -> Result<(), FlexibleDecoderError> {
        let mut pending = Vec::new();
        let drain_decoder = {
            let mut state = self.state.lock();
            let (guard, taken) = StateGuard::take(&mut state);
            match taken {
                DecoderState::ShutDown => {
                    guard.commit(DecoderState::ShutDown);
                    return Err(FlexibleDecoderError::ShutDown);
                }
                DecoderState::Active {
                    decoder,
                    worker_join,
                    worker_stop,
                    ..
                } => {
                    worker_stop.store(true, Ordering::Relaxed);
                    if let Some(jh) = worker_join {
                        let _ = jh.join();
                    }
                    guard.commit(DecoderState::ShutDown);
                    Some(decoder)
                }
                DecoderState::Detecting { buffered, .. } => {
                    for pkt in buffered {
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: pkt.frame,
                            data: Some(pkt.data),
                            reason: SkipReason::WaitingForKeyframe,
                        });
                    }
                    guard.commit(DecoderState::ShutDown);
                    None
                }
                DecoderState::Idle => {
                    guard.commit(DecoderState::ShutDown);
                    None
                }
            }
        };
        // State lock released. Drain outside the lock so the callback
        // immediately frees GPU buffers back to the pool.
        if let Some(decoder) = drain_decoder {
            let on_output = Arc::clone(&self.on_output);
            let fm = Arc::clone(&self.frame_map);
            let _ = decoder.graceful_shutdown(Some(self.config.idle_timeout), |out| {
                (on_output)(convert_output(&fm, out));
            });
        }
        self.emit_all(pending);
        Ok(())
    }

    /// Immediate teardown — frames in flight are lost.
    ///
    /// Terminal — subsequent [`submit`](Self::submit) / [`source_eos`](Self::source_eos)
    /// calls return `Err(ShutDown)`.
    pub fn shutdown(&self) {
        let mut state = self.state.lock();
        let (guard, taken) = StateGuard::take(&mut state);
        match taken {
            DecoderState::ShutDown => {
                guard.commit(DecoderState::ShutDown);
            }
            DecoderState::Active {
                decoder,
                worker_join,
                worker_stop,
                ..
            } => {
                worker_stop.store(true, Ordering::Relaxed);
                let _ = decoder.shutdown();
                if let Some(jh) = worker_join {
                    let _ = jh.join();
                }
                self.frame_map.lock().clear();
                guard.commit(DecoderState::ShutDown);
            }
            DecoderState::Detecting { .. } | DecoderState::Idle => {
                guard.commit(DecoderState::ShutDown);
            }
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn emit(&self, output: FlexibleDecoderOutput) {
        (self.on_output)(output);
    }

    fn emit_all(&self, pending: Vec<FlexibleDecoderOutput>) {
        for out in pending {
            (self.on_output)(out);
        }
    }

    fn extract_payload(&self, frame: &VideoFrameProxy, data: Option<&[u8]>) -> Option<Vec<u8>> {
        if let Some(b) = data {
            return Some(b.to_vec());
        }
        match frame.get_content().as_ref() {
            VideoFrameContent::Internal(v) => Some(v.clone()),
            VideoFrameContent::External(_) | VideoFrameContent::None => None,
        }
    }

    /// Apply the user-supplied decoder-config callback (if any).
    ///
    /// The callback takes ownership of the incoming [`DecoderConfig`] and
    /// returns a (possibly modified) config. When no callback is installed,
    /// this is the identity function.
    #[inline]
    fn apply_decoder_config_callback(
        &self,
        decoder_config: DecoderConfig,
        frame: &VideoFrameProxy,
    ) -> DecoderConfig {
        match &self.config.decoder_config_callback {
            Some(cb) => cb(decoder_config, frame),
            None => decoder_config,
        }
    }

    /// Create a new [`NvDecoder`], its buffer pool, and a worker thread.
    fn activate(
        &self,
        decoder_config: DecoderConfig,
        gst_codec: VideoCodec,
        width: i64,
        height: i64,
        frame: &VideoFrameProxy,
    ) -> Result<ActivatedDecoder, String> {
        let decoder_config = self.apply_decoder_config_callback(decoder_config, frame);

        if let DecoderConfig::Jpeg(ref j) = decoder_config {
            if j.backend == JpegBackend::Gpu
                && (width < 16 || height < 16 || width % 8 != 0 || height % 8 != 0)
            {
                return Err(format!(
                    "hardware JPEG requires 8-aligned dims >= 16x16, got {width}x{height}"
                ));
            }
        }

        let (pool_w, pool_h, fps_num, fps_den) = stream_pool_params(frame)?;

        let pool = BufferGenerator::builder(VideoFormat::RGBA, pool_w, pool_h)
            .fps(fps_num, fps_den)
            .gpu_id(self.config.gpu_id)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(self.config.pool_size)
            .max_buffers(self.config.pool_size)
            .build()
            .map_err(|e| format!("buffer pool creation failed: {e}"))?;

        let cuda_stream = CudaStream::new_non_blocking()
            .map_err(|e| format!("CUDA stream creation failed: {e}"))?;
        let transform = TransformConfig {
            interpolation: Interpolation::Nearest,
            cuda_stream,
            ..TransformConfig::default()
        };

        let nv_config = NvDecoderConfig::new(self.config.gpu_id, decoder_config).name(format!(
            "{}-{}",
            self.config.source_id,
            gst_codec.name()
        ));

        let decoder = NvDecoder::new(nv_config, pool, transform)
            .map_err(|e| format!("NvDecoder creation failed: {e}"))?;
        let decoder = Arc::new(decoder);

        let stop = Arc::new(AtomicBool::new(false));
        let on_output = Arc::clone(&self.on_output);
        let fm = Arc::clone(&self.frame_map);
        let dec_clone = Arc::clone(&decoder);
        let stop_clone = Arc::clone(&stop);
        let source_id = self.config.source_id.clone();

        let worker_join = std::thread::Builder::new()
            .name(format!("flex-dec-{source_id}"))
            .spawn(move || worker_loop(dec_clone, on_output, fm, stop_clone))
            .map_err(|e| format!("failed to spawn worker thread: {e}"))?;

        Ok((decoder, worker_join, stop))
    }
}

// ── Free helpers ─────────────────────────────────────────────────────

/// Extract positive `(width, height)` and `(fps_num, fps_den)` from a [`VideoFrameProxy`].
fn stream_pool_params(frame: &VideoFrameProxy) -> Result<(u32, u32, i32, i32), String> {
    let w = frame.get_width();
    let h = frame.get_height();
    if w <= 0 || h <= 0 || w > u32::MAX as i64 || h > u32::MAX as i64 {
        return Err(format!("invalid frame dimensions: {w}x{h}"));
    }
    let (n, d) = frame.get_fps();
    let (n, d) = (n as i32, d as i32);
    let (fps_num, fps_den) = if n > 0 && d > 0 { (n, d) } else { (30, 1) };
    Ok((w as u32, h as u32, fps_num, fps_den))
}

/// Look up and remove the [`VideoFrameProxy`] for a decoded frame.
fn take_frame_proxy(fm: &FrameMap, frame_id: Option<u128>) -> Option<VideoFrameProxy> {
    frame_id.and_then(|id| fm.lock().remove(&id))
}

/// Convert an [`NvDecoderOutput`] to a [`FlexibleDecoderOutput`].
fn convert_output(fm: &FrameMap, out: NvDecoderOutput) -> FlexibleDecoderOutput {
    match out {
        NvDecoderOutput::Frame(df) => {
            if let Some(proxy) = take_frame_proxy(fm, df.frame_id) {
                FlexibleDecoderOutput::Frame {
                    frame: proxy,
                    decoded: df,
                    seal: Arc::new(ReleaseSeal::new()),
                }
            } else {
                FlexibleDecoderOutput::OrphanFrame { decoded: df }
            }
        }
        NvDecoderOutput::SourceEos { source_id } => FlexibleDecoderOutput::SourceEos { source_id },
        NvDecoderOutput::Error(e) => FlexibleDecoderOutput::Error(e),
        NvDecoderOutput::Event(e) => FlexibleDecoderOutput::Event(e),
        NvDecoderOutput::Eos => {
            FlexibleDecoderOutput::Error(deepstream_decoders::DecoderError::PipelineError(
                "unexpected stream-level EOS during drain".into(),
            ))
        }
    }
}

#[cfg(test)]
impl FlexibleDecoder {
    /// Test-only hook: exercise the decoder-config callback path without
    /// spinning up an [`NvDecoder`].
    pub(crate) fn apply_decoder_config_callback_for_test(
        &self,
        cfg: DecoderConfig,
        frame: &VideoFrameProxy,
    ) -> DecoderConfig {
        self.apply_decoder_config_callback(cfg, frame)
    }
}

/// Background worker: polls [`NvDecoder`] output and forwards to the callback.
fn worker_loop(
    decoder: Arc<NvDecoder>,
    on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync>,
    fm: FrameMap,
    stop: Arc<AtomicBool>,
) {
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        match decoder.recv_timeout(WORKER_POLL_INTERVAL) {
            Ok(Some(NvDecoderOutput::Frame(df))) => {
                if let Some(proxy) = take_frame_proxy(&fm, df.frame_id) {
                    on_output(FlexibleDecoderOutput::Frame {
                        frame: proxy,
                        decoded: df,
                        seal: Arc::new(ReleaseSeal::new()),
                    });
                } else {
                    on_output(FlexibleDecoderOutput::OrphanFrame { decoded: df });
                }
            }
            Ok(Some(NvDecoderOutput::SourceEos { source_id })) => {
                on_output(FlexibleDecoderOutput::SourceEos { source_id });
            }
            Ok(Some(NvDecoderOutput::Error(e))) => {
                on_output(FlexibleDecoderOutput::Error(e));
                break;
            }
            Ok(Some(NvDecoderOutput::Event(e))) => {
                on_output(FlexibleDecoderOutput::Event(e));
            }
            Ok(Some(NvDecoderOutput::Eos)) => {
                on_output(FlexibleDecoderOutput::Error(
                    deepstream_decoders::DecoderError::PipelineError(
                        "unexpected stream-level EOS".into(),
                    ),
                ));
                break;
            }
            Ok(None) => {}
            Err(e) => {
                on_output(FlexibleDecoderOutput::Error(e));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_decoders::Vp8DecoderConfig;
    use savant_core::primitives::frame::{VideoFrameContent, VideoFrameTranscodingMethod};
    use savant_core::primitives::video_codec::VideoCodec;

    fn test_frame() -> VideoFrameProxy {
        VideoFrameProxy::new(
            "src",
            (30, 1),
            64,
            48,
            VideoFrameContent::None,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::Vp8),
            None,
            (1, 30),
            0,
            None,
            None,
        )
        .expect("test frame")
    }

    #[test]
    fn callback_none_is_identity() {
        let config = FlexibleDecoderConfig::new("src", 0, 4);
        let dec = FlexibleDecoder::new(config, |_| {});
        let input = DecoderConfig::Vp8(Vp8DecoderConfig::default().num_extra_surfaces(1));
        let out = dec.apply_decoder_config_callback_for_test(input, &test_frame());
        match out {
            DecoderConfig::Vp8(c) => assert_eq!(c.num_extra_surfaces, Some(1)),
            _ => panic!("expected Vp8"),
        }
    }

    #[test]
    fn callback_transforms_vp8_num_extra_surfaces() {
        let config = FlexibleDecoderConfig::new("src", 0, 4).decoder_config_callback(|cfg, f| {
            assert_eq!(f.get_source_id(), "src");
            match cfg {
                DecoderConfig::Vp8(c) => DecoderConfig::Vp8(c.num_extra_surfaces(7)),
                other => other,
            }
        });
        let dec = FlexibleDecoder::new(config, |_| {});
        let input = DecoderConfig::Vp8(Vp8DecoderConfig::default());
        let out = dec.apply_decoder_config_callback_for_test(input, &test_frame());
        match out {
            DecoderConfig::Vp8(c) => assert_eq!(c.num_extra_surfaces, Some(7)),
            _ => panic!("expected Vp8"),
        }
    }
}
