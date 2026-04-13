//! [`FlexibleDecoder`] — single-stream adaptive decoder that wraps [`NvDecoder`]
//! and automatically handles codec/resolution changes.

use crate::codec_resolve::{resolve_video_codec, CodecResolve};
use deepstream_buffers::{
    BufferGenerator, CudaStream, Interpolation, NvBufSurfaceMemType, TransformConfig, VideoFormat,
};
use deepstream_decoders::{
    detect_stream_config, is_random_access_point, DecoderConfig, JpegBackend, NvDecoder,
    NvDecoderConfig, NvDecoderOutput,
};
use log::warn;
use parking_lot::Mutex;
use savant_core::primitives::frame::{VideoFrameContent, VideoFrameProxy};
use savant_core::primitives::gstreamer_frame_time::frame_clock_ns;
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::Codec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use super::config::FlexibleDecoderConfig;
use super::error::FlexibleDecoderError;
use super::output::{DecoderParameters, FlexibleDecoderOutput, SkipReason};

/// Worker polls NvDecoder output at this interval.
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Shared map from frame UUID to the original [`VideoFrameProxy`].
///
/// Populated by [`FlexibleDecoder::submit`] when a packet is handed to
/// [`NvDecoder`], consumed by the worker thread / drain callback when the
/// decoded frame comes back.
type FrameMap = Arc<Mutex<HashMap<u128, VideoFrameProxy>>>;

/// Tuple returned from [`FlexibleDecoder::activate`].
type ActivatedDecoder = (Arc<NvDecoder>, JoinHandle<()>, Arc<AtomicBool>);

/// Packet buffered during H.264/HEVC stream detection.
struct BufferedPacket {
    frame: VideoFrameProxy,
    frame_id: u128,
    data: Vec<u8>,
    pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
}

/// Internal decoder lifecycle state.
enum DecoderState {
    Idle,
    Detecting {
        gst_codec: Codec,
        video_codec: VideoCodec,
        width: i64,
        height: i64,
        buffered: Vec<BufferedPacket>,
    },
    Active {
        decoder: Arc<NvDecoder>,
        worker_join: Option<JoinHandle<()>>,
        worker_stop: Arc<AtomicBool>,
        gst_codec: Codec,
        video_codec: VideoCodec,
        width: i64,
        height: i64,
    },
    ShutDown,
}

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
            frame_map: Arc::new(Mutex::new(HashMap::new())),
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

        // Locked section: lightweight outputs (Skipped, SourceEos, etc.) are
        // collected into `pending` and emitted after the state lock is released.
        //
        // Drain operations (graceful_shutdown on the NvDecoder) MUST happen
        // outside the lock because the drain callback delivers
        // FlexibleDecoderOutput::Frame which holds a GPU buffer from the pool.
        // Holding those buffers in a vec while the pipeline tries to decode more
        // frames would exhaust the fixed-size pool and deadlock.
        let mut pending = Vec::new();

        // `resolve` is wrapped in Option so that the Active param-change arm
        // can defer its consumption to phase 2 (outside the state lock).
        let mut resolve_opt = Some(resolve);

        // Phase 1 — may need to tear down an old Active session outside the lock.
        #[allow(clippy::type_complexity)]
        let drain_job: Option<(Arc<NvDecoder>, DecoderParameters, DecoderParameters)>;

        let result = {
            let mut state = self.state.lock();

            match std::mem::replace(&mut *state, DecoderState::Idle) {
                DecoderState::ShutDown => {
                    *state = DecoderState::ShutDown;
                    drain_job = None;
                    Err(FlexibleDecoderError::ShutDown)
                }

                DecoderState::Idle => {
                    drain_job = None;
                    self.handle_idle(
                        &mut state,
                        &mut pending,
                        resolve_opt.take().unwrap(),
                        video_codec,
                        width,
                        height,
                        frame,
                        &payload,
                        frame_id,
                        &clk,
                    )
                }

                DecoderState::Detecting {
                    gst_codec: det_gst_codec,
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
                        self.handle_idle(
                            &mut state,
                            &mut pending,
                            resolve_opt.take().unwrap(),
                            video_codec,
                            width,
                            height,
                            frame,
                            &payload,
                            frame_id,
                            &clk,
                        )
                    } else {
                        self.handle_detecting(
                            &mut state,
                            &mut pending,
                            det_gst_codec,
                            det_video_codec,
                            det_width,
                            det_height,
                            buffered,
                            &payload,
                            frame_id,
                            &clk,
                            frame,
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
                    let codec_changed = video_codec != Some(active_video_codec);
                    let dims_changed = width != active_width || height != active_height;

                    if codec_changed || dims_changed {
                        // Stop the worker under the lock (worker never holds
                        // the state lock, so no deadlock).
                        worker_stop.store(true, Ordering::Relaxed);
                        if let Some(jh) = worker_join {
                            let _ = jh.join();
                        }

                        let old_params = DecoderParameters {
                            codec: active_gst_codec,
                            width: active_width,
                            height: active_height,
                        };
                        let resolve_ref = resolve_opt.as_ref().unwrap();
                        let new_gst_codec = match resolve_ref {
                            CodecResolve::Ready(cfg) => cfg.codec(),
                            CodecResolve::NeedDetection { codec } => *codec,
                        };
                        let new_params = DecoderParameters {
                            codec: new_gst_codec,
                            width,
                            height,
                        };

                        // Drain + new-session activation happen after the lock
                        // is released (state is Idle from the mem::replace).
                        drain_job = Some((decoder, old_params, new_params));
                        Ok(()) // placeholder; real result computed in phase 2
                    } else {
                        drain_job = None;
                        // Steady-state: same codec/resolution — submit directly.
                        self.register_frame(frame_id, frame);
                        match decoder.submit_packet(
                            &payload,
                            frame_id,
                            clk.submission_order_ns,
                            clk.dts_ns,
                            clk.duration_ns,
                        ) {
                            Ok(()) => {
                                *state = DecoderState::Active {
                                    decoder,
                                    worker_join,
                                    worker_stop,
                                    gst_codec: active_gst_codec,
                                    video_codec: active_video_codec,
                                    width: active_width,
                                    height: active_height,
                                };
                                Ok(())
                            }
                            Err(e) => {
                                // Decoder is still valid; restore Active state.
                                *state = DecoderState::Active {
                                    decoder,
                                    worker_join,
                                    worker_stop,
                                    gst_codec: active_gst_codec,
                                    video_codec: active_video_codec,
                                    width: active_width,
                                    height: active_height,
                                };
                                self.frame_map.lock().remove(&frame_id);
                                Err(e.into())
                            }
                        }
                    }
                }
            }
        };
        // State lock released.

        // Phase 2 — drain the old decoder *outside* the lock so that the
        // callback immediately frees GPU buffers back to the pool.
        let result = if let Some((old_decoder, old_params, new_params)) = drain_job {
            let on_output = Arc::clone(&self.on_output);
            let fm = Arc::clone(&self.frame_map);
            let _ = old_decoder.graceful_shutdown(Some(self.config.idle_timeout), |out| {
                if let Some(fout) = convert_output(&fm, out) {
                    (on_output)(fout);
                }
            });

            // Re-acquire the lock to start the new session.
            let mut state = self.state.lock();
            let handle_result = self.handle_idle(
                &mut state,
                &mut pending,
                resolve_opt.take().unwrap(),
                video_codec,
                width,
                height,
                frame,
                &payload,
                frame_id,
                &clk,
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

    /// Drain the current internal decoder, forwarding all remaining frames
    /// to the callback (bounded by `config.idle_timeout`), then tear down.
    ///
    /// Terminal — subsequent [`submit`](Self::submit) / [`source_eos`](Self::source_eos)
    /// calls return `Err(ShutDown)`.
    pub fn graceful_shutdown(&mut self) -> Result<(), FlexibleDecoderError> {
        let mut pending = Vec::new();
        let drain_decoder = {
            let mut state = self.state.lock();
            match std::mem::replace(&mut *state, DecoderState::ShutDown) {
                DecoderState::ShutDown => return Err(FlexibleDecoderError::ShutDown),
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
                    None
                }
                DecoderState::Idle => None,
            }
        };
        // State lock released. Drain outside the lock so the callback
        // immediately frees GPU buffers back to the pool.
        if let Some(decoder) = drain_decoder {
            let on_output = Arc::clone(&self.on_output);
            let fm = Arc::clone(&self.frame_map);
            let _ = decoder.graceful_shutdown(Some(self.config.idle_timeout), |out| {
                if let Some(fout) = convert_output(&fm, out) {
                    (on_output)(fout);
                }
            });
        }
        self.emit_all(pending);
        Ok(())
    }

    /// Immediate teardown — frames in flight are lost.
    ///
    /// Terminal — subsequent [`submit`](Self::submit) / [`source_eos`](Self::source_eos)
    /// calls return `Err(ShutDown)`.
    pub fn shutdown(&mut self) {
        let mut state = self.state.lock();
        match std::mem::replace(&mut *state, DecoderState::ShutDown) {
            DecoderState::ShutDown => {}
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
            }
            DecoderState::Detecting { .. } | DecoderState::Idle => {}
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

    fn register_frame(&self, frame_id: u128, frame: &VideoFrameProxy) {
        self.frame_map.lock().insert(frame_id, frame.clone());
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

    /// Handle a submit when the decoder is in `Idle` state.
    ///
    /// On success, `*state` is set to `Active` or `Detecting`.
    /// On failure, `*state` is set to `Idle` and a skip output may be pushed.
    /// Infrastructure errors (e.g. `submit_packet` failure after activation)
    /// are returned as `Err` after tearing down the partially-created session.
    #[allow(clippy::too_many_arguments)]
    fn handle_idle(
        &self,
        state: &mut DecoderState,
        pending: &mut Vec<FlexibleDecoderOutput>,
        resolve: CodecResolve,
        video_codec: Option<VideoCodec>,
        width: i64,
        height: i64,
        frame: &VideoFrameProxy,
        payload: &[u8],
        frame_id: u128,
        clk: &savant_core::primitives::gstreamer_frame_time::FrameClockNs,
    ) -> Result<(), FlexibleDecoderError> {
        match resolve {
            CodecResolve::Ready(decoder_config) => {
                let gst_codec = decoder_config.codec();
                match self.activate(decoder_config, gst_codec, width, height, frame) {
                    Ok((decoder, worker_join, worker_stop)) => {
                        let vc = match video_codec {
                            Some(vc) => vc,
                            None => {
                                teardown_activated(&decoder, worker_join, &worker_stop);
                                pending.push(FlexibleDecoderOutput::Skipped {
                                    frame: frame.clone(),
                                    data: Some(payload.to_vec()),
                                    reason: SkipReason::UnsupportedCodec(None),
                                });
                                *state = DecoderState::Idle;
                                return Ok(());
                            }
                        };
                        self.register_frame(frame_id, frame);
                        match decoder.submit_packet(
                            payload,
                            frame_id,
                            clk.submission_order_ns,
                            clk.dts_ns,
                            clk.duration_ns,
                        ) {
                            Ok(()) => {
                                *state = DecoderState::Active {
                                    decoder,
                                    worker_join: Some(worker_join),
                                    worker_stop,
                                    gst_codec,
                                    video_codec: vc,
                                    width,
                                    height,
                                };
                                Ok(())
                            }
                            Err(e) => {
                                teardown_activated(&decoder, worker_join, &worker_stop);
                                self.frame_map.lock().remove(&frame_id);
                                *state = DecoderState::Idle;
                                Err(e.into())
                            }
                        }
                    }
                    Err(msg) => {
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: frame.clone(),
                            data: Some(payload.to_vec()),
                            reason: SkipReason::DecoderCreationFailed(msg),
                        });
                        *state = DecoderState::Idle;
                        Ok(())
                    }
                }
            }
            CodecResolve::NeedDetection { codec: gst_codec } => {
                let vc = match video_codec {
                    Some(vc) => vc,
                    None => {
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: frame.clone(),
                            data: Some(payload.to_vec()),
                            reason: SkipReason::UnsupportedCodec(None),
                        });
                        *state = DecoderState::Idle;
                        return Ok(());
                    }
                };
                let mut buffered = Vec::new();

                if is_random_access_point(gst_codec, payload) {
                    if let Some(cfg) = detect_stream_config(gst_codec, payload) {
                        let real_gst_codec = cfg.codec();
                        match self.activate(cfg, real_gst_codec, width, height, frame) {
                            Ok((decoder, worker_join, worker_stop)) => {
                                self.register_frame(frame_id, frame);
                                match decoder.submit_packet(
                                    payload,
                                    frame_id,
                                    clk.submission_order_ns,
                                    clk.dts_ns,
                                    clk.duration_ns,
                                ) {
                                    Ok(()) => {
                                        *state = DecoderState::Active {
                                            decoder,
                                            worker_join: Some(worker_join),
                                            worker_stop,
                                            gst_codec: real_gst_codec,
                                            video_codec: vc,
                                            width,
                                            height,
                                        };
                                        return Ok(());
                                    }
                                    Err(e) => {
                                        teardown_activated(&decoder, worker_join, &worker_stop);
                                        self.frame_map.lock().remove(&frame_id);
                                        *state = DecoderState::Idle;
                                        return Err(e.into());
                                    }
                                }
                            }
                            Err(msg) => {
                                pending.push(FlexibleDecoderOutput::Skipped {
                                    frame: frame.clone(),
                                    data: Some(payload.to_vec()),
                                    reason: SkipReason::DecoderCreationFailed(msg),
                                });
                                *state = DecoderState::Idle;
                                return Ok(());
                            }
                        }
                    }
                }

                buffered.push(BufferedPacket {
                    frame: frame.clone(),
                    frame_id,
                    data: payload.to_vec(),
                    pts_ns: clk.submission_order_ns,
                    dts_ns: clk.dts_ns,
                    duration_ns: clk.duration_ns,
                });
                *state = DecoderState::Detecting {
                    gst_codec,
                    video_codec: vc,
                    width,
                    height,
                    buffered,
                };
                Ok(())
            }
        }
    }

    /// Handle a submit when the decoder is in `Detecting` state.
    ///
    /// On activation, replays buffered packets. If any replay submission fails,
    /// the entire activation is aborted: the decoder is torn down, all buffered
    /// packets (plus the current one) are emitted as `Skipped`, and the state
    /// reverts to `Idle`.
    #[allow(clippy::too_many_arguments)]
    fn handle_detecting(
        &self,
        state: &mut DecoderState,
        pending: &mut Vec<FlexibleDecoderOutput>,
        gst_codec: Codec,
        video_codec: VideoCodec,
        width: i64,
        height: i64,
        mut buffered: Vec<BufferedPacket>,
        payload: &[u8],
        frame_id: u128,
        clk: &savant_core::primitives::gstreamer_frame_time::FrameClockNs,
        frame: &VideoFrameProxy,
    ) -> Result<(), FlexibleDecoderError> {
        if is_random_access_point(gst_codec, payload) {
            if let Some(cfg) = detect_stream_config(gst_codec, payload) {
                let real_gst_codec = cfg.codec();
                match self.activate(cfg, real_gst_codec, width, height, frame) {
                    Ok((decoder, worker_join, worker_stop)) => {
                        // Replay buffered packets; abort on first failure.
                        let mut registered_ids: Vec<u128> = Vec::new();
                        let mut replay_err: Option<String> = None;

                        for pkt in &buffered {
                            self.register_frame(pkt.frame_id, &pkt.frame);
                            registered_ids.push(pkt.frame_id);
                            if let Err(e) = decoder.submit_packet(
                                &pkt.data,
                                pkt.frame_id,
                                pkt.pts_ns,
                                pkt.dts_ns,
                                pkt.duration_ns,
                            ) {
                                replay_err = Some(format!("buffered packet replay failed: {e}"));
                                break;
                            }
                        }

                        if let Some(err_msg) = replay_err {
                            teardown_activated(&decoder, worker_join, &worker_stop);
                            {
                                let mut fm = self.frame_map.lock();
                                for id in &registered_ids {
                                    fm.remove(id);
                                }
                            }
                            for pkt in buffered {
                                pending.push(FlexibleDecoderOutput::Skipped {
                                    frame: pkt.frame,
                                    data: Some(pkt.data),
                                    reason: SkipReason::DecoderCreationFailed(err_msg.clone()),
                                });
                            }
                            pending.push(FlexibleDecoderOutput::Skipped {
                                frame: frame.clone(),
                                data: Some(payload.to_vec()),
                                reason: SkipReason::DecoderCreationFailed(err_msg),
                            });
                            *state = DecoderState::Idle;
                            return Ok(());
                        }

                        // Submit the current frame.
                        self.register_frame(frame_id, frame);
                        match decoder.submit_packet(
                            payload,
                            frame_id,
                            clk.submission_order_ns,
                            clk.dts_ns,
                            clk.duration_ns,
                        ) {
                            Ok(()) => {
                                *state = DecoderState::Active {
                                    decoder,
                                    worker_join: Some(worker_join),
                                    worker_stop,
                                    gst_codec: real_gst_codec,
                                    video_codec,
                                    width,
                                    height,
                                };
                                return Ok(());
                            }
                            Err(e) => {
                                teardown_activated(&decoder, worker_join, &worker_stop);
                                {
                                    let mut fm = self.frame_map.lock();
                                    for id in &registered_ids {
                                        fm.remove(id);
                                    }
                                    fm.remove(&frame_id);
                                }
                                *state = DecoderState::Idle;
                                return Err(e.into());
                            }
                        }
                    }
                    Err(msg) => {
                        for pkt in buffered.drain(..) {
                            pending.push(FlexibleDecoderOutput::Skipped {
                                frame: pkt.frame,
                                data: Some(pkt.data),
                                reason: SkipReason::DecoderCreationFailed(msg.clone()),
                            });
                        }
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: frame.clone(),
                            data: Some(payload.to_vec()),
                            reason: SkipReason::DecoderCreationFailed(msg),
                        });
                        *state = DecoderState::Idle;
                        return Ok(());
                    }
                }
            }
        }

        buffered.push(BufferedPacket {
            frame: frame.clone(),
            frame_id,
            data: payload.to_vec(),
            pts_ns: clk.submission_order_ns,
            dts_ns: clk.dts_ns,
            duration_ns: clk.duration_ns,
        });

        if buffered.len() > self.config.detect_buffer_limit {
            for pkt in buffered.drain(..) {
                pending.push(FlexibleDecoderOutput::Skipped {
                    frame: pkt.frame,
                    data: Some(pkt.data),
                    reason: SkipReason::DetectionBufferOverflow,
                });
            }
            *state = DecoderState::Idle;
        } else {
            *state = DecoderState::Detecting {
                gst_codec,
                video_codec,
                width,
                height,
                buffered,
            };
        }

        Ok(())
    }

    /// Create a new [`NvDecoder`], its buffer pool, and a worker thread.
    fn activate(
        &self,
        decoder_config: DecoderConfig,
        gst_codec: Codec,
        width: i64,
        height: i64,
        frame: &VideoFrameProxy,
    ) -> Result<ActivatedDecoder, String> {
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

/// Stop the worker, shut down the decoder, and join the thread.
///
/// Used to clean up a newly-activated session when a subsequent operation
/// (e.g. `submit_packet`) fails before the state is committed to `Active`.
fn teardown_activated(decoder: &NvDecoder, worker_join: JoinHandle<()>, worker_stop: &AtomicBool) {
    worker_stop.store(true, Ordering::Relaxed);
    let _ = decoder.shutdown();
    let _ = worker_join.join();
}

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
///
/// Returns `None` for events that have no user-visible representation (e.g.
/// internal events, stream-level EOS during drain) and for decoded frames
/// whose original [`VideoFrameProxy`] is missing from the frame map (logged
/// as a warning; the decoded buffer is dropped).
fn convert_output(fm: &FrameMap, out: NvDecoderOutput) -> Option<FlexibleDecoderOutput> {
    match out {
        NvDecoderOutput::Frame(df) => {
            if let Some(proxy) = take_frame_proxy(fm, df.frame_id) {
                Some(FlexibleDecoderOutput::Frame {
                    frame: proxy,
                    decoded: df,
                })
            } else {
                warn!(
                    "FlexibleDecoder: decoded frame with frame_id {:?} has no matching \
                     VideoFrameProxy; dropping decoded buffer",
                    df.frame_id
                );
                None
            }
        }
        NvDecoderOutput::SourceEos { source_id } => {
            Some(FlexibleDecoderOutput::SourceEos { source_id })
        }
        NvDecoderOutput::Error(e) => Some(FlexibleDecoderOutput::Error(e)),
        NvDecoderOutput::Event(_) => None,
        NvDecoderOutput::Eos => {
            warn!("FlexibleDecoder: unexpected stream-level EOS during drain");
            None
        }
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
                    });
                } else {
                    warn!(
                        "FlexibleDecoder worker: decoded frame with frame_id {:?} has no \
                         matching VideoFrameProxy; dropping decoded buffer",
                        df.frame_id
                    );
                }
            }
            Ok(Some(NvDecoderOutput::SourceEos { source_id })) => {
                on_output(FlexibleDecoderOutput::SourceEos { source_id });
            }
            Ok(Some(NvDecoderOutput::Error(e))) => {
                on_output(FlexibleDecoderOutput::Error(e));
                break;
            }
            Ok(Some(NvDecoderOutput::Event(_))) => {}
            Ok(Some(NvDecoderOutput::Eos)) => {
                warn!("FlexibleDecoder worker: unexpected stream-level EOS");
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
