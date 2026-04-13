//! [`MultiStreamDecoder`] — façade over per-`source_id` [`NvDecoder`] instances.

use super::codec_resolve::{resolve_video_codec, CodecResolve};
use super::config::MultiStreamDecoderConfig;
use super::error::{DecoderOutput, EvictionVerdict, MultiStreamError, StopReason, UndecodedReason};
use super::frame_tracker::FrameTracker;
use super::stream_slot::{
    teardown_stream_entry, ActiveHandle, DetectingState, QueueItem, StreamEntry,
};
use super::watchdog::{spawn_watchdog, WatchdogSignal};
use crossbeam::channel::{bounded, RecvTimeoutError, SendTimeoutError, TrySendError};
use deepstream_buffers::{
    BufferGenerator, CudaStream, NvBufSurfaceMemType, SharedBuffer, SurfaceView, TransformConfig,
    VideoFormat,
};
use deepstream_decoders::{
    detect_stream_config, is_random_access_point, DecoderConfig, JpegBackend, NvDecoder,
    NvDecoderConfig, NvDecoderOutput,
};
use log::{error, warn};
use parking_lot::Mutex;
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTransformation,
};
use savant_core::primitives::gstreamer_frame_time::{frame_clock_ns, normalize_frame_to_gst_ns};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type EvictionCallback = Arc<dyn Fn(&str) -> EvictionVerdict + Send + Sync + 'static>;

/// Set [`VideoFrameProxy`] width/height and [`VideoFrameTransformation::InitialSize`] from the
/// decoded NVMM surface before handing the frame to the user.
fn apply_decoded_surface_to_frame(
    frame: &mut VideoFrameProxy,
    buffer: &SharedBuffer,
    fallback_w: u32,
    fallback_h: u32,
) {
    let (fw, fh) = match SurfaceView::from_buffer(buffer, 0) {
        Ok(v) => (v.width(), v.height()),
        Err(e) => {
            warn!(
                "decoded frame: SurfaceView::from_buffer failed ({e}); using stream pool size {}x{} (source_id={})",
                fallback_w,
                fallback_h,
                frame.get_source_id()
            );
            (fallback_w, fallback_h)
        }
    };

    let pw = frame.get_width();
    let ph = frame.get_height();
    if pw != fw as i64 || ph != fh as i64 {
        warn!(
            "VideoFrameProxy size {}x{} does not match decoded surface {}x{} (source_id={})",
            pw,
            ph,
            fw,
            fh,
            frame.get_source_id()
        );
    }

    frame.clear_transformations();
    frame.add_transformation(VideoFrameTransformation::InitialSize(
        u64::from(fw),
        u64::from(fh),
    ));
    if let Err(e) = frame.set_width(fw as i64) {
        warn!(
            "set_width({fw}) after decode failed for {}: {e}",
            frame.get_source_id()
        );
    }
    if let Err(e) = frame.set_height(fh as i64) {
        warn!(
            "set_height({fh}) after decode failed for {}: {e}",
            frame.get_source_id()
        );
    }
}

/// Extract positive `(width, height)` and `(fps_num, fps_den)` from a [`VideoFrameProxy`].
///
/// Returns `Err` when width/height ≤ 0 or do not fit into `u32`.
fn stream_pool_params(frame: &VideoFrameProxy) -> Result<(u32, u32, i32, i32), MultiStreamError> {
    let w = frame.get_width();
    let h = frame.get_height();
    if w <= 0 || h <= 0 || w > u32::MAX as i64 || h > u32::MAX as i64 {
        return Err(MultiStreamError::InvalidOutputDimensions {
            width: w,
            height: h,
        });
    }
    let (n, d) = frame.get_fps();
    let (n, d) = (n as i32, d as i32);
    let (fps_num, fps_den) = if n > 0 && d > 0 { (n, d) } else { (30, 1) };
    Ok((w as u32, h as u32, fps_num, fps_den))
}

/// Reject hardware JPEG frames whose dimensions are not 8-aligned or are below 16.
fn validate_hw_jpeg_dims(
    cfg: &DecoderConfig,
    frame: &VideoFrameProxy,
) -> Result<(), MultiStreamError> {
    if let DecoderConfig::Jpeg(j) = cfg {
        if j.backend == JpegBackend::Gpu {
            let w = frame.get_width();
            let h = frame.get_height();
            if w < 16 || h < 16 || w % 8 != 0 || h % 8 != 0 {
                return Err(MultiStreamError::InvalidJpegDimensions {
                    width: w,
                    height: h,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum SessionResetReason {
    CodecChanged,
    ResolutionChanged,
    TimestampRegressed,
}

/// Queue depth after a successful enqueue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitResult {
    pub queue_depth: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum EosKind {
    User,
    Idle,
    /// Session boundary (codec / resolution / timestamp change).
    /// Carries the [`StopReason`] and whether the consumer should receive
    /// [`DecoderOutput::Eos`].
    SessionReset {
        stop_reason: StopReason,
        emit_eos: bool,
    },
}

/// Signals the feeder that the drain thread received [`NvDecoderOutput::Eos`].
struct EosSync {
    mu: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl EosSync {
    fn new() -> Self {
        Self {
            mu: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        }
    }

    fn notify(&self) {
        let mut g = self.mu.lock().expect("eos mutex");
        *g = true;
        self.cv.notify_all();
    }

    fn wait(&self, timeout: Duration) {
        let mut g = self.mu.lock().expect("eos mutex");
        let start = Instant::now();
        while !*g {
            let left = timeout.saturating_sub(start.elapsed());
            if left.is_zero() {
                break;
            }
            let (_guard, timeout_res) = self.cv.wait_timeout(g, left).expect("eos condvar");
            g = _guard;
            if timeout_res.timed_out() && !*g {
                warn!("eos wait timed out after {timeout:?}");
                break;
            }
        }
    }
}

/// Multi-stream decode engine.
pub struct MultiStreamDecoder {
    config: MultiStreamDecoderConfig,
    streams: Arc<Mutex<HashMap<String, StreamEntry>>>,
    on_output: Arc<dyn Fn(DecoderOutput) + Send + Sync + 'static>,
    on_eviction: Option<EvictionCallback>,
    shutdown_flag: Arc<AtomicBool>,
    watchdog_signal: Arc<WatchdogSignal>,
    watchdog_join: Mutex<Option<JoinHandle<()>>>,
    /// Handles of threads spawned to tear down streams asynchronously (EOS / session-reset
    /// callbacks run on GStreamer threads and must not block, so teardown is offloaded).
    /// [`shutdown`](Self::shutdown) joins all of them to guarantee that every `NvDecoder` has
    /// been fully dropped before the method returns.
    pending_teardowns: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl MultiStreamDecoder {
    /// Create a decoder and start the watchdog thread.
    pub fn new<Out, Ev>(
        config: MultiStreamDecoderConfig,
        on_output: Out,
        on_eviction: Option<Ev>,
    ) -> Self
    where
        Out: Fn(DecoderOutput) + Send + Sync + 'static,
        Ev: Fn(&str) -> EvictionVerdict + Send + Sync + 'static,
    {
        let streams: Arc<Mutex<HashMap<String, StreamEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let on_output: Arc<dyn Fn(DecoderOutput) + Send + Sync + 'static> = Arc::new(on_output);
        let on_eviction = on_eviction.map(|f| {
            let a: EvictionCallback = Arc::new(f);
            a
        });
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let watchdog_signal = Arc::new(WatchdogSignal::new());
        let scan = config.idle_timeout / 2;
        let scan = if scan.is_zero() {
            Duration::from_millis(100)
        } else {
            scan
        };
        let wd_streams = streams.clone();
        let wd_flag = shutdown_flag.clone();
        let wd_sig = watchdog_signal.clone();
        let idle = config.idle_timeout;
        let on_out_wd = on_output.clone();
        let pending_teardowns: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));
        let join = spawn_watchdog(wd_streams, idle, scan, wd_sig, wd_flag, on_out_wd);
        Self {
            config,
            streams,
            on_output,
            on_eviction,
            shutdown_flag,
            watchdog_signal,
            watchdog_join: Mutex::new(Some(join)),
            pending_teardowns,
        }
    }

    /// Active `source_id` list (may include detecting / failed until watchdog removes them).
    pub fn active_streams(&self) -> Vec<String> {
        self.streams.lock().keys().cloned().collect()
    }

    pub fn stream_count(&self) -> usize {
        self.streams.lock().len()
    }

    /// Block until every background teardown (from EOS / session-reset) has completed.
    ///
    /// Call this when you need a synchronization barrier — e.g. after an EOS cycle
    /// before starting the next decode pass on the same `source_id`.  Unlike
    /// [`shutdown`](Self::shutdown) this does **not** stop the decoder; new streams
    /// can still be activated afterwards.
    pub fn wait_for_pending_teardowns(&self) {
        self.join_pending_teardowns();
    }

    /// Stop all workers and the watchdog. Idempotent.
    ///
    /// Blocks until every `NvDecoder` (including those being torn down by background EOS /
    /// session-reset callbacks) has been fully dropped.
    pub fn shutdown(&mut self) {
        if self.shutdown_flag.swap(true, Ordering::AcqRel) {
            return;
        }
        self.watchdog_signal.notify_shutdown();
        let entries: Vec<StreamEntry> = {
            let mut guard = self.streams.lock();
            guard.drain().map(|(_, v)| v).collect()
        };
        for entry in entries {
            teardown_stream_entry(entry);
        }
        if let Some(j) = self.watchdog_join.lock().take() {
            let _ = j.join();
        }
        self.join_pending_teardowns();
    }

    /// Block until all pending background teardown threads finish.
    fn join_pending_teardowns(&self) {
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.pending_teardowns.lock());
        for h in handles {
            let _ = h.join();
        }
    }

    fn emit(&self, o: DecoderOutput) {
        (self.on_output)(o);
    }

    fn active_session_reset_reason_locked(
        &self,
        frame: &VideoFrameProxy,
        a: &ActiveHandle,
    ) -> Option<SessionResetReason> {
        let cur = frame
            .get_codec()
            .map(|c| c.name().to_string())
            .unwrap_or_default();
        if !cur.is_empty() && cur != a.codec_str.as_str() {
            return Some(SessionResetReason::CodecChanged);
        }
        let w = frame.get_width();
        let h = frame.get_height();
        let lw = *a.last_width.lock();
        let lh = *a.last_height.lock();
        if w != lw || h != lh {
            return Some(SessionResetReason::ResolutionChanged);
        }
        let clk = frame_clock_ns(frame);
        let lk = a.last_order_key_ns.lock();
        if let Some(prev) = *lk {
            if clk.submission_order_ns <= prev {
                return Some(SessionResetReason::TimestampRegressed);
            }
        }
        None
    }

    /// Drain the old decoder, deliver all in-flight frames, then tear down.
    ///
    /// Sends `QueueItem::Eos` through the feeder channel so `NvDecoder` processes
    /// every queued packet, the drain thread pulls all decoded frames (delivered
    /// via `DecoderOutput::Decoded`), and finally the EOS callback fires
    /// `StreamStopped` with the correct session-reset reason.
    ///
    /// **Blocks** until the feeder thread exits (drain complete).
    fn emit_session_reset(
        &self,
        source_id: String,
        entry: StreamEntry,
        reason: SessionResetReason,
    ) {
        let emit_eos = match reason {
            SessionResetReason::CodecChanged => self.config.session_boundary_eos.on_codec_change,
            SessionResetReason::ResolutionChanged => {
                self.config.session_boundary_eos.on_resolution_change
            }
            SessionResetReason::TimestampRegressed => {
                self.config.session_boundary_eos.on_timestamp_regress
            }
        };
        let stop = match reason {
            SessionResetReason::CodecChanged => StopReason::CodecChanged,
            SessionResetReason::ResolutionChanged => StopReason::ResolutionChanged,
            SessionResetReason::TimestampRegressed => StopReason::TimestampRegressed,
        };

        if let StreamEntry::Active(ref a) = entry {
            // Tell the NvDecoder EOS callback which StopReason / Eos policy to use.
            *a.eos_kind.lock() = Some(EosKind::SessionReset {
                stop_reason: stop.clone(),
                emit_eos,
            });

            // Trigger a clean drain: feeder processes remaining packets, then
            // sends EOS to NvDecoder, drain thread pulls every decoded frame,
            // EOS callback fires StreamStopped.
            let _ = a.queue_tx.send(QueueItem::Eos);

            // Keep pending_frames alive so we can report leftovers.
            let pending = a.pending_frames.clone();

            // Block until feeder + NvDecoder drain completes.
            teardown_stream_entry(entry);

            // Any frames still in pending_frames were queued behind the drain
            // EOS or silently dropped by the decoder — report them.
            let remaining = pending.lock().drain();
            for (_fid, frame) in remaining {
                self.emit(DecoderOutput::Undecoded {
                    frame,
                    data: None,
                    reason: UndecodedReason::SessionReset,
                });
            }
        } else {
            // Detecting / Failed entry — no decoder to drain.
            if emit_eos {
                self.emit(DecoderOutput::Eos {
                    source_id: source_id.clone(),
                });
            }
            self.emit(DecoderOutput::StreamStopped {
                source_id,
                reason: stop,
            });
        }
    }

    /// RGBA pool for one stream — dimensions and fps come from the activating frame.
    fn make_pool_for_stream(
        &self,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
    ) -> Result<BufferGenerator, MultiStreamError> {
        BufferGenerator::builder(VideoFormat::RGBA, width, height)
            .fps(fps_num, fps_den)
            .gpu_id(self.config.gpu_id)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(self.config.output_pool_size)
            .max_buffers(self.config.output_pool_size)
            .build()
            .map_err(|e| MultiStreamError::PoolCreationFailed(e.to_string()))
    }

    fn extract_payload(
        &self,
        frame: &VideoFrameProxy,
        data: Option<&[u8]>,
    ) -> Result<Vec<u8>, MultiStreamError> {
        if let Some(b) = data {
            return Ok(b.to_vec());
        }
        match frame.get_content().as_ref() {
            VideoFrameContent::Internal(v) => Ok(v.clone()),
            VideoFrameContent::External(_) => Err(MultiStreamError::NoData),
            VideoFrameContent::None => Err(MultiStreamError::NoData),
        }
    }

    /// Submit one compressed frame (or raw) for `source_id` derived from `frame`.
    pub fn submit(
        &self,
        frame: VideoFrameProxy,
        data: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<SubmitResult, MultiStreamError> {
        self.do_submit(frame, data, Some(timeout))
    }

    /// Non-blocking submit; returns `QueueFull` if the per-stream queue is full.
    pub fn try_submit(
        &self,
        frame: VideoFrameProxy,
        data: Option<&[u8]>,
    ) -> Result<SubmitResult, MultiStreamError> {
        self.do_submit(frame, data, None)
    }

    fn do_submit(
        &self,
        mut frame: VideoFrameProxy,
        data: Option<&[u8]>,
        queue_timeout: Option<Duration>,
    ) -> Result<SubmitResult, MultiStreamError> {
        if self.shutdown_flag.load(Ordering::Acquire) {
            return Err(MultiStreamError::ChannelDisconnected("shutdown".into()));
        }
        normalize_frame_to_gst_ns(&mut frame);
        let source_id = frame.get_source_id();

        let payload = match self.extract_payload(&frame, data) {
            Ok(b) => b,
            Err(MultiStreamError::NoData) => {
                let reason = match frame.get_content().as_ref() {
                    VideoFrameContent::External(_) => UndecodedReason::ExternalContent,
                    _ => UndecodedReason::NoPayload,
                };
                self.emit(DecoderOutput::Undecoded {
                    frame,
                    data: None,
                    reason,
                });
                return Ok(SubmitResult { queue_depth: 0 });
            }
            Err(e) => return Err(e),
        };

        let mut guard = self.streams.lock();

        if let Some(StreamEntry::Active(a)) = guard.get(&source_id) {
            if let Some(reason) = self.active_session_reset_reason_locked(&frame, a) {
                let entry = guard.remove(&source_id).unwrap();
                drop(guard);
                self.emit_session_reset(source_id.clone(), entry, reason);
                return self.do_submit(frame, data, queue_timeout);
            }
        }

        if let Some(StreamEntry::Active(a)) = guard.get(&source_id) {
            if a.alive.load(std::sync::atomic::Ordering::Acquire) {
                let clk = frame_clock_ns(&frame);
                let item = QueueItem::Packet {
                    frame_id: frame.get_uuid_u128(),
                    data: payload,
                    pts_ns: clk.submission_order_ns,
                    dts_ns: clk.dts_ns,
                    duration_ns: clk.duration_ns,
                };
                drop(guard);
                return self.enqueue_active(&source_id, frame, item, queue_timeout);
            }
            // Feeder is dead (EOS callback in progress) — remove the stale
            // entry and release the lock *before* spawning the blocking
            // teardown, then retry.  The teardown joins the feeder which
            // waits on the drain thread's EOS callback, and that callback
            // needs the streams lock — holding it here would deadlock.
            let stale = guard.remove(&source_id).unwrap();
            drop(guard);
            let h = std::thread::spawn(move || teardown_stream_entry(stale));
            self.pending_teardowns.lock().push(h);
            return self.do_submit(frame, data, queue_timeout);
        }

        if let Some(StreamEntry::Detecting(d)) = guard.get_mut(&source_id) {
            d.last_seen = Instant::now();
            return self.process_detecting(guard, source_id, frame, payload, queue_timeout);
        }

        if let Some(StreamEntry::Failed { message: _ }) = guard.get(&source_id) {
            self.emit(DecoderOutput::Undecoded {
                frame,
                data: Some(payload),
                reason: UndecodedReason::DetectionFailed("stream failed".into()),
            });
            return Ok(SubmitResult { queue_depth: 0 });
        }

        self.start_or_detect(guard, source_id, frame, payload, queue_timeout)
    }

    fn enqueue_active(
        &self,
        source_id: &str,
        frame: VideoFrameProxy,
        mut item: QueueItem,
        queue_timeout: Option<Duration>,
    ) -> Result<SubmitResult, MultiStreamError> {
        let guard = self.streams.lock();
        let StreamEntry::Active(a) = guard
            .get(source_id)
            .ok_or_else(|| MultiStreamError::UnknownStream(source_id.into()))?
        else {
            return Err(MultiStreamError::UnknownStream(source_id.into()));
        };
        let tx = a.queue_tx.clone();
        let cap = self.config.per_stream_queue_size;
        let clk = frame_clock_ns(&frame);
        if let QueueItem::Packet {
            ref mut pts_ns,
            ref mut dts_ns,
            ref mut duration_ns,
            ..
        } = &mut item
        {
            *pts_ns = clk.submission_order_ns;
            *dts_ns = clk.dts_ns;
            *duration_ns = clk.duration_ns;
        }
        let tracker = a.pending_frames.clone();
        {
            let mut t = tracker.lock();
            let _ = t.insert(frame.get_uuid_u128(), frame.clone());
        }
        drop(guard);
        let err_map = |e: MultiStreamError| match e {
            MultiStreamError::QueueFull { .. } => MultiStreamError::QueueFull {
                source_id: source_id.to_string(),
                queue_size: cap,
            },
            MultiStreamError::ChannelDisconnected(_) => {
                MultiStreamError::ChannelDisconnected(source_id.to_string())
            }
            o => o,
        };
        let send_res = match queue_timeout {
            Some(d) => tx.send_timeout(item, d).map_err(|e| match e {
                SendTimeoutError::Timeout(_) => MultiStreamError::QueueFull {
                    source_id: String::new(),
                    queue_size: cap,
                },
                SendTimeoutError::Disconnected(_) => {
                    MultiStreamError::ChannelDisconnected(String::new())
                }
            }),
            None => tx.try_send(item).map_err(|e| match e {
                TrySendError::Full(_) => MultiStreamError::QueueFull {
                    source_id: String::new(),
                    queue_size: cap,
                },
                TrySendError::Disconnected(_) => {
                    MultiStreamError::ChannelDisconnected(String::new())
                }
            }),
        };
        send_res.map_err(err_map)?;
        let depth = tx.len();
        {
            let g = self.streams.lock();
            if let Some(StreamEntry::Active(a)) = g.get(source_id) {
                *a.last_order_key_ns.lock() = Some(clk.submission_order_ns);
                *a.last_width.lock() = frame.get_width();
                *a.last_height.lock() = frame.get_height();
            }
        }
        Ok(SubmitResult { queue_depth: depth })
    }

    fn process_detecting(
        &self,
        mut guard: parking_lot::MutexGuard<'_, HashMap<String, StreamEntry>>,
        source_id: String,
        frame: VideoFrameProxy,
        payload: Vec<u8>,
        queue_timeout: Option<Duration>,
    ) -> Result<SubmitResult, MultiStreamError> {
        let max = self.config.max_detection_buffer;

        {
            // The entry may have vanished if a concurrent EOS callback from a
            // previous session ran between the caller's guard release and here.
            // Retry via do_submit so the frame gets routed correctly.
            let entry = match guard.get_mut(&source_id) {
                Some(e) => e,
                None => {
                    drop(guard);
                    // Payload is already owned; pass it as a slice so do_submit
                    // can use it without re-extracting from frame content.
                    return self.do_submit(frame, Some(&payload), queue_timeout);
                }
            };
            let StreamEntry::Detecting(d) = entry else {
                return Ok(SubmitResult { queue_depth: 0 });
            };
            if let CodecResolve::NeedDetection { codec } = &d.resolve {
                let c = *codec;
                if is_random_access_point(c, &payload) {
                    if let Some(cfg) = detect_stream_config(c, &payload) {
                        let pending = std::mem::take(&mut d.pending);
                        let codec_str = d.codec_str.clone();
                        guard.remove(&source_id);
                        drop(guard);
                        for (f, data) in pending {
                            self.emit(DecoderOutput::Undecoded {
                                frame: f,
                                data: Some(data),
                                reason: UndecodedReason::AwaitingKeyframe,
                            });
                        }
                        self.activate_stream(
                            source_id,
                            codec_str,
                            cfg,
                            frame,
                            payload,
                            queue_timeout,
                        )?;
                        return Ok(SubmitResult { queue_depth: 1 });
                    }
                }
            }
        }

        {
            match guard.get_mut(&source_id) {
                None => return Ok(SubmitResult { queue_depth: 0 }),
                Some(entry) => {
                    let StreamEntry::Detecting(d) = entry else {
                        return Ok(SubmitResult { queue_depth: 0 });
                    };
                    match &d.resolve {
                        CodecResolve::NeedDetection { .. } => {
                            d.pending.push((frame, payload));
                        }
                        CodecResolve::Ready(_) => {
                            d.pending.push((frame, payload));
                        }
                    }
                }
            }
        }

        let fail_pending = {
            match guard.get_mut(&source_id) {
                None => return Ok(SubmitResult { queue_depth: 0 }),
                Some(entry) => {
                    let StreamEntry::Detecting(d) = entry else {
                        return Ok(SubmitResult { queue_depth: 0 });
                    };
                    if d.pending.len() > max {
                        Some(std::mem::take(&mut d.pending))
                    } else {
                        None
                    }
                }
            }
        };

        if let Some(pending) = fail_pending {
            guard.insert(
                source_id.clone(),
                StreamEntry::Failed {
                    message: "detection buffer exceeded".into(),
                },
            );
            drop(guard);
            for (f, data) in pending {
                self.emit(DecoderOutput::Undecoded {
                    frame: f,
                    data: Some(data),
                    reason: UndecodedReason::DetectionFailed(
                        "max_detection_buffer exceeded without RAP".into(),
                    ),
                });
            }
            return Ok(SubmitResult { queue_depth: 0 });
        }

        let depth = guard
            .get(&source_id)
            .and_then(|e| match e {
                StreamEntry::Detecting(det) => Some(det.pending.len()),
                _ => None,
            })
            .unwrap_or(0);
        Ok(SubmitResult { queue_depth: depth })
    }

    fn start_or_detect(
        &self,
        mut guard: parking_lot::MutexGuard<'_, HashMap<String, StreamEntry>>,
        source_id: String,
        frame: VideoFrameProxy,
        payload: Vec<u8>,
        queue_timeout: Option<Duration>,
    ) -> Result<SubmitResult, MultiStreamError> {
        let codec_opt = frame.get_codec();
        let fw = frame.get_width().max(0) as u32;
        let fh = frame.get_height().max(0) as u32;
        let resolve = match resolve_video_codec(codec_opt, fw, fh) {
            Ok(r) => r,
            Err(None) => {
                self.emit(DecoderOutput::Undecoded {
                    frame,
                    data: Some(payload),
                    reason: UndecodedReason::UnsupportedCodec(None),
                });
                return Ok(SubmitResult { queue_depth: 0 });
            }
            Err(Some(s)) => {
                self.emit(DecoderOutput::Undecoded {
                    frame,
                    data: Some(payload),
                    reason: UndecodedReason::UnsupportedCodec(Some(s)),
                });
                return Ok(SubmitResult { queue_depth: 0 });
            }
        };

        match resolve {
            CodecResolve::NeedDetection { codec: _ } => {
                let det = DetectingState {
                    pending: Vec::new(),
                    codec_str: codec_opt.map(|c| c.name().to_string()).unwrap_or_default(),
                    resolve,
                    last_seen: Instant::now(),
                };
                guard.insert(source_id.clone(), StreamEntry::Detecting(det));
                drop(guard);
                self.process_detecting(
                    self.streams.lock(),
                    source_id,
                    frame,
                    payload,
                    queue_timeout,
                )
            }
            CodecResolve::Ready(cfg) => {
                drop(guard);
                self.activate_stream(
                    source_id,
                    codec_opt.map(|c| c.name().to_string()).unwrap_or_default(),
                    cfg,
                    frame,
                    payload,
                    queue_timeout,
                )
            }
        }
    }

    fn activate_stream(
        &self,
        source_id: String,
        codec_str: String,
        cfg: DecoderConfig,
        frame: VideoFrameProxy,
        payload: Vec<u8>,
        queue_timeout: Option<Duration>,
    ) -> Result<SubmitResult, MultiStreamError> {
        // Block until any prior teardown for this (or any) source finishes.
        // This guarantees hardware decoder resources (NVDEC slots) are released
        // before we create a new NvDecoder, preventing Jetson pipeline hangs.
        self.join_pending_teardowns();
        validate_hw_jpeg_dims(&cfg, &frame)?;
        let (pool_w, pool_h, fps_num, fps_den) = stream_pool_params(&frame)?;
        let pool = self.make_pool_for_stream(pool_w, pool_h, fps_num, fps_den)?;
        let cuda_stream = CudaStream::new_non_blocking().map_err(|e| {
            MultiStreamError::PoolCreationFailed(format!("CUDA stream for transform: {e}"))
        })?;
        let transform = TransformConfig {
            interpolation: self.config.interpolation,
            cuda_stream,
            ..TransformConfig::default()
        };
        let qsize = self.config.per_stream_queue_size;
        let (tx, rx) = bounded::<QueueItem>(qsize);
        let alive = Arc::new(AtomicBool::new(true));
        let join_slot = Arc::new(Mutex::new(None));
        let last_width = Arc::new(Mutex::new(frame.get_width()));
        let last_height = Arc::new(Mutex::new(frame.get_height()));
        let last_order_key_ns = Arc::new(Mutex::new(None));
        let track_cap = (self.config.output_pool_size as usize)
            .saturating_mul(2)
            .max(8);
        let pending_frames = Arc::new(Mutex::new(FrameTracker::new(track_cap)));
        let on_output = self.on_output.clone();
        let on_eviction = self.on_eviction.clone();
        let idle_timeout = self.config.idle_timeout;
        let gpu_id = self.config.gpu_id;
        let eos_kind = Arc::new(Mutex::new(None::<EosKind>));
        let eos_sync = Arc::new(EosSync::new());
        let streams_map = self.streams.clone();
        let sid_for_map = source_id.clone();

        let dec_codec = cfg.codec();
        let decoder = NvDecoder::new(NvDecoderConfig::new(gpu_id, cfg), pool, transform)
            .map_err(|e| MultiStreamError::DecoderCreationFailed(e.to_string()))?;
        let decoder = Arc::new(decoder);

        spawn_drain(
            source_id.clone(),
            Arc::clone(&decoder),
            pending_frames.clone(),
            on_output.clone(),
            alive.clone(),
            eos_kind.clone(),
            eos_sync.clone(),
            streams_map.clone(),
            sid_for_map.clone(),
            self.pending_teardowns.clone(),
            pool_w,
            pool_h,
        );

        (self.on_output)(DecoderOutput::StreamStarted {
            source_id: source_id.clone(),
            codec: dec_codec,
        });

        let join = spawn_feeder(
            source_id.clone(),
            rx,
            decoder,
            idle_timeout,
            on_eviction,
            eos_kind.clone(),
            eos_sync,
            alive.clone(),
        );
        *join_slot.lock() = Some(join);

        let handle = ActiveHandle {
            queue_tx: tx.clone(),
            alive: alive.clone(),
            join: join_slot.clone(),
            codec_str,
            last_width,
            last_height,
            last_order_key_ns,
            eos_kind: eos_kind.clone(),
            pending_frames,
        };
        self.streams
            .lock()
            .insert(source_id.clone(), StreamEntry::Active(handle));

        let first_fid = frame.get_uuid_u128();
        // First packet: use submit path
        self.enqueue_active(
            &source_id,
            frame,
            QueueItem::Packet {
                frame_id: first_fid,
                data: payload,
                pts_ns: 0,
                dts_ns: None,
                duration_ns: None,
            },
            queue_timeout,
        )
        .inspect_err(|_| {
            let _ = self.streams.lock().remove(&source_id);
        })
    }

    pub fn submit_eos(&self, eos: &EndOfStream, timeout: Duration) -> Result<(), MultiStreamError> {
        self.do_submit_eos(eos, Some(timeout))
    }

    pub fn try_submit_eos(&self, eos: &EndOfStream) -> Result<(), MultiStreamError> {
        self.do_submit_eos(eos, None)
    }

    fn do_submit_eos(
        &self,
        eos: &EndOfStream,
        queue_timeout: Option<Duration>,
    ) -> Result<(), MultiStreamError> {
        if self.shutdown_flag.load(Ordering::Acquire) {
            return Err(MultiStreamError::ChannelDisconnected("shutdown".into()));
        }
        let sid = eos.get_source_id().to_string();
        let mut guard = self.streams.lock();

        let detecting_pending: Option<Vec<(VideoFrameProxy, Vec<u8>)>> = match guard.get_mut(&sid) {
            Some(StreamEntry::Detecting(d)) => Some(std::mem::take(&mut d.pending)),
            _ => None,
        };

        if let Some(pending) = detecting_pending {
            guard.remove(&sid);
            drop(guard);
            for (f, data) in pending {
                self.emit(DecoderOutput::Undecoded {
                    frame: f,
                    data: Some(data),
                    reason: UndecodedReason::StreamEvicted,
                });
            }
            self.emit(DecoderOutput::Eos {
                source_id: sid.clone(),
            });
            self.emit(DecoderOutput::StreamStopped {
                source_id: sid,
                reason: StopReason::Eos,
            });
            return Ok(());
        }

        match guard.get(&sid) {
            None => {
                self.emit(DecoderOutput::Eos {
                    source_id: sid.clone(),
                });
                self.emit(DecoderOutput::StreamStopped {
                    source_id: sid,
                    reason: StopReason::Eos,
                });
                Ok(())
            }
            Some(StreamEntry::Detecting(_)) => {
                unreachable!("Detecting slot should have been removed in submit_eos pre-pass")
            }
            Some(StreamEntry::Failed { message: _ }) => {
                guard.remove(&sid);
                drop(guard);
                self.emit(DecoderOutput::Eos {
                    source_id: sid.clone(),
                });
                self.emit(DecoderOutput::StreamStopped {
                    source_id: sid,
                    reason: StopReason::Eos,
                });
                Ok(())
            }
            Some(StreamEntry::Active(a)) => {
                let tx = a.queue_tx.clone();
                let cap = self.config.per_stream_queue_size;
                drop(guard);
                let send_res = match queue_timeout {
                    Some(d) => tx.send_timeout(QueueItem::Eos, d).map_err(|e| match e {
                        SendTimeoutError::Timeout(_) => MultiStreamError::QueueFull {
                            source_id: String::new(),
                            queue_size: cap,
                        },
                        SendTimeoutError::Disconnected(_) => {
                            MultiStreamError::ChannelDisconnected(String::new())
                        }
                    }),
                    None => tx.try_send(QueueItem::Eos).map_err(|e| match e {
                        TrySendError::Full(_) => MultiStreamError::QueueFull {
                            source_id: String::new(),
                            queue_size: cap,
                        },
                        TrySendError::Disconnected(_) => {
                            MultiStreamError::ChannelDisconnected(String::new())
                        }
                    }),
                };
                send_res.map_err(|e| match e {
                    MultiStreamError::QueueFull { .. } => MultiStreamError::QueueFull {
                        source_id: sid.clone(),
                        queue_size: cap,
                    },
                    MultiStreamError::ChannelDisconnected(_) => {
                        MultiStreamError::ChannelDisconnected(sid.clone())
                    }
                    o => o,
                })
            }
        }
    }
}

/// Drain thread: pulls [`NvDecoderOutput`] from the decoder and delivers
/// decoded frames / EOS / errors through `on_output`.
#[allow(clippy::too_many_arguments)]
fn spawn_drain(
    source_id: String,
    decoder: Arc<NvDecoder>,
    pending_frames: Arc<Mutex<FrameTracker>>,
    on_output: Arc<dyn Fn(DecoderOutput) + Send + Sync + 'static>,
    alive: Arc<AtomicBool>,
    eos_kind: Arc<Mutex<Option<EosKind>>>,
    eos_sync: Arc<EosSync>,
    streams_map: Arc<Mutex<HashMap<String, StreamEntry>>>,
    sid_for_map: String,
    teardown_joins: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pool_w: u32,
    pool_h: u32,
) {
    std::thread::Builder::new()
        .name(format!("ms-dec-drain-{source_id}"))
        .spawn(move || loop {
            let ev = decoder.recv();
            match ev {
                Ok(NvDecoderOutput::Frame(df)) => {
                    let fid = df.frame_id.unwrap_or(0);
                    let buf = df.buffer;
                    let mut t = pending_frames.lock();
                    if let Some(mut f) = t.remove(fid) {
                        apply_decoded_surface_to_frame(&mut f, &buf, pool_w, pool_h);
                        (on_output)(DecoderOutput::Decoded {
                            frame: f,
                            buffer: buf,
                        });
                    }
                }
                Ok(NvDecoderOutput::Eos) => {
                    handle_eos_or_error(
                        &source_id,
                        &eos_kind,
                        &on_output,
                        &alive,
                        &eos_sync,
                        &streams_map,
                        &sid_for_map,
                        &teardown_joins,
                        None,
                    );
                    break;
                }
                Ok(NvDecoderOutput::Error(e)) => {
                    error!("decoder error for {source_id}: {e}");
                    handle_eos_or_error(
                        &source_id,
                        &eos_kind,
                        &on_output,
                        &alive,
                        &eos_sync,
                        &streams_map,
                        &sid_for_map,
                        &teardown_joins,
                        Some(e.to_string()),
                    );
                    break;
                }
                Ok(NvDecoderOutput::Event(_)) => {}
                Err(_) => {
                    alive.store(false, Ordering::Release);
                    eos_sync.notify();
                    break;
                }
            }
        })
        .expect("drain thread");
}

/// Shared EOS / error handling for the drain thread.
#[allow(clippy::too_many_arguments)]
fn handle_eos_or_error(
    source_id: &str,
    eos_kind: &Mutex<Option<EosKind>>,
    on_output: &Arc<dyn Fn(DecoderOutput) + Send + Sync + 'static>,
    alive: &Arc<AtomicBool>,
    eos_sync: &EosSync,
    streams_map: &Arc<Mutex<HashMap<String, StreamEntry>>>,
    sid_for_map: &str,
    teardown_joins: &Arc<Mutex<Vec<JoinHandle<()>>>>,
    error_msg: Option<String>,
) {
    if let Some(err) = error_msg {
        (on_output)(DecoderOutput::StreamStopped {
            source_id: source_id.to_string(),
            reason: StopReason::Error(err),
        });
    } else {
        let kind = eos_kind.lock().take();
        let (do_emit_eos, reason) = match kind {
            Some(EosKind::User) | None => (true, StopReason::Eos),
            Some(EosKind::Idle) => (true, StopReason::IdleEviction),
            Some(EosKind::SessionReset {
                stop_reason,
                emit_eos,
            }) => (emit_eos, stop_reason),
        };
        if do_emit_eos {
            (on_output)(DecoderOutput::Eos {
                source_id: source_id.to_string(),
            });
        }
        (on_output)(DecoderOutput::StreamStopped {
            source_id: source_id.to_string(),
            reason,
        });
    }
    alive.store(false, Ordering::Release);
    eos_sync.notify();
    // Guard against removing a newer session that reused this source_id.
    let maybe_ent = {
        let mut guard = streams_map.lock();
        let is_mine = guard.get(sid_for_map).is_some_and(
            |ent| matches!(ent, StreamEntry::Active(a) if Arc::ptr_eq(&a.alive, alive)),
        );
        if is_mine {
            guard.remove(sid_for_map)
        } else {
            None
        }
    };
    if let Some(ent) = maybe_ent {
        let h = std::thread::spawn(move || teardown_stream_entry(ent));
        teardown_joins.lock().push(h);
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_feeder(
    source_id: String,
    rx: crossbeam::channel::Receiver<QueueItem>,
    decoder: Arc<NvDecoder>,
    idle_timeout: Duration,
    on_eviction: Option<EvictionCallback>,
    eos_kind: Arc<Mutex<Option<EosKind>>>,
    eos_sync: Arc<EosSync>,
    alive: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("ms-dec-feed-{source_id}"))
        .spawn(move || {
            let mut wait = idle_timeout;
            loop {
                match rx.recv_timeout(wait) {
                    Ok(QueueItem::Packet {
                        data,
                        frame_id,
                        pts_ns,
                        dts_ns,
                        duration_ns,
                    }) => {
                        wait = idle_timeout;
                        if let Err(e) =
                            decoder.submit_packet(&data, frame_id, pts_ns, dts_ns, duration_ns)
                        {
                            error!("submit_packet {source_id}: {e}");
                        }
                    }
                    Ok(QueueItem::Eos) => {
                        {
                            let mut k = eos_kind.lock();
                            if k.is_none() {
                                *k = Some(EosKind::User);
                            }
                        }
                        if let Err(e) = decoder.send_eos() {
                            error!("send_eos {source_id}: {e}");
                        }
                        eos_sync.wait(Duration::from_secs(120));
                        alive.store(false, Ordering::Release);
                        break;
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        let verdict = on_eviction
                            .as_ref()
                            .map(|f| f(source_id.as_str()))
                            .unwrap_or(EvictionVerdict::Approve);
                        match verdict {
                            EvictionVerdict::Approve => {
                                {
                                    let mut k = eos_kind.lock();
                                    if k.is_none() {
                                        *k = Some(EosKind::Idle);
                                    }
                                }
                                if let Err(e) = decoder.send_eos() {
                                    error!("idle send_eos {source_id}: {e}");
                                }
                                eos_sync.wait(Duration::from_secs(120));
                                alive.store(false, Ordering::Release);
                                break;
                            }
                            EvictionVerdict::Extend(d) => {
                                wait = d;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        alive.store(false, Ordering::Release);
                        break;
                    }
                }
            }
        })
        .expect("feeder")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: i64, h: i64) -> VideoFrameProxy {
        use savant_core::primitives::frame::{VideoFrameContent, VideoFrameTranscodingMethod};
        use savant_core::primitives::video_codec::VideoCodec;
        VideoFrameProxy::new(
            "test",
            (30, 1),
            w,
            h,
            VideoFrameContent::None,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::Jpeg),
            None,
            (1, 1_000_000_000),
            0,
            None,
            None,
        )
        .unwrap()
    }

    fn gpu_jpeg_cfg() -> DecoderConfig {
        use deepstream_decoders::JpegDecoderConfig;
        DecoderConfig::Jpeg(JpegDecoderConfig::gpu())
    }

    fn cpu_jpeg_cfg() -> DecoderConfig {
        use deepstream_decoders::JpegDecoderConfig;
        DecoderConfig::Jpeg(JpegDecoderConfig::cpu())
    }

    #[test]
    fn hw_jpeg_rejects_unaligned_width() {
        let f = make_frame(100, 64);
        assert!(validate_hw_jpeg_dims(&gpu_jpeg_cfg(), &f).is_err());
    }

    #[test]
    fn hw_jpeg_rejects_unaligned_height() {
        let f = make_frame(64, 100);
        assert!(validate_hw_jpeg_dims(&gpu_jpeg_cfg(), &f).is_err());
    }

    #[test]
    fn hw_jpeg_rejects_too_small() {
        let f = make_frame(8, 8);
        assert!(validate_hw_jpeg_dims(&gpu_jpeg_cfg(), &f).is_err());
    }

    #[test]
    fn hw_jpeg_accepts_valid() {
        let f = make_frame(320, 240);
        assert!(validate_hw_jpeg_dims(&gpu_jpeg_cfg(), &f).is_ok());
    }

    #[test]
    fn hw_jpeg_accepts_minimum() {
        let f = make_frame(16, 16);
        assert!(validate_hw_jpeg_dims(&gpu_jpeg_cfg(), &f).is_ok());
    }

    #[test]
    fn cpu_jpeg_skips_validation() {
        let f = make_frame(100, 100);
        assert!(validate_hw_jpeg_dims(&cpu_jpeg_cfg(), &f).is_ok());
    }

    #[test]
    fn non_jpeg_skips_validation() {
        let f = make_frame(100, 100);
        let cfg = DecoderConfig::Vp8(deepstream_decoders::Vp8DecoderConfig::default());
        assert!(validate_hw_jpeg_dims(&cfg, &f).is_ok());
    }
}
