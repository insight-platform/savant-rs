use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Receiver, RecvTimeoutError, Sender, TrySendError};
use log::{info, warn};

use glib::translate::from_glib_none;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::{AppSink, AppSrc};
use parking_lot::Mutex;

use crate::pipeline::bridge_meta::bridge_savant_id_meta_across;
use crate::pipeline::config::{PipelineConfig, PtsPolicy};
use crate::pipeline::error::PipelineError;
use crate::pipeline::watchdog::spawn_watchdog;

#[derive(Debug)]
pub enum PipelineInput {
    Buffer(gst::Buffer),
    Event(gst::Event),
    Eos,
}

#[derive(Debug)]
pub enum PipelineOutput {
    Buffer(gst::Buffer),
    Event(gst::Event),
    Eos,
    Error(PipelineError),
}

/// Send `msg` on `tx`, retrying every 50 ms.  Returns `Err` (giving back
/// the message) when `shutdown` becomes `true` or the channel disconnects.
/// This replaces bare `tx.send(msg)` in background threads so that a full
/// output channel never prevents the thread from observing the shutdown flag.
pub(crate) fn send_or_shutdown_on(
    tx: &Sender<PipelineOutput>,
    shutdown: &AtomicBool,
    msg: PipelineOutput,
) -> Result<(), PipelineOutput> {
    let mut m = msg;
    loop {
        if shutdown.load(Ordering::Acquire) {
            return Err(m);
        }
        match tx.send_timeout(m, Duration::from_millis(50)) {
            Ok(()) => return Ok(()),
            Err(channel::SendTimeoutError::Timeout(ret)) => m = ret,
            Err(channel::SendTimeoutError::Disconnected(_)) => return Err(PipelineOutput::Eos),
        }
    }
}

/// GStreamer pipeline with appsrc/appsink and background threads.
///
/// All background threads (feeder, drain, watchdog) use timeout-based sends
/// on the output channel so that they always observe the shutdown flag within
/// 50 ms, even when the channel is full.  This guarantees that thread joins
/// complete promptly after `shutdown()` sets the flag and transitions the
/// pipeline to Null.
///
/// The `pipeline` and `appsrc` fields are wrapped in `Option` so that
/// [`shutdown`](Self::shutdown) can `.take()` them after joining threads,
/// triggering immediate GObject finalization and GPU/NvMap resource release.
/// This allows callers behind `&self` (e.g. PyO3 wrappers) to reclaim GPU
/// memory without requiring ownership transfer.
pub struct GstPipeline {
    name: String,
    pipeline: Option<gst::Pipeline>,
    appsrc: Option<AppSrc>,
    shutdown: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    feeder_thread: Option<JoinHandle<()>>,
    drain_thread: Option<JoinHandle<()>>,
    watchdog_thread: Option<JoinHandle<()>>,
    is_shut_down: Arc<AtomicBool>,
    draining: Arc<AtomicBool>,
    leak_on_finalize: bool,
}

impl GstPipeline {
    pub fn start(
        mut config: PipelineConfig,
    ) -> Result<(Sender<PipelineInput>, Receiver<PipelineOutput>, Self), PipelineError> {
        gst::init().map_err(|e| PipelineError::InitFailed(e.to_string()))?;

        let pipeline = gst::Pipeline::new();

        let appsrc = gst::ElementFactory::make("appsrc")
            .name("src")
            .build()
            .map_err(|_| PipelineError::ElementCreationFailed("appsrc".into()))?
            .dynamic_cast::<AppSrc>()
            .map_err(|_| PipelineError::ElementCreationFailed("appsrc cast".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| PipelineError::ElementCreationFailed("appsink".into()))?
            .dynamic_cast::<AppSink>()
            .map_err(|_| PipelineError::ElementCreationFailed("appsink cast".into()))?;

        let appsrc_elem: &gst::Element = appsrc.upcast_ref();
        appsrc_elem.set_property("caps", &config.appsrc_caps);
        appsrc_elem.set_property_from_str("format", "time");
        appsrc_elem.set_property_from_str("stream-type", "stream");

        let appsink_elem: &gst::Element = appsink.upcast_ref();
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("emit-signals", false);

        let mut chain: Vec<gst::Element> = Vec::with_capacity(config.elements.len() + 2);
        chain.push(appsrc.clone().upcast());
        chain.extend(config.elements);
        chain.push(appsink.clone().upcast());

        for element in &chain {
            pipeline
                .add(element)
                .map_err(|e| PipelineError::ElementAddFailed(e.to_string()))?;
        }
        let refs: Vec<&gst::Element> = chain.iter().collect();
        gst::Element::link_many(&refs)
            .map_err(|_| PipelineError::LinkFailed("appsrc -> GAP -> appsink".into()))?;

        let appsrc_src_pad = appsrc
            .static_pad("src")
            .ok_or_else(|| PipelineError::MissingPad("appsrc src".into()))?;
        let appsink_sink_pad = appsink
            .static_pad("sink")
            .ok_or_else(|| PipelineError::MissingPad("appsink sink".into()))?;
        bridge_savant_id_meta_across(&appsrc_src_pad, &appsink_sink_pad)?;

        if let Some(probe) = config.appsrc_probe.take() {
            probe(&appsrc_src_pad);
        }

        let (input_tx, input_rx) = channel::bounded::<PipelineInput>(config.input_channel_capacity);
        let (output_tx, output_rx) =
            channel::bounded::<PipelineOutput>(config.output_channel_capacity);

        // Capture all downstream non-EOS events that arrive to appsink.
        // Uses try_send to avoid blocking the GStreamer streaming thread when the
        // bounded output channel is full (which would deadlock set_state(Null)).
        let event_out = output_tx.clone();
        appsink_sink_pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
            if let Some(event) = info.event() {
                if event.type_() != gst::EventType::Eos {
                    let _ = event_out.try_send(PipelineOutput::Event(event.to_owned()));
                }
            }
            gst::PadProbeReturn::Ok
        });

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| PipelineError::StateChangeFailed(format!("set Playing: {:?}", e)))?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let is_shut_down = Arc::new(AtomicBool::new(false));
        let draining = Arc::new(AtomicBool::new(false));
        let in_flight: Arc<Mutex<HashMap<u64, Instant>>> = Arc::new(Mutex::new(HashMap::new()));

        let feeder_shutdown = shutdown.clone();
        let feeder_failed = failed.clone();
        let feeder_appsrc = appsrc.clone();
        let feeder_tx = output_tx.clone();
        let feeder_in_flight = in_flight.clone();
        let feeder_pts_policy = config.pts_policy;
        let feeder_name = format!("{}-feeder", config.name);
        let feeder_thread = std::thread::Builder::new()
            .name(feeder_name)
            .spawn(move || {
                let mut last_key: Option<u64> = None;

                loop {
                    if feeder_shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    match input_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(PipelineInput::Buffer(buffer)) => {
                            if let Some(policy) = feeder_pts_policy {
                                if let Some(violation) =
                                    check_pts_policy(policy, &buffer, &mut last_key)
                                {
                                    let _ = send_or_shutdown_on(
                                        &feeder_tx,
                                        &feeder_shutdown,
                                        PipelineOutput::Error(violation),
                                    );
                                    continue;
                                }
                            }
                            let pts = buffer.pts().map(|p| p.nseconds());
                            if let Some(pts_ns) = pts {
                                feeder_in_flight.lock().insert(pts_ns, Instant::now());
                            }
                            if let Err(e) = feeder_appsrc.push_buffer(buffer) {
                                if let Some(pts_ns) = pts {
                                    feeder_in_flight.lock().remove(&pts_ns);
                                }
                                feeder_failed.store(true, Ordering::Release);
                                let _ = send_or_shutdown_on(
                                    &feeder_tx,
                                    &feeder_shutdown,
                                    PipelineOutput::Error(PipelineError::RuntimeError(format!(
                                        "appsrc push failed: {:?}",
                                        e
                                    ))),
                                );
                                break;
                            }
                        }
                        Ok(PipelineInput::Event(event)) => {
                            let _ = feeder_appsrc.send_event(event);
                        }
                        Ok(PipelineInput::Eos) => {
                            let _ = feeder_appsrc.end_of_stream();
                            break;
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => {
                            let _ = feeder_appsrc.end_of_stream();
                            break;
                        }
                    }
                }
            })
            .map_err(|e| PipelineError::RuntimeError(format!("spawn feeder failed: {}", e)))?;

        let drain_shutdown = shutdown.clone();
        let drain_failed = failed.clone();
        let drain_sink = appsink.clone();
        let drain_pipeline = pipeline.clone();
        let drain_tx = output_tx.clone();
        let drain_in_flight = in_flight.clone();
        let drain_timeout = config.drain_poll_interval;
        let drain_name = format!("{}-drain", config.name);
        let drain_thread = std::thread::Builder::new()
            .name(drain_name)
            .spawn(move || {
                let bus = drain_pipeline.bus();

                loop {
                    if drain_shutdown.load(Ordering::Acquire) {
                        break;
                    }

                    match drain_sink.try_pull_sample(gst::ClockTime::from_nseconds(
                        drain_timeout.as_nanos() as u64,
                    )) {
                        Some(sample) => {
                            if let Some(buffer_ref) = sample.buffer() {
                                let buffer: gst::Buffer =
                                    unsafe { from_glib_none(buffer_ref.as_ptr()) };
                                if let Some(pts) = buffer.pts() {
                                    drain_in_flight.lock().remove(&pts.nseconds());
                                }
                                if send_or_shutdown_on(
                                    &drain_tx,
                                    &drain_shutdown,
                                    PipelineOutput::Buffer(buffer),
                                )
                                .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        None if drain_sink.is_eos() => {
                            let _ = send_or_shutdown_on(
                                &drain_tx,
                                &drain_shutdown,
                                PipelineOutput::Eos,
                            );
                            break;
                        }
                        None => {
                            if let Some(bus) = &bus {
                                if let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error]) {
                                    if let gst::MessageView::Error(err) = msg.view() {
                                        drain_failed.store(true, Ordering::Release);
                                        let error_msg = format!(
                                            "{} ({})",
                                            err.error(),
                                            err.debug().unwrap_or_default()
                                        );
                                        let _ = send_or_shutdown_on(
                                            &drain_tx,
                                            &drain_shutdown,
                                            PipelineOutput::Error(PipelineError::RuntimeError(
                                                error_msg,
                                            )),
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            })
            .map_err(|e| PipelineError::RuntimeError(format!("spawn drain failed: {}", e)))?;

        let watchdog_thread = config.operation_timeout.and_then(|timeout| {
            spawn_watchdog(
                format!("{}-watchdog", config.name),
                timeout,
                in_flight,
                shutdown.clone(),
                failed.clone(),
                output_tx,
            )
        });

        Ok((
            input_tx,
            output_rx,
            Self {
                name: config.name,
                pipeline: Some(pipeline),
                appsrc: Some(appsrc),
                shutdown,
                failed,
                feeder_thread: Some(feeder_thread),
                drain_thread: Some(drain_thread),
                watchdog_thread,
                is_shut_down,
                draining,
                leak_on_finalize: config.leak_on_finalize,
            },
        ))
    }

    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Push a GStreamer event directly through the appsrc element, bypassing
    /// the feeder thread and `input_tx` channel.  Returns `true` if the event
    /// was delivered.
    ///
    /// Use this for lifecycle events (e.g. `GST_NVEVENT_PAD_DELETED`) that must
    /// reach downstream elements even when a shutdown is about to begin.
    pub fn send_event_direct(&self, event: gst::Event) -> bool {
        if let Some(ref appsrc) = self.appsrc {
            appsrc.send_event(event)
        } else {
            false
        }
    }

    /// `true` once [`graceful_shutdown`](Self::graceful_shutdown) or
    /// [`shutdown`](Self::shutdown) has been called.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    /// Graceful shutdown: send EOS through `input_tx` (ordered after any queued input),
    /// drain `output_rx` until `PipelineOutput::Eos` or `timeout`, then stop the pipeline.
    ///
    /// Returns all outputs received before the terminal EOS (the EOS itself is not included).
    /// On timeout, sets the shutdown flag, joins threads, and returns whatever was collected.
    pub fn graceful_shutdown(
        &mut self,
        timeout: Duration,
        input_tx: &Sender<PipelineInput>,
        output_rx: &Receiver<PipelineOutput>,
    ) -> Result<Vec<PipelineOutput>, PipelineError> {
        self.draining.store(true, Ordering::Release);

        input_tx
            .send(PipelineInput::Eos)
            .map_err(|_| PipelineError::ChannelDisconnected)?;

        let mut out = Vec::new();
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match output_rx.recv_timeout(remaining) {
                Ok(PipelineOutput::Eos) => {
                    self.finish_shutdown_after_threads()?;
                    return Ok(out);
                }
                Ok(PipelineOutput::Error(e)) => {
                    out.push(PipelineOutput::Error(e));
                    self.is_shut_down.store(true, Ordering::Release);
                    self.force_shutdown_join();
                    self.set_null_and_release()?;
                    return Ok(out);
                }
                Ok(other) => out.push(other),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    self.is_shut_down.store(true, Ordering::Release);
                    self.force_shutdown_join();
                    self.set_null_and_release()?;
                    return Err(PipelineError::ChannelDisconnected);
                }
            }
        }

        // Timeout or partial drain: force threads down.
        self.is_shut_down.store(true, Ordering::Release);
        self.force_shutdown_join();
        self.set_null_and_release()?;
        Ok(out)
    }

    fn finish_shutdown_after_threads(&mut self) -> Result<(), PipelineError> {
        self.is_shut_down.store(true, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        self.join_all_threads();
        self.set_null_and_release()
    }

    fn force_shutdown_join(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(appsrc) = &self.appsrc {
            let _ = appsrc.end_of_stream();
        }
        self.join_all_threads();
    }

    fn join_all_threads(&mut self) {
        if let Some(handle) = self.feeder_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.watchdog_thread.take() {
            let _ = handle.join();
        }
    }

    /// Transition the pipeline to Null, then release GObjects via
    /// [`release_gobjects`](Self::release_gobjects).
    fn set_null_and_release(&mut self) -> Result<(), PipelineError> {
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Null)
                .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;
        }
        self.release_gobjects();
        Ok(())
    }

    /// Either finalize or intentionally leak GObjects depending on
    /// [`leak_on_finalize`](Self::leak_on_finalize).
    ///
    /// When `false` (default), objects are enqueued on the shared [`FINALIZER`]
    /// thread and the caller blocks until finalization completes.
    ///
    /// When `true`, objects are leaked with [`std::mem::forget`].  Callers must
    /// ensure essential resource cleanup happens *before* this call (e.g.
    /// sending `GST_NVEVENT_PAD_DELETED` for all sources).
    fn release_gobjects(&mut self) {
        if self.leak_on_finalize {
            if let Some(appsrc) = self.appsrc.take() {
                warn!(
                    "pipeline '{}': leaking AppSrc GObject (leak_on_finalize=true)",
                    self.name
                );
                std::mem::forget(appsrc);
            }
            if let Some(pipeline) = self.pipeline.take() {
                warn!(
                    "pipeline '{}': leaking gst::Pipeline GObject (leak_on_finalize=true)",
                    self.name
                );
                std::mem::forget(pipeline);
            }
        } else {
            let done = enqueue_release(&self.name, self.appsrc.take(), self.pipeline.take());
            wait_for_finalization(&self.name, &done);
        }
    }

    pub fn shutdown(&mut self) -> Result<(), PipelineError> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.draining.store(true, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        if let Some(ref appsrc) = self.appsrc {
            let _ = appsrc.end_of_stream();
        }

        // Transition to Null BEFORE joining threads: this stops all GStreamer
        // streaming, which unblocks any element (e.g. nvtracker) that might be
        // waiting to push data downstream.  The drain thread's try_pull_sample
        // returns immediately once appsink is in Null, so joins complete quickly.
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Null)
                .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;
        }

        self.join_all_threads();
        self.release_gobjects();

        Ok(())
    }
}

impl Drop for GstPipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

// ── GObject finalizer thread ────────────────────────────────────────

/// Work item for the singleton finalizer thread.
struct FinalizerItem {
    label: String,
    appsrc: Option<AppSrc>,
    pipeline: Option<gst::Pipeline>,
    done: Sender<()>,
}

/// Maximum number of pending finalization items.  If the queue is full
/// (because a previous finalization is blocking), the objects are leaked
/// instead of hanging the caller — identical to the old `ManuallyDrop`
/// behaviour.
const FINALIZER_QUEUE_CAPACITY: usize = 64;

/// Process-wide singleton: a bounded channel feeding a single background
/// thread that performs GObject finalization.
///
/// The thread drops items sequentially.  If a particular DeepStream
/// element blocks in `GObject::finalize` (e.g. `nvtracker` CUDA worker
/// shutdown), the thread stalls and the bounded queue fills up; any
/// further items are leaked with a warning via the `try_send` fallback
/// in [`enqueue_release`] — matching the safety of the old `ManuallyDrop`
/// approach.
static FINALIZER: OnceLock<Sender<FinalizerItem>> = OnceLock::new();

fn finalizer_sender() -> &'static Sender<FinalizerItem> {
    FINALIZER.get_or_init(|| {
        let (tx, rx) = channel::bounded::<FinalizerItem>(FINALIZER_QUEUE_CAPACITY);
        std::thread::Builder::new()
            .name("gst-finalizer".into())
            .spawn(move || {
                while let Ok(item) = rx.recv() {
                    info!(
                        "gst-finalizer: releasing GObjects for pipeline '{}'",
                        item.label
                    );
                    let start = Instant::now();
                    let label = item.label;
                    drop(item.appsrc);
                    drop(item.pipeline);
                    info!(
                        "gst-finalizer: released GObjects for pipeline '{label}' in {:.1?}",
                        start.elapsed()
                    );
                    let _ = item.done.send(());
                }
            })
            .expect("failed to spawn gst-finalizer thread");
        tx
    })
}

/// Enqueue GStreamer GObjects for finalization on the shared background thread
/// and return a [`Receiver`] that fires when finalization completes.
///
/// The caller should block on the receiver with [`FINALIZE_WAIT_TIMEOUT`] so
/// that the test/process does not exit while the finalizer is mid-drop (which
/// would cause SIGSEGV on GPU/CUDA teardown).
///
/// If the queue is full (a previous finalization is stuck), the GObjects are
/// leaked with a warning — matching the safety of the old `ManuallyDrop`
/// approach — and the returned receiver fires immediately.
fn enqueue_release(
    label: &str,
    appsrc: Option<AppSrc>,
    pipeline: Option<gst::Pipeline>,
) -> Receiver<()> {
    let (done_tx, done_rx) = channel::bounded(1);

    if appsrc.is_none() && pipeline.is_none() {
        let _ = done_tx.send(());
        return done_rx;
    }
    let item = FinalizerItem {
        label: label.to_string(),
        appsrc,
        pipeline,
        done: done_tx,
    };
    match finalizer_sender().try_send(item) {
        Ok(()) => {}
        Err(TrySendError::Full(item)) => {
            warn!(
                "gst-finalizer: queue full, leaking GObjects for pipeline '{}' \
                 (a previous finalization is likely stuck)",
                item.label
            );
            std::mem::forget(item);
        }
        Err(TrySendError::Disconnected(item)) => {
            warn!(
                "gst-finalizer: thread gone, leaking GObjects for pipeline '{}'",
                item.label
            );
            std::mem::forget(item);
        }
    }
    done_rx
}

/// Block until the finalizer thread finishes dropping GObjects for the given
/// pipeline.
///
/// We block without timeout because upstream callers (e.g. `NvTracker`) now
/// send `GST_NVEVENT_PAD_DELETED` before shutdown, so `NvMOT_DeInit` completes
/// within seconds rather than hanging indefinitely.  Blocking guarantees the
/// GObjects are fully finalized before the caller returns, which prevents
/// SIGSEGV at process exit (the test binary exiting while the finalizer thread
/// is inside `G_OBJECT_CLASS::finalize`).
fn wait_for_finalization(label: &str, done: &Receiver<()>) {
    match done.recv() {
        Ok(()) => {}
        Err(_) => {
            warn!(
                "gst-finalizer: completion channel disconnected for pipeline '{label}', \
                 GObjects were likely leaked"
            );
        }
    }
}

/// Extract the ordering key from a buffer according to `policy` and verify it
/// is strictly greater than `last_key`. Returns `Some(PipelineError)` on
/// violation and leaves `last_key` unchanged; returns `None` on success after
/// updating `last_key`.
fn check_pts_policy(
    policy: PtsPolicy,
    buffer: &gst::Buffer,
    last_key: &mut Option<u64>,
) -> Option<PipelineError> {
    let key = match policy {
        PtsPolicy::StrictPts => buffer.pts().map(|p| p.nseconds()),
        PtsPolicy::StrictDecodeOrder => buffer
            .dts()
            .map(|d| d.nseconds())
            .or_else(|| buffer.pts().map(|p| p.nseconds())),
    };
    let current = key?;
    if let Some(prev) = *last_key {
        if current <= prev {
            return Some(PipelineError::TimestampOrderViolation {
                policy,
                current_key_ns: current,
                prev_key_ns: prev,
            });
        }
    }
    *last_key = Some(current);
    None
}
