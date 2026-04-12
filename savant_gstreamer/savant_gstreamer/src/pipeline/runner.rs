use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Receiver, RecvTimeoutError, Sender};

use glib::translate::from_glib_none;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::{AppSink, AppSrc};
use parking_lot::Mutex;

use crate::pipeline::bridge_meta::bridge_savant_id_meta_across;
use crate::pipeline::config::PipelineConfig;
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

/// GStreamer pipeline with appsrc/appsink and background threads.
///
/// The inner `gst::Pipeline` is wrapped in [`ManuallyDrop`] because
/// DeepStream elements (nvtracker, nvinfer) spawn CUDA worker threads that
/// may block indefinitely during GObject finalization. After
/// [`shutdown`](Self::shutdown) transitions the pipeline to `Null`, the
/// wrapper is intentionally **not** dropped so the process can exit cleanly.
pub struct GstPipeline {
    pipeline: ManuallyDrop<gst::Pipeline>,
    appsrc: ManuallyDrop<AppSrc>,
    shutdown: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    feeder_thread: Option<JoinHandle<()>>,
    drain_thread: Option<JoinHandle<()>>,
    watchdog_thread: Option<JoinHandle<()>>,
    is_shut_down: bool,
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
        let in_flight: Arc<Mutex<HashMap<u64, Instant>>> = Arc::new(Mutex::new(HashMap::new()));

        let feeder_shutdown = shutdown.clone();
        let feeder_failed = failed.clone();
        let feeder_appsrc = appsrc.clone();
        let feeder_tx = output_tx.clone();
        let feeder_in_flight = in_flight.clone();
        let feeder_name = format!("{}-feeder", config.name);
        let feeder_thread = std::thread::Builder::new()
            .name(feeder_name)
            .spawn(move || loop {
                if feeder_shutdown.load(Ordering::Acquire) {
                    break;
                }
                match input_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(PipelineInput::Buffer(buffer)) => {
                        let pts = buffer.pts().map(|p| p.nseconds());
                        if let Some(pts_ns) = pts {
                            feeder_in_flight.lock().insert(pts_ns, Instant::now());
                        }
                        if let Err(e) = feeder_appsrc.push_buffer(buffer) {
                            if let Some(pts_ns) = pts {
                                feeder_in_flight.lock().remove(&pts_ns);
                            }
                            feeder_failed.store(true, Ordering::Release);
                            let _ = feeder_tx.send(PipelineOutput::Error(
                                PipelineError::RuntimeError(format!("appsrc push failed: {:?}", e)),
                            ));
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

                // Send to the bounded output channel, but bail out if the
                // shutdown flag is raised while the channel is full.
                let send_or_shutdown =
                    |msg: PipelineOutput| -> std::result::Result<(), PipelineOutput> {
                        let mut m = msg;
                        loop {
                            if drain_shutdown.load(Ordering::Acquire) {
                                return Err(m);
                            }
                            match drain_tx.send_timeout(m, Duration::from_millis(50)) {
                                Ok(()) => return Ok(()),
                                Err(channel::SendTimeoutError::Timeout(ret)) => m = ret,
                                Err(channel::SendTimeoutError::Disconnected(_)) => {
                                    return Err(PipelineOutput::Eos)
                                }
                            }
                        }
                    };

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
                                if send_or_shutdown(PipelineOutput::Buffer(buffer)).is_err() {
                                    break;
                                }
                            }
                        }
                        None if drain_sink.is_eos() => {
                            let _ = send_or_shutdown(PipelineOutput::Eos);
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
                                        let _ = send_or_shutdown(PipelineOutput::Error(
                                            PipelineError::RuntimeError(error_msg),
                                        ));
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
                pipeline: ManuallyDrop::new(pipeline),
                appsrc: ManuallyDrop::new(appsrc),
                shutdown,
                failed,
                feeder_thread: Some(feeder_thread),
                drain_thread: Some(drain_thread),
                watchdog_thread,
                is_shut_down: false,
            },
        ))
    }

    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
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
                    self.is_shut_down = true;
                    self.force_shutdown_join();
                    self.pipeline.set_state(gst::State::Null).map_err(|e| {
                        PipelineError::StateChangeFailed(format!("set Null: {:?}", e))
                    })?;
                    return Ok(out);
                }
                Ok(other) => out.push(other),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    self.is_shut_down = true;
                    self.force_shutdown_join();
                    self.pipeline.set_state(gst::State::Null).map_err(|e| {
                        PipelineError::StateChangeFailed(format!("set Null: {:?}", e))
                    })?;
                    return Err(PipelineError::ChannelDisconnected);
                }
            }
        }

        // Timeout or partial drain: force threads down.
        self.is_shut_down = true;
        self.force_shutdown_join();
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;
        Ok(out)
    }

    fn finish_shutdown_after_threads(&mut self) -> Result<(), PipelineError> {
        self.is_shut_down = true;
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.feeder_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.watchdog_thread.take() {
            let _ = handle.join();
        }
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;
        Ok(())
    }

    fn force_shutdown_join(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.appsrc.end_of_stream();
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

    pub fn shutdown(&mut self) -> Result<(), PipelineError> {
        if self.is_shut_down {
            return Ok(());
        }
        self.is_shut_down = true;
        self.shutdown.store(true, Ordering::Release);
        let _ = self.appsrc.end_of_stream();

        // Transition to Null BEFORE joining threads: this stops all GStreamer
        // streaming, which unblocks any element (e.g. nvtracker) that might be
        // waiting to push data downstream.  The drain thread's try_pull_sample
        // returns immediately once appsink is in Null, so joins complete quickly.
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;

        if let Some(handle) = self.feeder_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.watchdog_thread.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Drop for GstPipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
