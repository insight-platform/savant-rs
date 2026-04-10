use std::collections::HashMap;
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

pub struct GstPipeline {
    pipeline: gst::Pipeline,
    appsrc: AppSrc,
    shutdown: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    feeder_thread: Option<JoinHandle<()>>,
    drain_thread: Option<JoinHandle<()>>,
    watchdog_thread: Option<JoinHandle<()>>,
}

impl GstPipeline {
    pub fn start(
        config: PipelineConfig,
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

        let (input_tx, input_rx) = channel::bounded::<PipelineInput>(config.input_channel_capacity);
        let (output_tx, output_rx) =
            channel::bounded::<PipelineOutput>(config.output_channel_capacity);

        // Capture all downstream non-EOS events that arrive to appsink.
        let event_out = output_tx.clone();
        appsink_sink_pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
            if let Some(event) = info.event() {
                if event.type_() != gst::EventType::Eos {
                    let _ = event_out.send(PipelineOutput::Event(event.to_owned()));
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
                                if drain_tx.send(PipelineOutput::Buffer(buffer)).is_err() {
                                    break;
                                }
                            }
                        }
                        None if drain_sink.is_eos() => {
                            let _ = drain_tx.send(PipelineOutput::Eos);
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
                                        let _ = drain_tx.send(PipelineOutput::Error(
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
                pipeline,
                appsrc,
                shutdown,
                failed,
                feeder_thread: Some(feeder_thread),
                drain_thread: Some(drain_thread),
                watchdog_thread,
            },
        ))
    }

    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub fn shutdown(&mut self) -> Result<(), PipelineError> {
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

        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| PipelineError::StateChangeFailed(format!("set Null: {:?}", e)))?;
        Ok(())
    }
}

impl Drop for GstPipeline {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
