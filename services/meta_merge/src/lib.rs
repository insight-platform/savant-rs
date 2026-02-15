pub mod configuration;
pub mod egress;
pub mod ingress;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use egress::processor::EgressProcessor;
use ingress::IngressMessage;
use log::debug;

use crate::configuration::ServiceConfiguration;
use crate::egress::egress::Egress;
use crate::ingress::Ingress;
use savant_core::transport::zeromq::NonBlockingWriter;

/// Result of processing a batch of ingress messages.
pub struct ProcessResult {
    /// Number of video frames processed.
    pub frames: usize,
    /// Number of EOS messages processed.
    pub eos: usize,
}

/// Dispatch a batch of ingress messages to the egress processor.
///
/// Video frames are forwarded via [`EgressProcessor::process_frame`] and
/// end-of-stream messages via [`EgressProcessor::process_eos`].
/// Other message types (e.g. `UserData`) are already handled inside
/// [`Ingress::get`] and are not present in the messages vector.
pub fn process_ingress_messages(
    messages: Vec<IngressMessage>,
    processor: &mut EgressProcessor,
) -> Result<ProcessResult> {
    let mut frames = 0usize;
    let mut eos = 0usize;

    for ingress_message in messages {
        let topic = &ingress_message.topic;
        let message = ingress_message.message;
        let data = ingress_message.data;
        let ingress_name = &ingress_message.ingress_name;

        if message.is_video_frame() {
            let frame_proxy = message.as_video_frame().unwrap();
            let frame = savant_core_py::primitives::frame::VideoFrame(frame_proxy);
            let labels = message.get_labels();
            processor.process_frame(ingress_name, topic, frame, data, labels)?;
            frames += 1;
        } else if message.is_end_of_stream() {
            let eos_msg = message.as_end_of_stream().unwrap();
            let source_id = eos_msg.get_source_id().to_string();
            let labels = message.get_labels();
            processor.process_eos(source_id, data, labels)?;
            eos += 1;
        }
    }

    Ok(ProcessResult { frames, eos })
}

/// Run the meta_merge service loop.
///
/// Creates ingress reader(s) and egress writer from the configuration,
/// then enters the processing loop. The loop reads messages from ingress,
/// dispatches them to the egress processor, and sends ready/expired heads
/// even when idle (no incoming data).
///
/// When `shutdown` is `Some`, the loop checks the flag on every iteration
/// and exits when it is set to `true`. When `None`, the loop runs
/// indefinitely.
pub fn run_service_loop(
    conf: &ServiceConfiguration,
    shutdown: Option<Arc<AtomicBool>>,
) -> Result<()> {
    let mut ingress = Ingress::new(conf)?;

    let buffer = Egress::new(conf.common.queue.max_duration);
    let writer = NonBlockingWriter::try_from(&conf.egress.socket)?;
    let mut processor = EgressProcessor::new(buffer, conf.common.callbacks.clone(), writer);

    loop {
        if let Some(ref flag) = shutdown {
            if flag.load(Ordering::SeqCst) {
                debug!("Shutdown flag set, exiting service loop");
                break;
            }
        }

        let messages = ingress.get()?;
        if messages.is_empty() {
            // Even when idle, poll for expired heads
            processor.send_ready()?;
            std::thread::sleep(conf.common.idle_sleep);
            debug!(
                "No messages received, sleeping for {:?}",
                conf.common.idle_sleep
            );
            continue;
        }

        process_ingress_messages(messages, &mut processor)?;

        // After processing all messages, send any ready heads
        processor.send_ready()?;
    }

    Ok(())
}
