mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serial_test::serial;

use common::*;
use meta_merge::configuration::{
    CommonConfiguration, EgressConfiguration, QueueConfiguration, ServiceConfiguration,
};
use savant_core::message::Message;
use savant_core::test::gen_frame;
use savant_services_common::job_writer::SinkConfiguration;

/// `on_send` handler returns `Some("custom_topic")`.  Verify that frames
/// arrive at dest under the overridden topic instead of the default
/// (source_id).
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn topic_override_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));

    let ingress_ipc = ipc_addr("topic_ovr", "ingress");
    let egress_ipc = ipc_addr("topic_ovr", "egress");

    let mut callbacks = make_callbacks("merge_always_ready");
    callbacks.on_send = Some("topic_override_send_handler".into());

    let conf = ServiceConfiguration {
        ingress: vec![make_ingress_conf("test_ingress", &ingress_ipc)],
        egress: EgressConfiguration {
            socket: SinkConfiguration {
                url: format!("dealer+connect:{}", egress_ipc),
                options: None,
            },
        },
        common: CommonConfiguration {
            init: None,
            callbacks,
            idle_sleep: Duration::from_millis(1),
            queue: QueueConfiguration {
                max_duration: Duration::from_secs(5),
            },
        },
    };

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    let msg = Message::video_frame(&gen_frame());
    source_writer.send_message("test", &msg, &[])?.get()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut topic_ok = false;
    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message {
                message,
                topic,
                ..
            })) => {
                if message.is_video_frame() {
                    let topic_str = String::from_utf8_lossy(&topic);
                    assert_eq!(
                        topic_str, "custom_topic",
                        "Expected topic 'custom_topic', got '{}'",
                        topic_str
                    );
                    topic_ok = true;
                    break;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
    }
    assert!(topic_ok, "Frame with custom topic should arrive at dest");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
