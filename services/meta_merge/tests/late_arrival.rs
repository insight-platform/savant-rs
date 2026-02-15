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

/// Send frame A via ingress 1 with `never_ready` merge handler and a short
/// `max_duration`.  Wait for A to expire and arrive at dest.  Then push a
/// fresh frame B (so the queue is non-empty) and re-send A via ingress 2.
/// Since A's UUID < head (B), it is detected as a late arrival and
/// `on_late_arrival` is called.
///
/// Topology:
///   source_1 (req) → (rep) ┐
///                           ├─ meta_merge (dealer) → (router) dest
///   source_2 (req) → (rep) ┘
#[test]
#[serial]
fn late_arrival_test() -> Result<()> {
    init_python();
    register_all_handlers();
    LATE_ARRIVAL_CALLED.store(false, Ordering::SeqCst);

    let shutdown = Arc::new(AtomicBool::new(false));
    let max_duration = Duration::from_millis(200);

    let ingress_ipc_1 = ipc_addr("late", "ingress1");
    let ingress_ipc_2 = ipc_addr("late", "ingress2");
    let egress_ipc = ipc_addr("late", "egress");

    let mut callbacks = make_callbacks("merge_never_ready");
    callbacks.on_late_arrival = "tracking_late_arrival_handler".into();

    let conf = ServiceConfiguration {
        ingress: vec![
            make_ingress_conf("ingress_1", &ingress_ipc_1),
            make_ingress_conf("ingress_2", &ingress_ipc_2),
        ],
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
            queue: QueueConfiguration { max_duration },
        },
    };

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, shutdown.clone());
    let mut writer_1 = start_source_writer(&ingress_ipc_1)?;
    let mut writer_2 = start_source_writer(&ingress_ipc_2)?;

    let frame_a = gen_frame();
    let uuid_a = frame_a.get_uuid();
    let msg_a = Message::video_frame(&frame_a);

    writer_1.send_message("test", &msg_a, &[])?.get()?;

    let deadline = Instant::now() + max_duration + Duration::from_secs(2);
    let mut first_uuid = None;
    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    let f = message.as_video_frame().unwrap();
                    first_uuid = Some(f.get_uuid());
                    break;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
    }
    assert_eq!(first_uuid, Some(uuid_a), "First expired frame should be A");

    // Push a fresh frame B so the queue is non-empty (B.uuid > A.uuid)
    let frame_b = gen_frame();
    let msg_b = Message::video_frame(&frame_b);
    writer_1.send_message("test", &msg_b, &[])?.get()?;

    // Re-send frame A from ingress 2 — A.uuid < head (B) → late arrival
    writer_2.send_message("test", &msg_a, &[])?.get()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if LATE_ARRIVAL_CALLED.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        LATE_ARRIVAL_CALLED.load(Ordering::SeqCst),
        "on_late_arrival should have been called for the re-sent frame A"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    writer_1.shutdown()?;
    writer_2.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
