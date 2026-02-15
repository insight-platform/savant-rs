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

/// Send N frames that are never marked ready, so they expire in FIFO order.
/// Verify that frames arrive at dest in exactly the same UUID order they
/// were generated.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn frame_ordering_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 5u32;
    let max_duration = Duration::from_millis(200);

    let ingress_ipc = ipc_addr("ordering", "ingress");
    let egress_ipc = ipc_addr("ordering", "egress");

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
            callbacks: make_callbacks("merge_never_ready"),
            idle_sleep: Duration::from_millis(1),
            queue: QueueConfiguration { max_duration },
        },
    };

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    let frames: Vec<savant_core::primitives::frame::VideoFrameProxy> =
        (0..num_frames).map(|_| gen_frame()).collect();
    let expected_uuids: Vec<uuid::Uuid> = frames.iter().map(|f| f.get_uuid()).collect();

    for f in &frames {
        let msg = Message::video_frame(f);
        source_writer.send_message("test", &msg, &[])?.get()?;
    }

    let deadline = Instant::now() + max_duration + Duration::from_secs(2);
    let mut received_uuids = Vec::new();

    while Instant::now() < deadline && received_uuids.len() < num_frames as usize {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    let f = message.as_video_frame().unwrap();
                    received_uuids.push(f.get_uuid());
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
    }

    assert_eq!(
        received_uuids.len(),
        num_frames as usize,
        "Expected {} frames, got {}",
        num_frames,
        received_uuids.len()
    );
    assert_eq!(
        received_uuids, expected_uuids,
        "Frames arrived out of order"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
