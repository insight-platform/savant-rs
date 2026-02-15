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

/// `merge_handler` marks every frame as immediately ready, but
/// `on_head_ready` returns `None`, which tells the service to drop the
/// frame.  No video frames should arrive at dest.  EOS (sent on an empty
/// queue) should still be forwarded.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn drop_ready_frame_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 3u32;

    let ingress_ipc = ipc_addr("drop_ready", "ingress");
    let egress_ipc = ipc_addr("drop_ready", "egress");

    let mut callbacks = make_callbacks("merge_always_ready");
    callbacks.on_head_ready = "drop_head_ready_handler".into();

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

    for _ in 0..num_frames {
        let msg = Message::video_frame(&gen_frame());
        source_writer.send_message("test", &msg, &[])?.get()?;
    }

    source_writer.send_eos("test")?.get()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    let eos_ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
    assert!(
        eos_ok,
        "EOS should still be forwarded even when frames are dropped"
    );

    thread::sleep(Duration::from_millis(100));
    let mut frames_seen = 0;
    while let Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) =
        dest_reader.try_receive()
    {
        if message.is_video_frame() {
            frames_seen += 1;
        }
    }
    assert_eq!(
        frames_seen, 0,
        "No video frames should arrive when on_head_ready drops them"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
