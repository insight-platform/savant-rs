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

/// Frames are never marked ready, so they expire.  `on_head_expire`
/// returns `None`, suppressing their delivery.  EOS attached to the last
/// frame should still be forwarded.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn drop_expired_frame_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 3u32;
    let max_duration = Duration::from_millis(200);

    let ingress_ipc = ipc_addr("drop_expired", "ingress");
    let egress_ipc = ipc_addr("drop_expired", "egress");

    let mut callbacks = make_callbacks("merge_never_ready");
    callbacks.on_head_expire = "drop_head_expired_handler".into();

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
            queue: QueueConfiguration { max_duration },
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

    let deadline = Instant::now() + max_duration + Duration::from_secs(2);
    let mut eos_seen = false;
    let mut frames_seen = 0usize;

    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    frames_seen += 1;
                } else if message.is_end_of_stream() {
                    eos_seen = true;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
        if eos_seen {
            break;
        }
    }

    assert!(
        eos_seen,
        "EOS should still be forwarded when frames are dropped"
    );
    assert_eq!(
        frames_seen, 0,
        "No video frames should arrive when on_head_expire drops them"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
