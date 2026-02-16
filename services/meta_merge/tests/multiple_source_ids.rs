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

/// Frames from two different source_ids are sent on the same ingress.
/// The service maintains independent per-source queues.  Verify that
/// frames and EOS for each source_id are handled independently.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn multiple_source_ids_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));
    let frames_per_source = 3u32;

    let ingress_ipc = ipc_addr("multi_src", "ingress");
    let egress_ipc = ipc_addr("multi_src", "egress");

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
            callbacks: make_callbacks("merge_always_ready"),
            idle_sleep: Duration::from_millis(1),
            queue: QueueConfiguration {
                max_duration: Duration::from_secs(5),
            },
        },
    };

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, shutdown.clone());
    let mut source_writer = start_source_writer(&ingress_ipc)?;

    for _ in 0..frames_per_source {
        let mut fa = gen_frame();
        fa.set_source_id("source_a");
        let msg_a = Message::video_frame(&fa);
        source_writer.send_message("source_a", &msg_a, &[])?.get()?;

        let mut fb = gen_frame();
        fb.set_source_id("source_b");
        let msg_b = Message::video_frame(&fb);
        source_writer.send_message("source_b", &msg_b, &[])?.get()?;
    }

    source_writer.send_eos("source_a")?.get()?;
    source_writer.send_eos("source_b")?.get()?;

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut frames_a = 0usize;
    let mut frames_b = 0usize;
    let mut eos_a = false;
    let mut eos_b = false;

    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    let f = message.as_video_frame().unwrap();
                    match f.get_source_id().as_str() {
                        "source_a" => frames_a += 1,
                        "source_b" => frames_b += 1,
                        other => panic!("Unexpected source_id: {}", other),
                    }
                } else if message.is_end_of_stream() {
                    let eos = message.as_end_of_stream().unwrap();
                    match eos.get_source_id() {
                        "source_a" => eos_a = true,
                        "source_b" => eos_b = true,
                        other => panic!("Unexpected EOS source_id: {}", other),
                    }
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }

        if frames_a >= frames_per_source as usize
            && frames_b >= frames_per_source as usize
            && eos_a
            && eos_b
        {
            break;
        }
    }

    assert_eq!(frames_a, frames_per_source as usize, "source_a frame count");
    assert_eq!(frames_b, frames_per_source as usize, "source_b frame count");
    assert!(eos_a, "EOS for source_a should arrive");
    assert!(eos_b, "EOS for source_b should arrive");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
