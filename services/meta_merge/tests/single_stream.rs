mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serial_test::serial;

use common::*;
use meta_merge::configuration::{
    CommonConfiguration, EgressConfiguration, QueueConfiguration, ServiceConfiguration,
};
use savant_core::primitives::WithAttributes;
use savant_services_common::job_writer::SinkConfiguration;

/// Single ingress stream.  `merge_handler` marks every frame ready
/// immediately, so frames flow through `head_ready_handler` → `send_handler`
/// without delay.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn single_stream_test() -> Result<()> {
    init_python();
    register_all_handlers();
    UNSUPPORTED_HANDLER_CALLED.store(false, Ordering::SeqCst);

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 3u32;

    let ingress_ipc = ipc_addr("single", "ingress");
    let egress_ipc = ipc_addr("single", "egress");

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

    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    for step in test_steps(num_frames) {
        let deadline = Instant::now() + Duration::from_secs(2);
        step.send(&source_writer)?;

        if step.is_eos {
            let ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
            assert!(ok, "EOS was not delivered within 2 s");
            eos_at_dest = true;
        } else if step.message.as_ref().is_some_and(|m| m.is_video_frame()) {
            let ok = wait_for_dest(&dest_reader, deadline, |m| {
                if !m.is_video_frame() {
                    return false;
                }
                let f = m.as_video_frame().unwrap();
                assert!(
                    f.get_attribute("merge", "head_ready").is_some(),
                    "Frame should have (merge, head_ready) attribute"
                );
                true
            });
            assert!(ok, "frame was not delivered within 2 s");
            frames_at_dest += 1;
        } else {
            wait_for_unsupported(deadline);
        }
    }

    assert_eq!(frames_at_dest, num_frames as usize);
    assert!(eos_at_dest, "EOS should have been propagated to dest");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
