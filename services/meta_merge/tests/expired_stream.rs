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
use savant_core::primitives::WithAttributes;
use savant_services_common::job_writer::SinkConfiguration;

/// Two ingress streams are configured but only one actually sends data.
/// Because the second stream never contributes, `merge_handler` always
/// returns `false` and frames sit in the queue until `max_duration`
/// expires.  They then flow through `head_expired_handler` (which sets
/// `(merge, head_expired)`) → `send_handler`.
///
/// Topology:
///    source (req) → (rep) ┐
///                         ├─ meta_merge (dealer) → (router) dest
///        (nobody) → (rep) ┘
#[test]
#[serial]
fn expired_stream_test() -> Result<()> {
    init_python();
    register_all_handlers();
    UNSUPPORTED_HANDLER_CALLED.store(false, Ordering::SeqCst);

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 3u32;
    let max_duration = Duration::from_millis(200);

    let ingress_ipc_0 = ipc_addr("expired", "ingress0");
    let ingress_ipc_1 = ipc_addr("expired", "ingress1");
    let egress_ipc = ipc_addr("expired", "egress");

    let conf = ServiceConfiguration {
        ingress: vec![
            make_ingress_conf("active_ingress", &ingress_ipc_0),
            make_ingress_conf("silent_ingress", &ingress_ipc_1),
        ],
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
    let mut source_writer = start_source_writer(&ingress_ipc_0)?;

    for step in frame_eos_steps(num_frames) {
        step.send(&source_writer)?;
    }
    let send_done = Instant::now();

    let deadline = send_done + max_duration + Duration::from_secs(2);
    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    let f = message.as_video_frame().unwrap();
                    assert!(
                        f.get_attribute("merge", "head_expired").is_some(),
                        "Expired frame should have (merge, head_expired) attribute"
                    );
                    frames_at_dest += 1;
                } else if message.is_end_of_stream() {
                    eos_at_dest = true;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }

        if frames_at_dest >= num_frames as usize && eos_at_dest {
            break;
        }
    }

    let elapsed = send_done.elapsed();

    assert_eq!(
        frames_at_dest, num_frames as usize,
        "Expected {} frames at dest, got {}",
        num_frames, frames_at_dest
    );
    assert!(eos_at_dest, "EOS should have been propagated to dest");
    assert!(
        elapsed >= max_duration - Duration::from_millis(50),
        "Frames should have been delayed by at least ~max_duration ({:?}), \
         but arrived after {:?}",
        max_duration,
        elapsed
    );

    TestStep::unsupported().send(&source_writer)?;
    wait_for_unsupported(Instant::now() + Duration::from_secs(2));

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
