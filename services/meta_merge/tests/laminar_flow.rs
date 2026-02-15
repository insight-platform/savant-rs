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
use savant_core::primitives::attribute_value::AttributeValueVariant;
use savant_core::primitives::userdata::UserData;
use savant_core::primitives::WithAttributes;
use savant_core::test::gen_frame;
use savant_services_common::job_writer::SinkConfiguration;

/// Two ingress streams both send **the same** frames (same UUIDs).
/// `TwoStreamMergeHandler` returns `true` only when both copies have
/// arrived.  Each sender introduces a deterministic pseudo-random delay
/// within a fixed 30 ms window per frame, simulating realistic jitter.
///
/// Topology:
///   source_a (req) → (rep) ┐
///                           ├─ meta_merge (dealer) → (router) dest
///   source_b (req) → (rep) ┘
#[test]
#[serial]
fn laminar_flow_test() -> Result<()> {
    init_python();
    register_all_handlers();
    UNSUPPORTED_HANDLER_CALLED.store(false, Ordering::SeqCst);

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 10u32;
    let jitter_window = Duration::from_millis(30);

    let ingress_ipc_a = ipc_addr("laminar", "ingress_a");
    let ingress_ipc_b = ipc_addr("laminar", "ingress_b");
    let egress_ipc = ipc_addr("laminar", "egress");

    let conf = ServiceConfiguration {
        ingress: vec![
            make_ingress_conf("ingress_a", &ingress_ipc_a),
            make_ingress_conf("ingress_b", &ingress_ipc_b),
        ],
        egress: EgressConfiguration {
            socket: SinkConfiguration {
                url: format!("dealer+connect:{}", egress_ipc),
                options: None,
            },
        },
        common: CommonConfiguration {
            init: None,
            callbacks: make_callbacks("merge_two_stream"),
            idle_sleep: Duration::from_millis(1),
            queue: QueueConfiguration {
                max_duration: Duration::from_secs(5),
            },
        },
    };

    let mut dest_reader = start_dest_reader(&egress_ipc)?;
    let service_thread = start_service(conf, shutdown.clone());
    let mut writer_a = start_source_writer(&ingress_ipc_a)?;
    let mut writer_b = start_source_writer(&ingress_ipc_b)?;

    // Each route carries its own label; the merge handler is expected to
    // accumulate labels from both routes into the current state.
    let base_frames: Vec<_> = (0..num_frames).map(|_| gen_frame()).collect();

    for (i, frame) in base_frames.iter().enumerate() {
        let mut msg_a = Message::video_frame(frame);
        msg_a.set_labels(vec!["route_a".into()]);

        let mut msg_b = Message::video_frame(frame);
        msg_b.set_labels(vec!["route_b".into()]);

        let delay_a = pseudo_delay(i as u64 * 2, jitter_window);
        let delay_b = pseudo_delay(i as u64 * 2 + 1, jitter_window);

        if delay_a <= delay_b {
            thread::sleep(delay_a);
            writer_a.send_message("test", &msg_a, &[])?.get()?;
            thread::sleep(delay_b - delay_a);
            writer_b.send_message("test", &msg_b, &[])?.get()?;
        } else {
            thread::sleep(delay_b);
            writer_b.send_message("test", &msg_b, &[])?.get()?;
            thread::sleep(delay_a - delay_b);
            writer_a.send_message("test", &msg_a, &[])?.get()?;
        }
    }

    writer_a.send_eos("test")?.get()?;
    let ud = Message::user_data(UserData::new("test"));
    writer_b.send_message("test", &ud, &[])?.get()?;

    writer_a.shutdown()?;
    writer_b.shutdown()?;

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut frames_at_dest = 0usize;
    let mut eos_at_dest = false;

    while Instant::now() < deadline {
        match dest_reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if message.is_video_frame() {
                    let f = message.as_video_frame().unwrap();

                    assert!(
                        f.get_attribute("merge", "head_ready").is_some(),
                        "Frame #{} should have (merge, head_ready)",
                        frames_at_dest
                    );

                    let mc = f
                        .get_attribute("merge", "merge_count")
                        .expect("Frame should have (merge, merge_count)");
                    let vals = mc.values.as_ref();
                    assert_eq!(vals.len(), 1);
                    match vals[0].get() {
                        AttributeValueVariant::Integer(n) => {
                            assert_eq!(*n, 2, "merge_count should be 2, got {}", n)
                        }
                        other => panic!("merge_count should be Integer, got {:?}", other),
                    }

                    assert!(
                        f.get_attribute("merge", "first_ingress").is_some(),
                        "Frame should have (merge, first_ingress)"
                    );
                    assert!(
                        f.get_attribute("merge", "second_ingress").is_some(),
                        "Frame should have (merge, second_ingress)"
                    );

                    let labels = message.get_labels();
                    assert!(
                        labels.contains(&"route_a".to_string()),
                        "Outbound frame should carry label 'route_a', got {:?}",
                        labels
                    );
                    assert!(
                        labels.contains(&"route_b".to_string()),
                        "Outbound frame should carry label 'route_b', got {:?}",
                        labels
                    );

                    frames_at_dest += 1;
                } else if message.is_end_of_stream() {
                    eos_at_dest = true;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }

        if frames_at_dest >= num_frames as usize
            && eos_at_dest
            && UNSUPPORTED_HANDLER_CALLED.load(Ordering::SeqCst)
        {
            break;
        }
    }

    assert_eq!(
        frames_at_dest, num_frames as usize,
        "Expected {} merged frames at dest, got {}",
        num_frames, frames_at_dest
    );
    assert!(eos_at_dest, "EOS should have been propagated to dest");
    assert!(
        UNSUPPORTED_HANDLER_CALLED.load(Ordering::SeqCst),
        "unsupported_message_handler should have been called for UserData"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
