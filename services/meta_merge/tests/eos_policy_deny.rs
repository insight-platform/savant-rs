mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serial_test::serial;

use common::*;
use meta_merge::configuration::{
    CommonConfiguration, EgressConfiguration, IngressConfiguration, QueueConfiguration,
    ServiceConfiguration,
};
use savant_core::message::Message;
use savant_core::test::gen_frame;
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::{SourceConfiguration, SourceOptions, TopicPrefixSpec};

/// Ingress is configured without `EosPolicy::Allow`.  EOS messages are
/// therefore treated as unsupported and routed to `on_unsupported_message`
/// rather than being forwarded to the egress.  Video frames still flow
/// normally.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn eos_policy_deny_test() -> Result<()> {
    init_python();
    register_all_handlers();
    UNSUPPORTED_HANDLER_CALLED.store(false, Ordering::SeqCst);

    let shutdown = Arc::new(AtomicBool::new(false));
    let num_frames = 2u32;

    let ingress_ipc = ipc_addr("eos_deny", "ingress");
    let egress_ipc = ipc_addr("eos_deny", "egress");

    let ingress_conf = IngressConfiguration {
        name: "test_ingress".into(),
        socket: SourceConfiguration {
            url: format!("rep+bind:{}", ingress_ipc),
            options: Some(SourceOptions {
                receive_timeout: Duration::from_millis(100),
                receive_hwm: 1000,
                topic_prefix_spec: TopicPrefixSpec::None,
                source_cache_size: 100,
                fix_ipc_permissions: None,
                inflight_ops: 100,
            }),
        },
        handler: None,
        eos_policy: None,
    };

    let conf = ServiceConfiguration {
        ingress: vec![ingress_conf],
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

    for _ in 0..num_frames {
        let msg = Message::video_frame(&gen_frame());
        source_writer.send_message("test", &msg, &[])?.get()?;
        let deadline = Instant::now() + Duration::from_secs(2);
        assert!(
            wait_for_dest(&dest_reader, deadline, |m| m.is_video_frame()),
            "Frame should arrive at dest"
        );
    }

    source_writer.send_eos("test")?.get()?;
    wait_for_unsupported(Instant::now() + Duration::from_secs(2));

    thread::sleep(Duration::from_millis(200));
    let mut eos_seen = false;
    while let Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) =
        dest_reader.try_receive()
    {
        if message.is_end_of_stream() {
            eos_seen = true;
        }
    }
    assert!(
        !eos_seen,
        "EOS should NOT be forwarded when eos_policy is None"
    );

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
