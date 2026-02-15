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
use savant_services_common::job_writer::SinkConfiguration;

/// Send EOS without any preceding frames.  The queue is empty so the
/// service should forward the EOS directly to the egress socket.
///
/// Topology: source (req) → (rep) meta_merge (dealer) → (router) dest
#[test]
#[serial]
fn eos_on_empty_queue_test() -> Result<()> {
    init_python();
    register_all_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));

    let ingress_ipc = ipc_addr("eos_empty", "ingress");
    let egress_ipc = ipc_addr("eos_empty", "egress");

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

    source_writer.send_eos("test")?.get()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    let eos_ok = wait_for_dest(&dest_reader, deadline, |m| m.is_end_of_stream());
    assert!(eos_ok, "EOS should be forwarded immediately on empty queue");

    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked")?;
    source_writer.shutdown()?;
    dest_reader.shutdown()?;
    unregister_all_handlers();

    Ok(())
}
