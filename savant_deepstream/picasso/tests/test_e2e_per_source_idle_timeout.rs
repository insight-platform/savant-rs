//! E2E test: Per-source idle timeout override.
//!
//! Validates that sources can have different idle timeouts via
//! SourceSpec::idle_timeout_secs.

mod common;

use common::*;
use picasso::message::WorkerMessage;
use picasso::prelude::*;
use picasso::spec::PtsResetPolicy;
use picasso::worker::SourceWorker;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn e2e_per_source_idle_timeout() {
    gstreamer::init().unwrap();

    let callbacks = Callbacks {
        on_eviction: Some(Arc::new(TerminateEviction)),
        ..Default::default()
    };
    let callbacks = Arc::new(callbacks);

    let worker_fast = SourceWorker::spawn(
        "fast".to_string(),
        SourceSpec {
            codec: CodecSpec::Bypass,
            idle_timeout_secs: Some(1),
            ..Default::default()
        },
        callbacks.clone(),
        Duration::from_secs(1),
        16,
        PtsResetPolicy::default(),
    );

    let worker_slow = SourceWorker::spawn(
        "slow".to_string(),
        SourceSpec {
            codec: CodecSpec::Bypass,
            idle_timeout_secs: None,
            ..Default::default()
        },
        callbacks.clone(),
        Duration::from_secs(60),
        16,
        PtsResetPolicy::default(),
    );

    worker_fast
        .send(WorkerMessage::Frame(
            make_frame("fast"),
            make_surface_view(),
            None,
        ))
        .unwrap();
    worker_slow
        .send(WorkerMessage::Frame(
            make_frame("slow"),
            make_surface_view(),
            None,
        ))
        .unwrap();

    std::thread::sleep(Duration::from_secs(3));

    assert!(
        !worker_fast.is_alive(),
        "fast worker (1s timeout) should be dead after 3s"
    );
    assert!(
        worker_slow.is_alive(),
        "slow worker (60s timeout) should still be alive after 3s"
    );

    drop(worker_fast);
    drop(worker_slow);
}
