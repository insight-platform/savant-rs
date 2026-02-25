//! E2E test: Watchdog removes dead workers.
//!
//! Validates that the watchdog detects and removes dead workers,
//! and the engine can re-create a worker for the same source.

mod common;

use common::*;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn e2e_watchdog_reaps_dead_worker() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb {
            count: bypass_count.clone(),
        })),
        on_eviction: Some(Arc::new(TerminateEviction)),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 1,
        },
        callbacks,
    );

    engine
        .set_source_spec(
            "reap-test",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    engine
        .send_frame("reap-test", make_frame("reap-test"), make_gst_buffer())
        .unwrap();

    std::thread::sleep(Duration::from_secs(5));

    engine
        .set_source_spec(
            "reap-test",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    engine
        .send_frame("reap-test", make_frame("reap-test"), make_gst_buffer())
        .expect("second send_frame should succeed after watchdog reaps dead worker");
    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(
        bypass_count.load(Ordering::SeqCst),
        2,
        "bypass should fire for both frames (set_source_spec creates new worker after reap)"
    );

    engine.shutdown();
}
