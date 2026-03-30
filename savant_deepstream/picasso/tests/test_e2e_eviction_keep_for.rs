//! E2E test: Idle eviction with KeepFor then Terminate.
//!
//! Simulates a source going temporarily idle; the system keeps it alive
//! for a grace period, then terminates if still idle.

mod common;

use common::*;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Eviction callback: first call KeepFor(1), second call Terminate.
struct KeepForThenTerminate(Arc<AtomicUsize>);

impl OnEviction for KeepForThenTerminate {
    fn call(&self, _source_id: &str) -> EvictionDecision {
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            EvictionDecision::KeepFor(1)
        } else {
            EvictionDecision::Terminate
        }
    }
}

#[test]
fn e2e_eviction_keep_for_then_terminate() {
    gstreamer::init().unwrap();

    let eviction_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let eos_clone = eos_count.clone();

    struct BypassEosCb(Arc<AtomicUsize>);
    impl OnBypassFrame for BypassEosCb {
        fn call(&self, output: OutputMessage) {
            if matches!(output, OutputMessage::EndOfStream(_)) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    let callbacks = Callbacks {
        on_eviction: Some(Arc::new(KeepForThenTerminate(eviction_count.clone()))),
        on_bypass_frame: Some(Arc::new(BypassEosCb(eos_clone))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 1,
            ..Default::default()
        },
        callbacks,
    );

    engine
        .set_source_spec(
            "evict-test",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    engine
        .send_frame(
            "evict-test",
            make_frame("evict-test"),
            make_surface_view(),
            None,
        )
        .unwrap();

    std::thread::sleep(Duration::from_secs(3));

    assert_eq!(
        eviction_count.load(Ordering::SeqCst),
        2,
        "eviction callback should fire exactly 2 times"
    );
    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        1,
        "EOS sentinel should be delivered when Terminate is returned"
    );

    engine.shutdown();
}
