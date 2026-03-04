//! E2E test: Multi-source with staggered EOS.
//!
//! Simulates production deployment where multiple camera sources start
//! and stop at different times.

mod common;

use common::*;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn e2e_sustained_multi_source_eos_interleave() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb {
            count: bypass_count.clone(),
        })),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: Arc::new(AtomicUsize::new(0)),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    for src in &["s1", "s2", "s3", "s4"] {
        engine.set_source_spec(src, spec.clone()).unwrap();
    }

    for _ in 0..10 {
        for src in &["s1", "s2", "s3", "s4"] {
            engine
                .send_frame(src, make_frame(src), make_surface_view(), None)
                .unwrap();
        }
    }
    std::thread::sleep(Duration::from_millis(200));

    engine.send_eos("s1").unwrap();
    for _ in 0..5 {
        for src in &["s2", "s3", "s4"] {
            engine
                .send_frame(src, make_frame(src), make_surface_view(), None)
                .unwrap();
        }
    }

    engine.send_eos("s2").unwrap();
    for _ in 0..5 {
        for src in &["s3", "s4"] {
            engine
                .send_frame(src, make_frame(src), make_surface_view(), None)
                .unwrap();
        }
    }

    engine.send_eos("s3").unwrap();
    engine.send_eos("s4").unwrap();

    std::thread::sleep(Duration::from_millis(300));
    engine.shutdown();

    assert_eq!(bypass_count.load(Ordering::SeqCst), 10 * 4 + 5 * 3 + 5 * 2);
    assert_eq!(eos_count.load(Ordering::SeqCst), 4);
}
