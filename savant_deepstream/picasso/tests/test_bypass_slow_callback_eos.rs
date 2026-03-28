//! Tests for bypass-mode EOS delivery through `on_bypass_frame` and
//! correct ordering when the callback is slow.

mod common;

use common::*;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Bypass callback that counts frames and EOS separately, with a
/// configurable per-call delay to simulate slow processing.
struct SlowBypassCb {
    frame_count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
    delay: Duration,
}

impl OnBypassFrame for SlowBypassCb {
    fn call(&self, output: OutputMessage) {
        std::thread::sleep(self.delay);
        match output {
            OutputMessage::VideoFrame(_) => {
                self.frame_count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

/// Bypass callback that just counts frames and EOS without delay.
struct BypassCountCb {
    frame_count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
}

impl OnBypassFrame for BypassCountCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(_) => {
                self.frame_count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

/// EOS arrives through `on_bypass_frame` for a bypass-mode source.
#[test]
fn bypass_eos_delivered_via_on_bypass_frame() {
    gstreamer::init().unwrap();

    let frame_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(BypassCountCb {
            frame_count: frame_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        },
        callbacks,
    );

    engine
        .set_source_spec(
            "src",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    for _ in 0..3 {
        let frame = make_frame("src");
        let view = make_surface_view();
        engine.send_frame("src", frame, view, None).unwrap();
    }

    engine.send_eos("src").unwrap();

    std::thread::sleep(Duration::from_millis(300));
    engine.shutdown();

    assert_eq!(frame_count.load(Ordering::SeqCst), 3);
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
}

/// Send N frames (fewer than internal queue size) + EOS rapidly to a
/// bypass-mode engine with a slow callback. All frames and EOS must be
/// delivered after the callback finishes processing them.
#[test]
fn bypass_slow_callback_delivers_all_frames_and_eos() {
    gstreamer::init().unwrap();

    const N: usize = 5;
    let delay_per_call = Duration::from_millis(200);

    let frame_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(SlowBypassCb {
            frame_count: frame_count.clone(),
            eos_count: eos_count.clone(),
            delay: delay_per_call,
        })),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 60,
            ..Default::default()
        },
        callbacks,
    );

    engine
        .set_source_spec(
            "slow-src",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    for _ in 0..N {
        let frame = make_frame("slow-src");
        let view = make_surface_view();
        engine.send_frame("slow-src", frame, view, None).unwrap();
    }
    engine.send_eos("slow-src").unwrap();

    // All N frames + 1 EOS processed at ~200ms each = ~1200ms.
    // Wait with generous margin.
    let wait = delay_per_call * (N as u32 + 1) + Duration::from_secs(1);
    std::thread::sleep(wait);
    engine.shutdown();

    assert_eq!(
        frame_count.load(Ordering::SeqCst),
        N,
        "all {N} frames must be delivered despite slow callback"
    );
    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        1,
        "EOS must be delivered after all frames"
    );
}
