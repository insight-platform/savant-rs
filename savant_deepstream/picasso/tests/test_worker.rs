use picasso::prelude::*;
use picasso::spec::PtsResetPolicy;
use picasso::worker::SourceWorker;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrame, VideoFrameTranscodingMethod,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn make_frame(source_id: &str) -> VideoFrame {
    VideoFrame::new(
        source_id,
        (30, 1),
        320,
        240,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        None,
        None,
        (1, 1000000000),
        0,
        None,
        None,
    )
    .unwrap()
}

fn make_gst_buffer() -> gstreamer::Buffer {
    gstreamer::init().unwrap();
    gstreamer::Buffer::new()
}

fn make_surface_view() -> deepstream_buffers::SurfaceView {
    deepstream_buffers::SurfaceView::wrap(make_gst_buffer())
}

struct CountingBypassCb(Arc<AtomicUsize>);

impl OnBypassFrame for CountingBypassCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(_) => {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {}
        }
    }
}

struct TerminateEviction;

impl OnEviction for TerminateEviction {
    fn call(&self, _source_id: &str) -> EvictionDecision {
        EvictionDecision::TerminateImmediately
    }
}

#[test]
fn worker_drop_spec_discards_frames() {
    gstreamer::init().unwrap();

    let callbacks = Arc::new(Callbacks::default());
    let spec = SourceSpec {
        codec: CodecSpec::Drop,
        ..Default::default()
    };
    let worker = SourceWorker::spawn(
        "test-drop".to_string(),
        spec,
        callbacks,
        Duration::from_secs(60),
        16,
        PtsResetPolicy::default(),
    );

    let frame = make_frame("test-drop");
    let view = make_surface_view();
    worker.send_frame(frame, view, None).unwrap();

    std::thread::sleep(Duration::from_millis(100));
    assert!(worker.is_alive());

    drop(worker);
}

#[test]
fn worker_bypass_fires_callback() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb(bypass_count.clone()))),
        ..Default::default()
    };
    let callbacks = Arc::new(callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    let worker = SourceWorker::spawn(
        "test-bypass".to_string(),
        spec,
        callbacks,
        Duration::from_secs(60),
        16,
        PtsResetPolicy::default(),
    );

    for _ in 0..5 {
        let frame = make_frame("test-bypass");
        let view = make_surface_view();
        worker.send_frame(frame, view, None).unwrap();
    }

    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 5);

    drop(worker);
}

#[test]
fn worker_eos_fires_sentinel_for_bypass() {
    gstreamer::init().unwrap();

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
        on_bypass_frame: Some(Arc::new(BypassEosCb(eos_clone))),
        ..Default::default()
    };
    let callbacks = Arc::new(callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    let worker = SourceWorker::spawn(
        "test-eos".to_string(),
        spec,
        callbacks,
        Duration::from_secs(60),
        16,
        PtsResetPolicy::default(),
    );

    worker.send_eos().unwrap();

    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);

    drop(worker);
}

#[test]
fn worker_idle_timeout_terminates() {
    gstreamer::init().unwrap();

    let callbacks = Callbacks {
        on_eviction: Some(Arc::new(TerminateEviction)),
        ..Default::default()
    };
    let callbacks = Arc::new(callbacks);

    let spec = SourceSpec::default();
    let worker = SourceWorker::spawn(
        "test-idle".to_string(),
        spec,
        callbacks,
        Duration::from_millis(200),
        16,
        PtsResetPolicy::default(),
    );

    std::thread::sleep(Duration::from_millis(500));
    assert!(!worker.is_alive());

    drop(worker);
}

#[test]
fn worker_spec_update() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb(bypass_count.clone()))),
        ..Default::default()
    };
    let callbacks = Arc::new(callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Drop,
        ..Default::default()
    };
    let worker = SourceWorker::spawn(
        "test-update".to_string(),
        spec,
        callbacks,
        Duration::from_secs(60),
        16,
        PtsResetPolicy::default(),
    );

    // Send a frame with Drop spec — shouldn't fire bypass
    let frame = make_frame("test-update");
    let view = make_surface_view();
    worker.send_frame(frame, view, None).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 0);

    // Update to Bypass
    let new_spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    worker.send_update_spec(new_spec).unwrap();
    std::thread::sleep(Duration::from_millis(50));

    // Now send frames — should fire bypass
    for _ in 0..3 {
        let frame = make_frame("test-update");
        let view = make_surface_view();
        worker.send_frame(frame, view, None).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 3);

    drop(worker);
}
