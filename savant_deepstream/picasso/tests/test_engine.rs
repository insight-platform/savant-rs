use picasso::callbacks::{OnBypassFrame, OnEncodedFrame};
use picasso::message::{BypassOutput, EncodedOutput};
use picasso::spec::{CodecSpec, GeneralSpec, SourceSpec};
use picasso::{Callbacks, PicassoEngine};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn make_frame(source_id: &str) -> VideoFrameProxy {
    let f = VideoFrameProxy::new(
        source_id,
        "30/1",
        320,
        240,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1000000000),
        0,
        None,
        None,
    )
    .unwrap();
    f
}

fn make_gst_buffer() -> gstreamer::Buffer {
    gstreamer::Buffer::new()
}

struct CountingBypassCb {
    count: Arc<AtomicUsize>,
}

impl OnBypassFrame for CountingBypassCb {
    fn call(&self, _output: BypassOutput) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

struct CountingEncodedCb {
    count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
}

impl OnEncodedFrame for CountingEncodedCb {
    fn call(&self, output: EncodedOutput) {
        if output.is_eos {
            self.eos_count.fetch_add(1, Ordering::SeqCst);
        } else {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[test]
fn engine_auto_creates_worker() {
    gstreamer::init().unwrap();

    let callbacks = Callbacks::default();
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let frame = make_frame("auto-create");
    let buf = make_gst_buffer();
    engine.send_frame("auto-create", frame, buf).unwrap();

    std::thread::sleep(Duration::from_millis(100));
    engine.shutdown();
}

#[test]
fn engine_bypass_multi_source() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb {
            count: bypass_count.clone(),
        })),
        ..Default::default()
    };
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    engine.set_source_spec("src-a", spec.clone()).unwrap();
    engine.set_source_spec("src-b", spec).unwrap();

    for _ in 0..3 {
        let frame_a = make_frame("src-a");
        let buf_a = make_gst_buffer();
        engine.send_frame("src-a", frame_a, buf_a).unwrap();

        let frame_b = make_frame("src-b");
        let buf_b = make_gst_buffer();
        engine.send_frame("src-b", frame_b, buf_b).unwrap();
    }

    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 6);

    engine.shutdown();
}

#[test]
fn engine_eos_sends_sentinel() {
    gstreamer::init().unwrap();

    let eos_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: Arc::new(AtomicUsize::new(0)),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    engine.set_source_spec("eos-test", spec).unwrap();

    let frame = make_frame("eos-test");
    let buf = make_gst_buffer();
    engine.send_frame("eos-test", frame, buf).unwrap();

    engine.send_eos("eos-test").unwrap();

    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);

    engine.shutdown();
}

#[test]
fn engine_remove_source() {
    gstreamer::init().unwrap();

    let callbacks = Callbacks::default();
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec::default();
    engine.set_source_spec("removable", spec).unwrap();

    engine.remove_source_spec("removable");
    std::thread::sleep(Duration::from_millis(100));

    engine.shutdown();
}

#[test]
fn engine_shutdown_rejects_new_frames() {
    gstreamer::init().unwrap();

    let callbacks = Callbacks::default();
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);
    engine.shutdown();

    let frame = make_frame("rejected");
    let buf = make_gst_buffer();
    let result = engine.send_frame("rejected", frame, buf);
    assert!(result.is_err());
}

#[test]
fn engine_spec_hot_swap() {
    gstreamer::init().unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(CountingBypassCb {
            count: bypass_count.clone(),
        })),
        ..Default::default()
    };
    let general = GeneralSpec {
        idle_timeout_secs: 60,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let drop_spec = SourceSpec {
        codec: CodecSpec::Drop,
        ..Default::default()
    };
    engine.set_source_spec("swap-test", drop_spec).unwrap();

    // Send frames in Drop mode — no bypass callbacks
    for _ in 0..3 {
        let frame = make_frame("swap-test");
        let buf = make_gst_buffer();
        engine.send_frame("swap-test", frame, buf).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 0);

    // Swap to Bypass
    let bypass_spec = SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    };
    engine.set_source_spec("swap-test", bypass_spec).unwrap();
    std::thread::sleep(Duration::from_millis(100));

    // Now frames should trigger bypass callback
    for _ in 0..3 {
        let frame = make_frame("swap-test");
        let buf = make_gst_buffer();
        engine.send_frame("swap-test", frame, buf).unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(bypass_count.load(Ordering::SeqCst), 3);

    engine.shutdown();
}
