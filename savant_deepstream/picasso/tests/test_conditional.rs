//! End-to-end test for [`ConditionalSpec`] attribute gates.
//!
//! Verifies two independent gates:
//!
//! 1. **`encode_attribute`** — when set, frames *without* the attribute are
//!    silently dropped (no encoded output).  Frames that carry the attribute
//!    pass through the full pipeline normally.
//!
//! 2. **`render_attribute`** — when set, frames *without* the attribute are
//!    still encoded but Skia rendering is skipped.  Frames that carry the
//!    attribute go through the full render-and-encode path.  The `on_render`
//!    callback fires only for those frames.

mod common;

use deepstream_buffers::{BufferGenerator, TransformConfig};
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::WithAttributes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const FRAME_DUR_NS: u64 = 33_333_333;

fn init() {
    let _ = gstreamer::init();
    let _ = cuda_init(0);
}

fn make_encoder_config() -> EncoderConfig {
    common::make_default_encoder_config(W, H)
}

fn make_frame(source_id: &str, idx: u64) -> VideoFrameProxy {
    let frame = VideoFrameProxy::new(
        source_id,
        "30/1",
        W as i64,
        H as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap();
    let mut fm = frame.clone();
    fm.set_pts((idx * FRAME_DUR_NS) as i64).unwrap();
    frame
}

fn make_frame_with_attr(source_id: &str, idx: u64, ns: &str, name: &str) -> VideoFrameProxy {
    let frame = make_frame(source_id, idx);
    let mut fm = frame.clone();
    fm.set_persistent_attribute(ns, name, &None, false, vec![]);
    frame
}

fn make_buffer(gen: &BufferGenerator, idx: u64) -> deepstream_buffers::SurfaceView {
    let shared = gen.acquire(Some(idx as i64)).unwrap();
    shared.set_pts_ns(idx * FRAME_DUR_NS);
    shared.set_duration_ns(FRAME_DUR_NS);
    deepstream_buffers::SurfaceView::from_buffer(&shared, 0).unwrap()
}

struct EncodedCounter(Arc<AtomicUsize>);
impl OnEncodedFrame for EncodedCounter {
    fn call(&self, output: OutputMessage) {
        if matches!(output, OutputMessage::VideoFrame(_)) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct RenderCounter(Arc<AtomicUsize>);
impl OnRender for RenderCounter {
    fn call(
        &self,
        _source_id: &str,
        _renderer: &mut deepstream_buffers::SkiaRenderer,
        _frame: &VideoFrameProxy,
    ) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

// -----------------------------------------------------------------------
// Test: encode_attribute gate
// -----------------------------------------------------------------------

#[test]
fn encode_attribute_gate_drops_frames_without_attr() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedCounter(enc_count.clone()))),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        conditional: ConditionalSpec {
            encode_attribute: Some(("gate".into(), "process".into())),
            render_attribute: None,
        },
        use_on_render: false,
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Send 3 frames WITHOUT the gate attribute — all should be dropped.
    for i in 0..3u64 {
        let frame = make_frame("src", i);
        let buf = make_buffer(&gen, i);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Send 2 frames WITH the gate attribute — these should be encoded.
    for i in 3..5u64 {
        let frame = make_frame_with_attr("src", i, "gate", "process");
        let buf = make_buffer(&gen, i);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Give the pipeline time to process all frames.
    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    assert_eq!(
        encoded, 2,
        "Expected exactly 2 encoded frames (3 dropped, 2 passed), got {encoded}"
    );
}

// -----------------------------------------------------------------------
// Test: render_attribute gate
// -----------------------------------------------------------------------

#[test]
fn render_attribute_gate_skips_rendering_without_attr() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let render_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedCounter(enc_count.clone()))),
        on_render: Some(Arc::new(RenderCounter(render_count.clone()))),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        conditional: ConditionalSpec {
            encode_attribute: None,
            render_attribute: Some(("gate".into(), "render".into())),
        },
        use_on_render: true,
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Send 3 frames WITHOUT the render attribute — should be encoded, NOT rendered.
    for i in 0..3u64 {
        let frame = make_frame("src", i);
        let buf = make_buffer(&gen, i);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Send 2 frames WITH the render attribute — should be both rendered AND encoded.
    for i in 3..5u64 {
        let frame = make_frame_with_attr("src", i, "gate", "render");
        let buf = make_buffer(&gen, i);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Give the pipeline time to process.
    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    let rendered = render_count.load(Ordering::SeqCst);

    assert_eq!(
        encoded, 5,
        "All 5 frames should be encoded (render gate does not block encoding), got {encoded}"
    );
    assert_eq!(
        rendered, 2,
        "Only 2 frames with the render attribute should trigger on_render, got {rendered}"
    );
}

// -----------------------------------------------------------------------
// Test: both gates active simultaneously
// -----------------------------------------------------------------------

#[test]
fn both_gates_active() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let render_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedCounter(enc_count.clone()))),
        on_render: Some(Arc::new(RenderCounter(render_count.clone()))),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        conditional: ConditionalSpec {
            encode_attribute: Some(("gate".into(), "process".into())),
            render_attribute: Some(("gate".into(), "render".into())),
        },
        use_on_render: true,
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Frame 0: no attributes → dropped (encode gate blocks)
    {
        let frame = make_frame("src", 0);
        let buf = make_buffer(&gen, 0);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Frame 1: only encode attribute → encoded but NOT rendered
    {
        let frame = make_frame_with_attr("src", 1, "gate", "process");
        let buf = make_buffer(&gen, 1);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Frame 2: only render attribute (no encode attr) → dropped by encode gate
    {
        let frame = make_frame("src", 2);
        let mut fm = frame.clone();
        fm.set_persistent_attribute("gate", "render", &None, false, vec![]);
        let buf = make_buffer(&gen, 2);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Frame 3: both attributes → rendered AND encoded
    {
        let frame = make_frame("src", 3);
        let mut fm = frame.clone();
        fm.set_persistent_attribute("gate", "process", &None, false, vec![]);
        fm.set_persistent_attribute("gate", "render", &None, false, vec![]);
        let buf = make_buffer(&gen, 3);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    // Frame 4: both attributes → rendered AND encoded
    {
        let frame = make_frame("src", 4);
        let mut fm = frame.clone();
        fm.set_persistent_attribute("gate", "process", &None, false, vec![]);
        fm.set_persistent_attribute("gate", "render", &None, false, vec![]);
        let buf = make_buffer(&gen, 4);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    let rendered = render_count.load(Ordering::SeqCst);

    // Frames 0 and 2 are dropped by encode gate (3 pass: 1, 3, 4)
    assert_eq!(
        encoded, 3,
        "3 frames should pass the encode gate, got {encoded}"
    );
    // Of those 3, only frames 3 and 4 have the render attribute
    assert_eq!(
        rendered, 2,
        "2 frames should trigger rendering, got {rendered}"
    );
}

// -----------------------------------------------------------------------
// Test: no gates (default) — all frames pass through
// -----------------------------------------------------------------------

#[test]
fn no_gates_all_frames_pass() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let render_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EncodedCounter(enc_count.clone()))),
        on_render: Some(Arc::new(RenderCounter(render_count.clone()))),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        use_on_render: true,
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..4u64 {
        let frame = make_frame("src", i);
        let buf = make_buffer(&gen, i);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    let rendered = render_count.load(Ordering::SeqCst);

    assert_eq!(
        encoded, 4,
        "All 4 frames should be encoded with no gates, got {encoded}"
    );
    assert_eq!(
        rendered, 4,
        "All 4 frames should be rendered with no gates, got {rendered}"
    );
}
