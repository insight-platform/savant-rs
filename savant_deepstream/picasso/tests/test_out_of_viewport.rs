//! End-to-end test: objects with bounding boxes outside (or partially
//! outside) the viewport must not crash the rendering pipeline.
//!
//! Exercises the full `Encode` + Skia render path with extreme bbox
//! coordinates: fully off-screen in every direction, partially visible,
//! oversized (bigger than the canvas), zero-size, and rotated boxes
//! straddling the edges.  Picasso's internal renderer (bbox, dot, label,
//! blur) is active for all objects through [`ObjectDrawSpec`].
//!
//! Success criterion: all frames produce encoded output and no panic or
//! crash occurs.

mod common;

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::draw::{
    BoundingBoxDraw, ColorDraw, DotDraw, LabelDraw, LabelPosition, ObjectDraw, PaddingDraw,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
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

/// Full draw spec with bbox + dot + label + blur for maximum coverage.
fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(255, 0, 0, 255).unwrap();
    let bg = ColorDraw::new(255, 0, 0, 50).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::new(4, 4, 4, 4).unwrap()).unwrap();

    let dot_c = ColorDraw::new(0, 255, 0, 200).unwrap();
    let dot = DotDraw::new(dot_c, 6).unwrap();

    let fc = ColorDraw::new(255, 255, 255, 255).unwrap();
    let lbg = ColorDraw::new(0, 0, 0, 180).unwrap();
    let lbc = ColorDraw::new(255, 0, 0, 255).unwrap();
    let label = LabelDraw::new(
        fc,
        lbg,
        lbc,
        1.4,
        1,
        LabelPosition::default_position().unwrap(),
        PaddingDraw::new(4, 2, 4, 2).unwrap(),
        vec![
            "{label} #{id}".to_string(),
            "({det_xc:.0}, {det_yc:.0})".to_string(),
        ],
    )
    .unwrap();

    spec.insert(
        "det",
        "oob",
        ObjectDraw::new(Some(bb), Some(dot), Some(label), true),
    );
    spec
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
        _renderer: &mut deepstream_nvbufsurface::SkiaRenderer,
        _frame: &VideoFrameProxy,
    ) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Each entry is `(cx, cy, w, h, angle)`.
/// Covers every category of out-of-viewport placement.
const OOB_BOXES: &[(f32, f32, f32, f32, Option<f32>)] = &[
    // --- fully off-screen ---
    (-200.0, 240.0, 100.0, 80.0, None), // left
    (840.0, 240.0, 100.0, 80.0, None),  // right
    (320.0, -200.0, 100.0, 80.0, None), // above
    (320.0, 680.0, 100.0, 80.0, None),  // below
    (-500.0, -500.0, 60.0, 60.0, None), // top-left corner, far away
    (1200.0, 900.0, 60.0, 60.0, None),  // bottom-right corner, far away
    // --- partially visible (straddling edges) ---
    (0.0, 240.0, 100.0, 80.0, None),   // left edge
    (640.0, 240.0, 100.0, 80.0, None), // right edge
    (320.0, 0.0, 100.0, 80.0, None),   // top edge
    (320.0, 480.0, 100.0, 80.0, None), // bottom edge
    (0.0, 0.0, 100.0, 80.0, None),     // top-left corner
    (640.0, 480.0, 100.0, 80.0, None), // bottom-right corner
    // --- oversized (bigger than canvas) ---
    (320.0, 240.0, 2000.0, 1500.0, None),   // centered, huge
    (-100.0, -100.0, 3000.0, 3000.0, None), // offset huge
    // --- degenerate sizes ---
    (320.0, 240.0, 0.1, 0.1, None), // near-zero
    (320.0, 240.0, 1.0, 1.0, None), // tiny
    // --- rotated at boundary ---
    (0.0, 0.0, 200.0, 100.0, Some(45.0)), // rotated on top-left corner
    (640.0, 0.0, 200.0, 100.0, Some(-30.0)), // rotated on top-right
    (320.0, 480.0, 300.0, 150.0, Some(90.0)), // rotated on bottom edge
    (-50.0, 240.0, 200.0, 100.0, Some(60.0)), // rotated off left edge
    // --- extreme coordinates ---
    (1e6, 1e6, 100.0, 80.0, None),   // astronomically far
    (-1e6, -1e6, 100.0, 80.0, None), // negative extreme
    (320.0, 240.0, 1e5, 1e5, None),  // extreme dimensions
    // --- rotated + oversized + off-screen ---
    (-300.0, -300.0, 1500.0, 1500.0, Some(135.0)),
];

/// Sends one frame per out-of-viewport box through the full pipeline.
/// Each frame carries a single object so every placement is individually
/// exercised.
#[test]
fn out_of_viewport_objects_do_not_crash() {
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
    let mut engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        draw: build_draw_spec(),
        font_family: "sans-serif".to_string(),
        use_on_render: true,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("oob", spec).unwrap();

    let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, W, H, 1)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(32)
        .build()
        .unwrap();

    let num_frames = OOB_BOXES.len();

    for (i, &(cx, cy, w, h, angle)) in OOB_BOXES.iter().enumerate() {
        let frame = make_frame("oob", i as u64);
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("det".to_string())
            .label("oob".to_string())
            .detection_box(RBBox::new(cx, cy, w, h, angle))
            .confidence(Some(0.88))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
        let buf = common::make_gpu_surface_view_uniform(&gen, i as u64, FRAME_DUR_NS);
        engine.send_frame("oob", frame, buf, None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(3));
    engine.send_eos("oob").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    let rendered = render_count.load(Ordering::SeqCst);

    assert_eq!(
        encoded, num_frames,
        "all {num_frames} frames must be encoded (got {encoded})"
    );
    assert_eq!(
        rendered, num_frames,
        "all {num_frames} frames must trigger on_render (got {rendered})"
    );
}

/// Sends a single frame with ALL out-of-viewport boxes at once to stress
/// multi-object rendering in a single draw pass.
#[test]
fn all_oob_objects_on_single_frame() {
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
    let mut engine = PicassoEngine::new(general, callbacks);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(make_encoder_config()),
        },
        draw: build_draw_spec(),
        font_family: "sans-serif".to_string(),
        use_on_render: true,
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("oob-all", spec).unwrap();

    let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, W, H, 1)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(32)
        .build()
        .unwrap();

    let frame = make_frame("oob-all", 0);
    for &(cx, cy, w, h, angle) in OOB_BOXES {
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("det".to_string())
            .label("oob".to_string())
            .detection_box(RBBox::new(cx, cy, w, h, angle))
            .confidence(Some(0.77))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
    let buf = common::make_gpu_surface_view_uniform(&gen, 0, FRAME_DUR_NS);
    engine
        .send_frame("oob-all", frame.clone(), buf, None)
        .unwrap();

    std::thread::sleep(Duration::from_secs(3));
    engine.send_eos("oob-all").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    let rendered = render_count.load(Ordering::SeqCst);

    assert_eq!(encoded, 1, "single frame must be encoded (got {encoded})");
    assert_eq!(
        rendered, 1,
        "single frame must trigger on_render (got {rendered})"
    );

    let objects = frame.get_all_objects();
    assert_eq!(
        objects.len(),
        OOB_BOXES.len(),
        "all objects must survive on the frame"
    );
}
