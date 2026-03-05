//! E2E test: Many objects per frame (200 detections).
//!
//! Simulates a crowded scene to verify pipeline stability and rendering
//! correctness under heavy object load.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::draw::{BoundingBoxDraw, ColorDraw, ObjectDraw, PaddingDraw};
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;
const NUM_OBJECTS: usize = 200;

fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(0, 255, 0, 255).unwrap();
    let bg = ColorDraw::new(0, 255, 0, 40).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 1, PaddingDraw::default_padding()).unwrap();
    spec.insert("det", "obj", ObjectDraw::new(Some(bb), None, None, false));
    spec
}

#[test]
#[serial_test::serial]
fn e2e_high_object_count() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
        })),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw: build_draw_spec(),
        ..Default::default()
    };
    engine.set_source_spec("many", spec).unwrap();

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let mut frame = make_frame_sized("many", W as i64, H as i64);
    frame.set_pts(0).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();

    let cols = 20;
    let rows = 10;
    for i in 0..NUM_OBJECTS {
        let col = (i % cols) as f32;
        let row = (i / cols) as f32;
        let cx = 30.0 + col * (W as f32 / (cols as f32 + 1.0));
        let cy = 30.0 + row * (H as f32 / (rows as f32 + 1.0));
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("det".to_string())
            .label("obj".to_string())
            .detection_box(RBBox::new(cx, cy, 15.0, 15.0, None))
            .build()
            .unwrap();
        frame
            .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();
    }

    assert_eq!(
        frame.get_all_objects().len(),
        NUM_OBJECTS,
        "input frame should have 200 objects"
    );

    let start = Instant::now();
    let buf = make_gpu_surface_view(&gen, 0, DUR);
    engine.send_frame("many", frame, buf, None).unwrap();
    engine.send_eos("many").unwrap();

    std::thread::sleep(Duration::from_secs(5));
    engine.shutdown();
    let elapsed = start.elapsed();

    assert!(enc_count.load(Ordering::SeqCst) >= 1);
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "processing should complete in reasonable time, took {:?}",
        elapsed
    );
}
