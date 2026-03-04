//! E2E test: Font family change via spec update.
//!
//! Validates that changing font_family on SourceSpec update triggers
//! DrawContext rebuild and rendering continues correctly.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::draw::{BoundingBoxDraw, ColorDraw, ObjectDraw, PaddingDraw};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

fn build_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(0, 255, 0, 255).unwrap();
    let bg = ColorDraw::new(0, 255, 0, 40).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();
    spec.insert("det", "car", ObjectDraw::new(Some(bb), None, None, false));
    spec
}

#[test]
#[serial_test::serial]
fn e2e_font_family_hot_swap() {
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
        },
        callbacks,
    );

    let draw = build_draw_spec();
    let spec1 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw: draw.clone(),
        font_family: "sans-serif".to_string(),
        ..Default::default()
    };
    engine.set_source_spec("font", spec1).unwrap();

    let gen = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..5u64 {
        let mut frame = make_frame_sized("font", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        add_object(&frame, 100.0, 100.0, 50.0, 30.0);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("font", frame, buf, None).unwrap();
    }
    std::thread::sleep(Duration::from_secs(1));

    let spec2 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw,
        font_family: "monospace".to_string(),
        ..Default::default()
    };
    engine.set_source_spec("font", spec2).unwrap();

    for i in 5..10u64 {
        let mut frame = make_frame_sized("font", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        add_object(&frame, 200.0, 200.0, 60.0, 40.0);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("font", frame, buf, None).unwrap();
    }
    engine.send_eos("font").unwrap();
    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    assert!(enc_count.load(Ordering::SeqCst) >= 8);
}
