//! E2E test: Dynamic per-object draw override via OnObjectDrawSpec.
//!
//! Simulates an AI pipeline that dynamically highlights "alert" objects
//! by overriding the static draw spec at runtime.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::draw::{BoundingBoxDraw, ColorDraw, ObjectDraw, PaddingDraw};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

/// OnObjectDrawSpec callback that overrides "alert" with red bbox, uses static for "car", None for "ignored".
struct ObjectDrawSpecOverride {
    call_count: Arc<AtomicUsize>,
}

impl OnObjectDrawSpec for ObjectDrawSpecOverride {
    fn call(
        &self,
        _source_id: &str,
        object: &savant_core::primitives::object::BorrowedVideoObject,
        _current_spec: Option<&ObjectDraw>,
    ) -> Option<ObjectDraw> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let label = object.get_label();
        match label.as_str() {
            "alert" => {
                let border = ColorDraw::new(255, 0, 0, 255).unwrap();
                let bg = ColorDraw::new(255, 0, 0, 60).unwrap();
                let bb =
                    BoundingBoxDraw::new(border, bg, 5, PaddingDraw::default_padding()).unwrap();
                Some(ObjectDraw::new(Some(bb), None, None, false))
            }
            "car" => None,     // use static spec
            "ignored" => None, // no static spec, won't be drawn
            _ => None,
        }
    }
}

fn build_static_draw_spec() -> ObjectDrawSpec {
    let mut spec = ObjectDrawSpec::new();
    let border = ColorDraw::new(0, 255, 0, 255).unwrap();
    let bg = ColorDraw::new(0, 255, 0, 40).unwrap();
    let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();
    spec.insert("det", "car", ObjectDraw::new(Some(bb), None, None, false));
    spec
}

#[test]
#[serial_test::serial]
fn e2e_on_object_draw_spec() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let callback_count = Arc::new(AtomicUsize::new(0));
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_object_draw_spec: Some(Arc::new(ObjectDrawSpecOverride {
            call_count: callback_count.clone(),
        })),
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

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw: build_static_draw_spec(),
        ..Default::default()
    };
    engine.set_source_spec("obj-draw", spec).unwrap();

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

    let mut frame = make_frame_sized("obj-draw", W as i64, H as i64);
    frame.set_pts(0).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();

    // Add 3 objects: alert, car, ignored
    let obj_alert = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("alert".to_string())
        .detection_box(RBBox::new(100.0, 100.0, 50.0, 50.0, None))
        .build()
        .unwrap();
    frame
        .add_object(obj_alert, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();

    let obj_car = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("car".to_string())
        .detection_box(RBBox::new(200.0, 200.0, 80.0, 60.0, None))
        .build()
        .unwrap();
    frame
        .add_object(obj_car, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();

    let obj_ignored = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("ignored".to_string())
        .detection_box(RBBox::new(300.0, 300.0, 40.0, 40.0, None))
        .build()
        .unwrap();
    frame
        .add_object(obj_ignored, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();

    let buf = make_gpu_buffer(&gen, 0, DUR);
    engine.send_frame("obj-draw", frame, buf, None).unwrap();
    engine.send_eos("obj-draw").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    assert_eq!(
        callback_count.load(Ordering::SeqCst),
        3,
        "OnObjectDrawSpec should fire once per object"
    );
    assert!(
        enc_count.load(Ordering::SeqCst) >= 1,
        "encoded frame should be produced"
    );
}
