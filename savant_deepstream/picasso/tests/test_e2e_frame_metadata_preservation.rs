//! E2E test: Object/attribute survival through encode pipeline.
//!
//! Validates that objects, bboxes, and attributes on VideoFrameProxy
//! survive the full encode pipeline.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::primitives::attribute_value::AttributeValue;
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::rust::VideoFrameTranscodingMethod;
use savant_core::primitives::RBBox;
use savant_core::primitives::WithAttributes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

#[test]
#[serial_test::serial]
fn e2e_frame_metadata_preservation() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let captured: Arc<parking_lot::Mutex<Option<savant_core::primitives::frame::VideoFrameProxy>>> =
        Arc::new(parking_lot::Mutex::new(None));

    struct CaptureFrame {
        count: Arc<AtomicUsize>,
        captured: Arc<parking_lot::Mutex<Option<savant_core::primitives::frame::VideoFrameProxy>>>,
    }
    impl OnEncodedFrame for CaptureFrame {
        fn call(&self, output: picasso::message::EncodedOutput) {
            if let picasso::message::EncodedOutput::VideoFrame(frame) = output {
                self.count.fetch_add(1, Ordering::SeqCst);
                *self.captured.lock() = Some(frame);
            }
        }
    }

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CaptureFrame {
            count: enc_count.clone(),
            captured: captured.clone(),
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
        ..Default::default()
    };
    engine.set_source_spec("meta", spec).unwrap();

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

    let mut frame = make_frame_sized("meta", W as i64, H as i64);
    frame.set_pts(0).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();
    frame.set_persistent_attribute(
        "inference",
        "model_name",
        &None,
        false,
        vec![AttributeValue::string("yolo_v8", None)],
    );

    let obj_car = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("car".to_string())
        .detection_box(RBBox::new(100.0, 100.0, 50.0, 30.0, None))
        .attributes(vec![savant_core::primitives::Attribute::persistent(
            "tracker",
            "track_id",
            vec![AttributeValue::integer(42, None)],
            &None,
            false,
        )])
        .build()
        .unwrap();
    frame
        .add_object(obj_car, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();

    let obj_person = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("person".to_string())
        .detection_box(RBBox::new(200.0, 150.0, 40.0, 80.0, None))
        .build()
        .unwrap();
    frame
        .add_object(obj_person, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();

    let buf = make_gpu_surface_view(&gen, 0, DUR);
    let frame_uuid = frame.get_uuid();
    let frame_pts = frame.get_pts();
    engine.send_frame("meta", frame, buf, None).unwrap();
    engine.send_eos("meta").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    assert!(enc_count.load(Ordering::SeqCst) >= 1);

    let out = captured.lock().take().expect("expected captured frame");
    // out uuid == frame uuid
    assert_eq!(out.get_uuid(), frame_uuid);
    assert_eq!(out.get_pts(), frame_pts);
    let objs: Vec<_> = out.get_all_objects().into_iter().collect();
    assert_eq!(objs.len(), 2, "expected 2 objects on output");

    let has_car = objs
        .iter()
        .any(|o| o.get_namespace() == "det" && o.get_label() == "car");
    let has_person = objs
        .iter()
        .any(|o| o.get_namespace() == "det" && o.get_label() == "person");
    assert!(has_car);
    assert!(has_person);

    assert!(
        out.get_attribute("inference", "model_name").is_some(),
        "frame attribute should survive"
    );

    let car_obj = objs
        .iter()
        .find(|o| o.get_label() == "car")
        .expect("car object");
    assert!(
        car_obj.get_attribute("tracker", "track_id").is_some(),
        "object attribute should survive"
    );

    let transcoding_method = out.get_transcoding_method();
    assert!(
        matches!(transcoding_method, VideoFrameTranscodingMethod::Encoded),
        "transcoding method should be Encoded"
    );
    let content = out.get_content();
    assert!(
        matches!(*content, VideoFrameContent::Internal(_)),
        "output should have Internal content (encoded bitstream)"
    );
}
