//! E2E test: Production-realistic conditional encode+render.
//!
//! Simulates "record on alert": frames only encoded when attribute
//! present, rendering only when scene has objects.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use savant_core::primitives::attribute_value::AttributeValue;
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
use savant_core::primitives::WithAttributes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

struct RenderCounter(pub Arc<AtomicUsize>);
impl OnRender for RenderCounter {
    fn call(
        &self,
        _: &str,
        _: &mut deepstream_nvbufsurface::SkiaRenderer,
        _: &savant_core::primitives::frame::VideoFrameProxy,
    ) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
#[serial_test::serial]
fn e2e_conditional_selective_recording() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let render_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
        })),
        on_render: Some(Arc::new(RenderCounter(render_count.clone()))),
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
        conditional: ConditionalSpec {
            encode_attribute: Some(("alert".to_string(), "detected".to_string())),
            render_attribute: Some(("scene".to_string(), "has_objects".to_string())),
        },
        use_on_render: true,
        ..Default::default()
    };
    engine.set_source_spec("cond", spec).unwrap();

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

    let make_frame_with_attrs =
        |encode_attr: bool, render_attr: bool, has_objects: bool, idx: u64| {
            let mut frame = make_frame_sized("cond", W as i64, H as i64);
            frame.set_pts((idx * DUR) as i64).unwrap();
            frame.set_duration(Some(DUR as i64)).unwrap();
            if encode_attr {
                frame.set_persistent_attribute(
                    "alert",
                    "detected",
                    &None,
                    false,
                    vec![AttributeValue::boolean(true, None)],
                );
            }
            if render_attr {
                frame.set_persistent_attribute(
                    "scene",
                    "has_objects",
                    &None,
                    false,
                    vec![AttributeValue::boolean(true, None)],
                );
            }
            if has_objects {
                let obj = VideoObjectBuilder::default()
                    .id(0)
                    .namespace("det".to_string())
                    .label("car".to_string())
                    .detection_box(RBBox::new(100.0, 100.0, 50.0, 30.0, None))
                    .build()
                    .unwrap();
                frame
                    .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
                    .unwrap();
            }
            frame
        };

    let buf_a = make_gpu_surface_view(&gen, 0, DUR);
    engine
        .send_frame(
            "cond",
            make_frame_with_attrs(true, true, true, 0),
            buf_a,
            None,
        )
        .unwrap();

    let buf_b = make_gpu_surface_view(&gen, 1, DUR);
    engine
        .send_frame(
            "cond",
            make_frame_with_attrs(true, false, true, 1),
            buf_b,
            None,
        )
        .unwrap();

    let buf_c = make_gpu_surface_view(&gen, 2, DUR);
    engine
        .send_frame(
            "cond",
            make_frame_with_attrs(false, true, true, 2),
            buf_c,
            None,
        )
        .unwrap();

    let buf_d = make_gpu_surface_view(&gen, 3, DUR);
    engine
        .send_frame(
            "cond",
            make_frame_with_attrs(false, false, false, 3),
            buf_d,
            None,
        )
        .unwrap();

    engine.send_eos("cond").unwrap();
    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    assert_eq!(enc_count.load(Ordering::SeqCst), 2, "Frame A and B encoded");
    assert_eq!(
        render_count.load(Ordering::SeqCst),
        1,
        "Only Frame A rendered"
    );
}
