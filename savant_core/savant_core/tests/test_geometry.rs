use savant_core::primitives::bbox::RBBox;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core::primitives::object::{
    BorrowedVideoObject, IdCollisionResolutionPolicy, ObjectOperations,
};

fn make_frame(w: i64, h: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        "test",
        (30, 1),
        w,
        h,
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

fn add_object(frame: &VideoFrameProxy, cx: f32, cy: f32, w: f32, h: f32) -> i64 {
    add_object_angled(frame, cx, cy, w, h, None)
}

fn add_object_angled(
    frame: &VideoFrameProxy,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle: Option<f32>,
) -> i64 {
    use savant_core::primitives::object::VideoObjectBuilder;
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("car".to_string())
        .detection_box(RBBox::new(cx, cy, w, h, angle))
        .build()
        .unwrap();
    let borrowed = frame
        .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();
    borrowed.get_id()
}

fn get_object(frame: &VideoFrameProxy, id: i64) -> BorrowedVideoObject {
    frame
        .get_all_objects()
        .into_iter()
        .find(|o| o.get_id() == id)
        .unwrap()
}

#[test]
fn transform_backward_letterbox() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object(&frame, 100.0, 100.0, 50.0, 50.0);

    frame_mut.transform_backward().unwrap();

    let chain = frame.get_transformations();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0], VideoFrameTransformation::InitialSize(1920, 1080));
    assert_eq!(frame.get_width(), 1920);
    assert_eq!(frame.get_height(), 1080);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let sx = 640.0 / 1920.0;
    let sy = 480.0 / 1080.0;
    let expected_x = (100.0 - 10.0) / sx;
    let expected_y = (100.0 - 10.0) / sy;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn transform_forward_letterbox() {
    // Object is in 1920×1080 (InitialSize) space, chain has LetterBox(660,500,10,10,10,10).
    // Forward affine: scale by (640/1920, 480/1080) then shift by (10, 10).
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object(&frame, 330.0, 250.0, 100.0, 100.0);

    frame_mut.transform_forward().unwrap();

    let chain = frame.get_transformations();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0], VideoFrameTransformation::InitialSize(660, 500));
    assert_eq!(frame.get_width(), 660);
    assert_eq!(frame.get_height(), 500);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let sx = 640.0 / 1920.0;
    let sy = 480.0 / 1080.0;
    let expected_x = 330.0 * sx + 10.0;
    let expected_y = 250.0 * sy + 10.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn identity_chain_transform_backward_is_noop() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();

    let obj_id = add_object(&frame, 500.0, 300.0, 100.0, 80.0);

    frame_mut.transform_backward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    assert!((det.get_xc() - 500.0).abs() < 0.01);
    assert!((det.get_yc() - 300.0).abs() < 0.01);
    assert!((det.get_width() - 100.0).abs() < 0.01);
    assert!((det.get_height() - 80.0).abs() < 0.01);
    assert_eq!(frame.get_width(), 1920);
    assert_eq!(frame.get_height(), 1080);
}

#[test]
fn identity_chain_transform_forward_is_noop() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();

    let obj_id = add_object(&frame, 500.0, 300.0, 100.0, 80.0);

    frame_mut.transform_forward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    assert!((det.get_xc() - 500.0).abs() < 0.01);
    assert!((det.get_yc() - 300.0).abs() < 0.01);
    assert!((det.get_width() - 100.0).abs() < 0.01);
    assert!((det.get_height() - 80.0).abs() < 0.01);
    assert_eq!(frame.get_width(), 1920);
    assert_eq!(frame.get_height(), 1080);
}

#[test]
fn transform_backward_complex_chain() {
    let frame = make_frame(2000, 2000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(1000, 1000, 0, 0, 0, 0));
    frame_mut.add_transformation(VideoFrameTransformation::Padding(100, 100, 100, 100));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(600, 600, 0, 0, 0, 0));

    let obj_id = add_object(&frame, 300.0, 300.0, 60.0, 60.0);

    frame_mut.transform_backward().unwrap();

    assert_eq!(frame.get_width(), 2000);
    assert_eq!(frame.get_height(), 2000);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let expected_x = 4.0 * 300.0 - 200.0;
    let expected_y = 4.0 * 300.0 - 200.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 1.0,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 1.0,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn transform_backward_crop_then_letterbox() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::Crop(160, 40, 160, 40));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(800, 500, 0, 0, 0, 0));

    let obj_id = add_object(&frame, 400.0, 250.0, 100.0, 100.0);

    frame_mut.transform_backward().unwrap();

    assert_eq!(frame.get_width(), 1920);
    assert_eq!(frame.get_height(), 1080);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let expected_x = 400.0 * 2.0 + 160.0;
    let expected_y = 250.0 * 2.0 + 40.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 1.0,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 1.0,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn transform_forward_crop_then_letterbox() {
    // Object in 1920×1080 space, chain: Crop(160,40,160,40) → LetterBox(800,500,0,0,0,0).
    // After crop: 1600×1000, scale 800/1600=0.5, 500/1000=0.5.
    // Forward affine: sx=0.5, sy=0.5, tx=-160*0.5=-80, ty=-40*0.5=-20.
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::Crop(160, 40, 160, 40));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(800, 500, 0, 0, 0, 0));

    let obj_id = add_object(&frame, 400.0, 250.0, 100.0, 100.0);

    frame_mut.transform_forward().unwrap();

    assert_eq!(frame.get_width(), 800);
    assert_eq!(frame.get_height(), 500);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    // forward: x' = 400*0.5 - 80 = 120, y' = 250*0.5 - 20 = 105
    let expected_x = 400.0 * 0.5 - 80.0;
    let expected_y = 250.0 * 0.5 - 20.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 1.0,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 1.0,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn transform_backward_fails_without_initial_size() {
    let frame = make_frame(640, 480);
    let mut frame_mut = frame.clone();
    frame_mut.clear_transformations();
    frame_mut.add_transformation(VideoFrameTransformation::Padding(10, 10, 10, 10));

    let result = frame_mut.transform_backward();
    assert!(result.is_err());
}

#[test]
fn transform_forward_fails_without_initial_size() {
    let frame = make_frame(640, 480);
    let mut frame_mut = frame.clone();
    frame_mut.clear_transformations();
    frame_mut.add_transformation(VideoFrameTransformation::Padding(10, 10, 10, 10));

    let result = frame_mut.transform_forward();
    assert!(result.is_err());
}

#[test]
fn transform_backward_asymmetric_letterbox() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(640, 600, 0, 60, 0, 60));

    let obj_id = add_object(&frame, 100.0, 100.0, 20.0, 20.0);

    frame_mut.transform_backward().unwrap();

    assert_eq!(frame.get_width(), 1920);
    assert_eq!(frame.get_height(), 1080);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let expected_x = 100.0 * 3.0;
    let expected_y = 100.0 * 2.25 - 135.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn transform_forward_asymmetric_letterbox() {
    // Symmetric counterpart: objects in 1920×1080 go forward through
    // LetterBox(640,600, 0,60,0,60) → inner 640×480, sx=640/1920=1/3,
    // sy=480/1080=4/9, tx=0, ty=60.
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(640, 600, 0, 60, 0, 60));

    let obj_id = add_object(&frame, 300.0, 540.0, 60.0, 60.0);

    frame_mut.transform_forward().unwrap();

    assert_eq!(frame.get_width(), 640);
    assert_eq!(frame.get_height(), 600);

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let sx = 640.0 / 1920.0;
    let sy = 480.0 / 1080.0;
    let expected_x = 300.0 * sx;
    let expected_y = 540.0 * sy + 60.0;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
}

#[test]
fn roundtrip_target_then_initial() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object(&frame, 960.0, 540.0, 200.0, 100.0);

    frame_mut.transform_forward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    let sx = 640.0 / 1920.0;
    let sy = 480.0 / 1080.0;
    let mid_x = 960.0 * sx + 10.0;
    let mid_y = 540.0 * sy + 10.0;
    assert!((det.get_xc() - mid_x).abs() < 0.5);
    assert!((det.get_yc() - mid_y).abs() < 0.5);

    // Rebuild the ORIGINAL chain to go backward.
    // After transform_forward, chain = [InitialSize(660,500)].
    // Replace with original chain so transform_backward can invert it.
    frame_mut.clear_transformations();
    frame_mut.add_transformation(VideoFrameTransformation::InitialSize(1920, 1080));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));
    frame_mut.transform_backward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    assert!(
        (det.get_xc() - 960.0).abs() < 1.0,
        "roundtrip xc={} expected=960",
        det.get_xc()
    );
    assert!(
        (det.get_yc() - 540.0).abs() < 1.0,
        "roundtrip yc={} expected=540",
        det.get_yc()
    );
}

// ── Angled bounding box tests ──────────────────────────────────────────

#[test]
fn forward_angle_37_uniform_scale() {
    // Uniform LetterBox (1000×1000 → 500×500): sx=sy=0.5.
    // Uniform scale preserves angle.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 600.0, 400.0, 120.0, 60.0, Some(37.0));

    frame_mut.transform_forward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!((det.get_xc() - 300.0).abs() < 0.5);
    assert!((det.get_yc() - 200.0).abs() < 0.5);
    assert!((det.get_width() - 60.0).abs() < 0.5);
    assert!((det.get_height() - 30.0).abs() < 0.5);
    assert!(
        (det.get_angle().unwrap() - 37.0).abs() < 0.01,
        "uniform scale must preserve angle, got {}",
        det.get_angle().unwrap()
    );
}

#[test]
fn backward_angle_37_uniform_scale() {
    // Inverse of the above: object in LetterBox space at (300,200),
    // transform_backward brings it back to 1000×1000.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 300.0, 200.0, 60.0, 30.0, Some(37.0));

    frame_mut.transform_backward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!((det.get_xc() - 600.0).abs() < 0.5);
    assert!((det.get_yc() - 400.0).abs() < 0.5);
    assert!((det.get_width() - 120.0).abs() < 0.5);
    assert!((det.get_height() - 60.0).abs() < 0.5);
    assert!(
        (det.get_angle().unwrap() - 37.0).abs() < 0.01,
        "uniform scale must preserve angle, got {}",
        det.get_angle().unwrap()
    );
}

#[test]
fn forward_angle_180_uniform_scale() {
    // Uniform scale preserves angle: sx=sy=0.5, angle stays 180°.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 600.0, 400.0, 80.0, 40.0, Some(180.0));

    frame_mut.transform_forward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!((det.get_xc() - 300.0).abs() < 0.5);
    assert!((det.get_yc() - 200.0).abs() < 0.5);
    assert!((det.get_width() - 40.0).abs() < 0.5);
    assert!((det.get_height() - 20.0).abs() < 0.5);
    assert!(
        (det.get_angle().unwrap() - 180.0).abs() < 0.01,
        "uniform scale must preserve angle, got {}",
        det.get_angle().unwrap()
    );
}

#[test]
fn forward_angle_90_uniform_scale_with_padding() {
    // Uniform LetterBox (500×500 + 10px padding all sides):
    // inner 480×480, sx=sy=480/1000=0.48, tx=ty=10.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        500, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object_angled(&frame, 400.0, 300.0, 100.0, 50.0, Some(90.0));

    frame_mut.transform_forward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    let s = 480.0 / 1000.0;
    assert!((det.get_xc() - (400.0 * s + 10.0)).abs() < 0.5);
    assert!((det.get_yc() - (300.0 * s + 10.0)).abs() < 0.5);
    assert!((det.get_width() - 100.0 * s).abs() < 0.5);
    assert!((det.get_height() - 50.0 * s).abs() < 0.5);
    assert!(
        (det.get_angle().unwrap() - 90.0).abs() < 0.01,
        "uniform scale must preserve angle, got {}",
        det.get_angle().unwrap()
    );
}

#[test]
fn forward_angle_45_non_uniform_scale() {
    // Non-uniform LetterBox: 1920×1080 → LetterBox(640,480,0,0,0,0).
    // sx=640/1920=1/3, sy=480/1080=4/9.  These differ, so the 45° angle
    // will change.  Center always follows the simple affine rule.
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(640, 480, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 960.0, 540.0, 100.0, 100.0, Some(45.0));

    frame_mut.transform_forward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    let sx = 640.0 / 1920.0_f32;
    let sy = 480.0 / 1080.0_f32;
    assert!(
        (det.get_xc() - 960.0 * sx).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        960.0 * sx
    );
    assert!(
        (det.get_yc() - 540.0 * sy).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        540.0 * sy
    );
    let angle = det.get_angle().unwrap();
    assert!(
        (angle - 45.0).abs() > 0.1,
        "non-uniform scale must change the 45° angle, got {}",
        angle
    );
    assert!(
        angle > 0.0 && angle < 90.0,
        "angle must stay in (0,90), got {}",
        angle
    );
}

#[test]
fn backward_angle_45_non_uniform_scale_center() {
    // Non-uniform backward: center must follow inverse affine exactly,
    // regardless of the angle.
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object_angled(&frame, 100.0, 100.0, 40.0, 20.0, Some(45.0));

    frame_mut.transform_backward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    let sx = 640.0 / 1920.0_f32;
    let sy = 480.0 / 1080.0_f32;
    let expected_x = (100.0 - 10.0) / sx;
    let expected_y = (100.0 - 10.0) / sy;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.5,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.5,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );
    assert!(det.get_angle().is_some(), "angle must remain set");
}

#[test]
fn roundtrip_forward_backward_uniform_preserves_angle() {
    // Uniform scale (sx == sy) is perfectly reversible for any angle.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 500.0, 500.0, 200.0, 100.0, Some(45.0));

    frame_mut.transform_forward().unwrap();

    let det_mid = get_object(&frame, obj_id).get_detection_box();
    assert!(
        (det_mid.get_xc() - 250.0).abs() < 0.5,
        "mid xc={} expected=250",
        det_mid.get_xc()
    );
    assert!(
        (det_mid.get_angle().unwrap() - 45.0).abs() < 0.01,
        "uniform scale preserves angle, got {}",
        det_mid.get_angle().unwrap()
    );

    frame_mut.clear_transformations();
    frame_mut.add_transformation(VideoFrameTransformation::InitialSize(1000, 1000));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));
    frame_mut.transform_backward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!(
        (det.get_xc() - 500.0).abs() < 1.0,
        "roundtrip xc={} expected=500",
        det.get_xc()
    );
    assert!(
        (det.get_yc() - 500.0).abs() < 1.0,
        "roundtrip yc={} expected=500",
        det.get_yc()
    );
    assert!(
        (det.get_width() - 200.0).abs() < 1.0,
        "roundtrip w={} expected=200",
        det.get_width()
    );
    assert!(
        (det.get_height() - 100.0).abs() < 1.0,
        "roundtrip h={} expected=100",
        det.get_height()
    );
    assert!(
        (det.get_angle().unwrap() - 45.0).abs() < 0.01,
        "roundtrip angle={} expected=45",
        det.get_angle().unwrap()
    );
}

#[test]
fn roundtrip_forward_backward_non_uniform_center_preserved() {
    // Non-uniform scale changes width/height/angle through trig approximation
    // in RBBox::scale, so w/h are NOT perfectly recoverable. But the center
    // always follows the linear affine and roundtrips exactly.
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));

    let obj_id = add_object_angled(&frame, 800.0, 400.0, 120.0, 60.0, Some(37.0));

    frame_mut.transform_forward().unwrap();

    let det_mid = get_object(&frame, obj_id).get_detection_box();
    let mid_angle = det_mid.get_angle().unwrap();
    assert!(
        (mid_angle - 37.0).abs() > 0.1,
        "non-uniform scale should change the angle, got {}",
        mid_angle
    );

    frame_mut.clear_transformations();
    frame_mut.add_transformation(VideoFrameTransformation::InitialSize(1920, 1080));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));
    frame_mut.transform_backward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!(
        (det.get_xc() - 800.0).abs() < 1.0,
        "roundtrip xc={} expected=800",
        det.get_xc()
    );
    assert!(
        (det.get_yc() - 400.0).abs() < 1.0,
        "roundtrip yc={} expected=400",
        det.get_yc()
    );
}

#[test]
fn forward_angle_0_is_same_as_no_angle() {
    // Explicit angle=0° should behave identically to None (no angle).
    let frame_a = make_frame(1920, 1080);
    let mut frame_a_mut = frame_a.clone();
    frame_a_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));
    let id_a = add_object_angled(&frame_a, 960.0, 540.0, 200.0, 100.0, Some(0.0));

    let frame_b = make_frame(1920, 1080);
    let mut frame_b_mut = frame_b.clone();
    frame_b_mut.add_transformation(VideoFrameTransformation::LetterBox(
        660, 500, 10, 10, 10, 10,
    ));
    let id_b = add_object(&frame_b, 960.0, 540.0, 200.0, 100.0);

    frame_a_mut.transform_forward().unwrap();
    frame_b_mut.transform_forward().unwrap();

    let det_a = get_object(&frame_a, id_a).get_detection_box();
    let det_b = get_object(&frame_b, id_b).get_detection_box();

    assert!((det_a.get_xc() - det_b.get_xc()).abs() < 0.01);
    assert!((det_a.get_yc() - det_b.get_yc()).abs() < 0.01);
    assert!((det_a.get_width() - det_b.get_width()).abs() < 0.01);
    assert!((det_a.get_height() - det_b.get_height()).abs() < 0.01);
}

#[test]
fn forward_angle_30_uniform_scale() {
    // Uniform scale preserves angle: center scales, w/h scale, angle stays 30°.
    let frame = make_frame(1000, 1000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0));

    let obj_id = add_object_angled(&frame, 500.0, 500.0, 100.0, 60.0, Some(30.0));

    frame_mut.transform_forward().unwrap();

    let det = get_object(&frame, obj_id).get_detection_box();
    assert!((det.get_xc() - 250.0).abs() < 0.5);
    assert!((det.get_yc() - 250.0).abs() < 0.5);
    assert!((det.get_width() - 50.0).abs() < 0.5);
    assert!((det.get_height() - 30.0).abs() < 0.5);
    assert!(
        (det.get_angle().unwrap() - 30.0).abs() < 0.01,
        "uniform scale must preserve angle, got {}",
        det.get_angle().unwrap()
    );
}
