use deepstream_buffers::{Padding, Rect, TransformConfig};
use picasso::rewrite_frame_transformations;
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
        "30/1",
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1000000000),
        0,
        None,
        None,
    )
    .unwrap()
}

fn add_object(frame: &VideoFrameProxy, cx: f32, cy: f32, w: f32, h: f32) -> i64 {
    use savant_core::primitives::object::VideoObjectBuilder;
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace("det".to_string())
        .label("car".to_string())
        .detection_box(RBBox::new(cx, cy, w, h, None))
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

fn assert_bbox_approx(
    frame: &VideoFrameProxy,
    id: i64,
    expected_cx: f32,
    expected_cy: f32,
    expected_w: f32,
    expected_h: f32,
    tol: f32,
) {
    let obj = get_object(frame, id);
    let det = obj.get_detection_box();
    assert!(
        (det.get_xc() - expected_cx).abs() < tol,
        "xc={} expected={}",
        det.get_xc(),
        expected_cx
    );
    assert!(
        (det.get_yc() - expected_cy).abs() < tol,
        "yc={} expected={}",
        det.get_yc(),
        expected_cy
    );
    assert!(
        (det.get_width() - expected_w).abs() < tol,
        "w={} expected={}",
        det.get_width(),
        expected_w
    );
    assert!(
        (det.get_height() - expected_h).abs() < tol,
        "h={} expected={}",
        det.get_height(),
        expected_h
    );
}

/// Delegate to the production `rewrite_frame_transformations` helper.
fn encode_transform(
    frame: &VideoFrameProxy,
    target_w: u32,
    target_h: u32,
    config: &TransformConfig,
    src_rect: Option<&Rect>,
) {
    rewrite_frame_transformations(frame, target_w, target_h, config, src_rect).unwrap();
}

fn config_with_padding(padding: Padding) -> TransformConfig {
    TransformConfig {
        padding,
        ..Default::default()
    }
}

fn config_with_crop(_rect: Rect, padding: Padding) -> TransformConfig {
    TransformConfig {
        padding,
        ..Default::default()
    }
}

// -----------------------------------------------------------------------
// Bypass tests: transform_backward reverts to initial coordinates
// -----------------------------------------------------------------------

#[test]
fn bypass_transforms_bbox_back_to_initial() {
    let frame = make_frame(1920, 1080);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(640, 480, 0, 0, 0, 0));
    frame_mut.add_transformation(VideoFrameTransformation::Padding(10, 10, 10, 10));

    let obj_id = add_object(&frame, 100.0, 100.0, 50.0, 50.0);

    let mut frame_mut = frame.clone();
    frame_mut.transform_backward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();

    let sx = 640.0 / 1920.0;
    let sy = 480.0 / 1080.0;
    let expected_x = (100.0 - 10.0) / sx;
    let expected_y = (100.0 - 10.0) / sy;
    assert!(
        (det.get_xc() - expected_x).abs() < 0.1,
        "xc={} expected={}",
        det.get_xc(),
        expected_x
    );
    assert!(
        (det.get_yc() - expected_y).abs() < 0.1,
        "yc={} expected={}",
        det.get_yc(),
        expected_y
    );

    let expected_w = 50.0 / sx;
    let expected_h = 50.0 / sy;
    assert!(
        (det.get_width() - expected_w).abs() < 0.1,
        "w={} expected={}",
        det.get_width(),
        expected_w
    );
    assert!(
        (det.get_height() - expected_h).abs() < 0.1,
        "h={} expected={}",
        det.get_height(),
        expected_h
    );

    let chain = frame.get_transformations();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0], VideoFrameTransformation::InitialSize(1920, 1080));
}

#[test]
fn identity_chain_preserves_coordinates() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 500.0, 300.0, 100.0, 80.0);

    let mut frame_mut = frame.clone();
    frame_mut.transform_backward().unwrap();

    let obj = get_object(&frame, obj_id);
    let det = obj.get_detection_box();
    assert!((det.get_xc() - 500.0).abs() < 0.01);
    assert!((det.get_yc() - 300.0).abs() < 0.01);
    assert!((det.get_width() - 100.0).abs() < 0.01);
    assert!((det.get_height() - 80.0).abs() < 0.01);
}

#[test]
fn multiple_letterbox_and_paddings() {
    let frame = make_frame(2000, 2000);
    let mut frame_mut = frame.clone();
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(1000, 1000, 0, 0, 0, 0));
    frame_mut.add_transformation(VideoFrameTransformation::Padding(100, 100, 100, 100));
    frame_mut.add_transformation(VideoFrameTransformation::LetterBox(600, 600, 0, 0, 0, 0));

    let obj_id = add_object(&frame, 300.0, 300.0, 60.0, 60.0);

    let mut frame_mut = frame.clone();
    frame_mut.transform_backward().unwrap();

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

// -----------------------------------------------------------------------
// Encode: erase chain → add GPU ops → transform_forward
// -----------------------------------------------------------------------

#[test]
fn symmetric_800x600_to_800x800_center_object() {
    let frame = make_frame(800, 600);
    let obj_id = add_object(&frame, 400.0, 300.0, 80.0, 60.0);
    encode_transform(
        &frame,
        800,
        800,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 400.0, 400.0, 80.0, 60.0, 0.5);
}

#[test]
fn symmetric_800x600_to_800x800_top_edge_object() {
    let frame = make_frame(800, 600);
    let obj_id = add_object(&frame, 400.0, 0.0, 80.0, 60.0);
    encode_transform(
        &frame,
        800,
        800,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 400.0, 100.0, 80.0, 60.0, 0.5);
}

#[test]
fn symmetric_800x600_to_800x800_bottom_edge_object() {
    let frame = make_frame(800, 600);
    let obj_id = add_object(&frame, 400.0, 600.0, 80.0, 60.0);
    encode_transform(
        &frame,
        800,
        800,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 400.0, 700.0, 80.0, 60.0, 0.5);
}

#[test]
fn same_aspect_1280x720_to_1920x1080_center() {
    let frame = make_frame(1280, 720);
    let obj_id = add_object(&frame, 640.0, 360.0, 100.0, 100.0);
    encode_transform(
        &frame,
        1920,
        1080,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 960.0, 540.0, 150.0, 150.0, 0.5);
}

#[test]
fn same_aspect_1280x720_to_1920x1080_origin() {
    let frame = make_frame(1280, 720);
    let obj_id = add_object(&frame, 0.0, 0.0, 20.0, 20.0);
    encode_transform(
        &frame,
        1920,
        1080,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 0.0, 0.0, 30.0, 30.0, 0.5);
}

#[test]
fn same_aspect_1280x720_to_1920x1080_corner() {
    let frame = make_frame(1280, 720);
    let obj_id = add_object(&frame, 1280.0, 720.0, 20.0, 20.0);
    encode_transform(
        &frame,
        1920,
        1080,
        &config_with_padding(Padding::Symmetric),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 1920.0, 1080.0, 30.0, 30.0, 0.5);
}

#[test]
fn right_bottom_1920x1080_to_800x600_center() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 960.0, 540.0, 100.0, 100.0);
    encode_transform(
        &frame,
        800,
        600,
        &config_with_padding(Padding::RightBottom),
        None,
    );
    let scale = 800.0 / 1920.0;
    assert_bbox_approx(
        &frame,
        obj_id,
        960.0 * scale,
        540.0 * scale,
        100.0 * scale,
        100.0 * scale,
        0.5,
    );
}

#[test]
fn right_bottom_1920x1080_to_800x600_origin() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 0.0, 0.0, 40.0, 40.0);
    encode_transform(
        &frame,
        800,
        600,
        &config_with_padding(Padding::RightBottom),
        None,
    );
    let scale = 800.0 / 1920.0;
    assert_bbox_approx(&frame, obj_id, 0.0, 0.0, 40.0 * scale, 40.0 * scale, 0.5);
}

#[test]
fn right_bottom_1920x1080_to_800x600_bottom_right_corner() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 1920.0, 1080.0, 40.0, 40.0);
    encode_transform(
        &frame,
        800,
        600,
        &config_with_padding(Padding::RightBottom),
        None,
    );
    let scale = 800.0 / 1920.0;
    assert_bbox_approx(
        &frame,
        obj_id,
        1920.0 * scale,
        1080.0 * scale,
        40.0 * scale,
        40.0 * scale,
        0.5,
    );
}

#[test]
fn symmetric_pillarbox_720x1280_to_1920x1080() {
    let frame = make_frame(720, 1280);
    let obj_id = add_object(&frame, 360.0, 640.0, 100.0, 100.0);
    encode_transform(
        &frame,
        1920,
        1080,
        &config_with_padding(Padding::Symmetric),
        None,
    );

    let scale = 1080.0 / 1280.0;
    let scaled_w = (1080.0_f64 * 720.0 / 1280.0).round() as f32;
    let offset_x = (1920.0 - scaled_w) / 2.0;

    assert_bbox_approx(
        &frame,
        obj_id,
        360.0 * scale + offset_x,
        640.0 * scale,
        100.0 * scale,
        100.0 * scale,
        1.0,
    );
}

// -----------------------------------------------------------------------
// Bypass: reverts boxes from current (padded) space back to initial
// -----------------------------------------------------------------------

#[test]
fn bypass_padding_reverts_to_initial() {
    let frame = make_frame(1280, 720);
    let mut fm = frame.clone();
    fm.add_transformation(VideoFrameTransformation::Padding(500, 200, 0, 0));

    let obj_id = add_object(&frame, 750.0, 350.0, 100.0, 100.0);

    let mut fm2 = frame.clone();
    fm2.transform_backward().unwrap();

    assert_bbox_approx(&frame, obj_id, 250.0, 150.0, 100.0, 100.0, 0.5);
}

#[test]
fn bypass_letterbox_and_padding_reverts_to_initial() {
    let frame = make_frame(1920, 1080);
    let mut fm = frame.clone();
    fm.add_transformation(VideoFrameTransformation::LetterBox(960, 540, 0, 0, 0, 0));
    fm.add_transformation(VideoFrameTransformation::Padding(20, 10, 20, 10));

    let obj_id = add_object(&frame, 520.0, 280.0, 40.0, 40.0);

    let mut fm2 = frame.clone();
    fm2.transform_backward().unwrap();

    assert_bbox_approx(&frame, obj_id, 1000.0, 540.0, 80.0, 80.0, 0.5);
}

#[test]
fn bypass_asymmetric_padding_only() {
    let frame = make_frame(640, 480);
    let mut fm = frame.clone();
    fm.add_transformation(VideoFrameTransformation::Padding(100, 50, 60, 30));

    let obj_id = add_object(&frame, 200.0, 150.0, 50.0, 50.0);

    let mut fm2 = frame.clone();
    fm2.transform_backward().unwrap();

    assert_bbox_approx(&frame, obj_id, 100.0, 100.0, 50.0, 50.0, 0.5);
}

// -----------------------------------------------------------------------
// Crop support
// -----------------------------------------------------------------------

#[test]
fn crop_center_no_padding() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 960.0, 540.0, 100.0, 100.0);

    let rect = Rect {
        left: 480,
        top: 270,
        width: 960,
        height: 540,
    };
    let cfg = config_with_crop(rect, Padding::None);
    encode_transform(&frame, 960, 540, &cfg, Some(&rect));

    assert_bbox_approx(&frame, obj_id, 480.0, 270.0, 100.0, 100.0, 0.5);
}

#[test]
fn crop_top_left_no_padding() {
    let frame = make_frame(1000, 1000);
    let obj_id = add_object(&frame, 250.0, 250.0, 80.0, 80.0);

    let rect = Rect {
        left: 0,
        top: 0,
        width: 500,
        height: 500,
    };
    let cfg = config_with_crop(rect, Padding::None);
    encode_transform(&frame, 500, 500, &cfg, Some(&rect));

    assert_bbox_approx(&frame, obj_id, 250.0, 250.0, 80.0, 80.0, 0.5);
}

#[test]
fn crop_and_scale_up_no_padding() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 960.0, 540.0, 50.0, 50.0);

    let rect = Rect {
        left: 640,
        top: 360,
        width: 640,
        height: 360,
    };
    let cfg = config_with_crop(rect, Padding::None);
    encode_transform(&frame, 1280, 720, &cfg, Some(&rect));

    assert_bbox_approx(&frame, obj_id, 640.0, 360.0, 100.0, 100.0, 0.5);
}

#[test]
fn crop_and_symmetric_letterbox() {
    let frame = make_frame(1920, 1080);
    let obj_id = add_object(&frame, 960.0, 540.0, 96.0, 96.0);

    let rect = Rect {
        left: 480,
        top: 60,
        width: 960,
        height: 960,
    };
    let cfg = config_with_crop(rect, Padding::Symmetric);
    encode_transform(&frame, 800, 800, &cfg, Some(&rect));

    let scale = 800.0 / 960.0;
    assert_bbox_approx(
        &frame,
        obj_id,
        480.0 * scale,
        480.0 * scale,
        96.0 * scale,
        96.0 * scale,
        1.0,
    );
}

#[test]
fn no_crop_no_padding_is_plain_stretch() {
    let frame = make_frame(800, 600);
    let obj_id = add_object(&frame, 400.0, 300.0, 100.0, 100.0);
    encode_transform(
        &frame,
        1600,
        1200,
        &config_with_padding(Padding::None),
        None,
    );
    assert_bbox_approx(&frame, obj_id, 800.0, 600.0, 200.0, 200.0, 0.5);
}
