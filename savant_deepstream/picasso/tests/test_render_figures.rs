//! CPU-rendered PNG test images for visual verification of every render figure
//! combination.
//!
//! Each test creates a CPU Skia surface, draws an object overlay using the
//! render modules, and saves the result as a PNG under the build target
//! directory (`target/…/rendered/`), keeping the source tree clean.
//!
//! Run with:
//! ```sh
//! cargo test -p picasso --test test_render_figures -- --nocapture
//! ```

use picasso::skia::common::ResolvedBBox;
use picasso::skia::context::DrawContext;
use picasso::skia::{bbox, blur, dot, label, object};
use savant_core::draw::*;
use savant_core::label_template::ParsedLabelFormats;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::path::PathBuf;
use std::sync::OnceLock;

const W: i32 = 480;
const H: i32 = 360;

fn output_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("rendered");
        std::fs::create_dir_all(&dir).expect("create rendered dir");
        eprintln!("render output → {}", dir.display());
        dir
    })
}

fn save_surface(surface: &mut skia_safe::Surface, name: &str) {
    let image = surface.image_snapshot();
    let data = image
        .encode(None, skia_safe::EncodedImageFormat::PNG, None)
        .expect("PNG encode failed");
    let path = output_dir().join(format!("{name}.png"));
    std::fs::write(&path, data.as_bytes()).expect("write PNG failed");
    eprintln!("wrote {}", path.display());
}

fn make_surface() -> skia_safe::Surface {
    skia_safe::surfaces::raster_n32_premul((W, H)).expect("failed to create raster surface")
}

fn draw_background(canvas: &skia_safe::Canvas) {
    canvas.clear(skia_safe::Color::from_argb(255, 30, 30, 40));
    let mut grid = skia_safe::Paint::default();
    grid.set_color(skia_safe::Color::from_argb(30, 255, 255, 255));
    grid.set_stroke_width(0.5);
    grid.set_style(skia_safe::PaintStyle::Stroke);
    for x in (0..W).step_by(40) {
        canvas.draw_line((x as f32, 0.0), (x as f32, H as f32), &grid);
    }
    for y in (0..H).step_by(40) {
        canvas.draw_line((0.0, y as f32), (W as f32, y as f32), &grid);
    }
}

fn make_ctx() -> DrawContext {
    DrawContext::new("sans-serif")
}

#[allow(clippy::too_many_arguments)]
fn make_frame_with_object(
    ns: &str,
    lbl: &str,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle: Option<f32>,
    track_box: Option<RBBox>,
    confidence: Option<f32>,
    track_id: Option<i64>,
) -> (VideoFrameProxy, i64) {
    let frame = VideoFrameProxy::new(
        "test",
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

    let mut builder = VideoObjectBuilder::default();
    builder
        .id(0)
        .namespace(ns.to_string())
        .label(lbl.to_string())
        .detection_box(RBBox::new(cx, cy, w, h, angle));
    if let Some(tb) = track_box {
        builder.track_box(Some(tb));
    }
    if let Some(c) = confidence {
        builder.confidence(Some(c));
    }
    if let Some(tid) = track_id {
        builder.track_id(Some(tid));
    }

    let obj = builder.build().unwrap();
    let borrowed = frame
        .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap();
    let obj_id = borrowed.get_id();
    (frame, obj_id)
}

fn resolved(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> ResolvedBBox {
    ResolvedBBox {
        cx,
        cy,
        w,
        h,
        angle,
    }
}

fn color(r: i64, g: i64, b: i64, a: i64) -> ColorDraw {
    ColorDraw::new(r, g, b, a).unwrap()
}

// -----------------------------------------------------------------------
// Bounding box tests
// -----------------------------------------------------------------------

#[test]
fn bbox_border_only() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 200.0, 120.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(255, 80, 80, 255),
        color(0, 0, 0, 0),
        3,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    save_surface(&mut surface, "bbox_border_only");
}

#[test]
fn bbox_border_and_fill() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 200.0, 120.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(80, 200, 255, 255),
        color(80, 200, 255, 60),
        2,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    save_surface(&mut surface, "bbox_border_and_fill");
}

#[test]
fn bbox_with_padding() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 160.0, 100.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(50, 255, 120, 255),
        color(50, 255, 120, 40),
        2,
        PaddingDraw::new(10, 10, 10, 10).unwrap(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);

    // Also draw the original bbox for comparison (thin white)
    let bb_orig = BoundingBoxDraw::new(
        color(255, 255, 255, 80),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb_orig, &mut ctx);
    save_surface(&mut surface, "bbox_with_padding");
}

#[test]
fn bbox_rotated() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 180.0, 100.0, 30.0);
    let bb = BoundingBoxDraw::new(
        color(255, 180, 40, 255),
        color(255, 180, 40, 50),
        2,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    save_surface(&mut surface, "bbox_rotated");
}

#[test]
fn bbox_rotated_with_label() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 200.0, 100.0, 35.0);

    let bb = BoundingBoxDraw::new(
        color(255, 180, 40, 255),
        color(255, 180, 40, 40),
        2,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);

    // Draw the wrapping AABB as a thin dashed guide so the label
    // alignment against it is visually verifiable.
    let aabb = r.wrapping_aabb();
    let guide_rect = skia_safe::Rect::from_xywh(
        aabb.cx - aabb.w / 2.0,
        aabb.cy - aabb.h / 2.0,
        aabb.w,
        aabb.h,
    );
    let mut guide_paint = skia_safe::Paint::default();
    guide_paint.set_anti_alias(true);
    guide_paint.set_style(skia_safe::PaintStyle::Stroke);
    guide_paint.set_color(skia_safe::Color::from_argb(100, 255, 255, 255));
    guide_paint.set_stroke_width(1.0);
    guide_paint.set_path_effect(skia_safe::PathEffect::dash(&[4.0, 4.0], 0.0));
    surface.canvas().draw_rect(guide_rect, &guide_paint);

    let (frame, obj_id) = make_frame_with_object(
        "detector",
        "car",
        r.cx,
        r.cy,
        r.w,
        r.h,
        Some(r.angle),
        None,
        Some(0.87),
        None,
    );
    let objects = frame.get_all_objects();
    let obj = objects.iter().find(|o| o.get_id() == obj_id).unwrap();

    let ld = make_label_draw(
        LabelPositionKind::TopLeftOutside,
        0,
        -2,
        vec!["{label}".into(), "angle={det_angle}".into()],
    );
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    label::draw_label(surface.canvas(), &r, obj, &ld, Some(&tmpl), &mut ctx);
    save_surface(&mut surface, "bbox_rotated_with_label");
}

// -----------------------------------------------------------------------
// Label tests
// -----------------------------------------------------------------------

fn make_label_draw(
    pos_kind: LabelPositionKind,
    mx: i64,
    my: i64,
    format: Vec<String>,
) -> LabelDraw {
    LabelDraw::new(
        color(0, 0, 0, 255),
        color(80, 200, 255, 200),
        color(40, 100, 128, 255),
        1.2,
        1,
        LabelPosition::new(pos_kind, mx, my).unwrap(),
        PaddingDraw::new(4, 2, 4, 2).unwrap(),
        format,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn draw_label_test(
    name: &str,
    label_draw: &LabelDraw,
    templates: Option<&ParsedLabelFormats>,
    r: &ResolvedBBox,
    obj_ns: &str,
    obj_label: &str,
    confidence: Option<f32>,
    track_id: Option<i64>,
) {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();

    // Draw the bbox outline so position is visible
    let bb = BoundingBoxDraw::new(
        color(255, 255, 255, 120),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), r, &bb, &mut ctx);

    let (frame, obj_id) = make_frame_with_object(
        obj_ns,
        obj_label,
        r.cx,
        r.cy,
        r.w,
        r.h,
        if r.angle.abs() > f32::EPSILON {
            Some(r.angle)
        } else {
            None
        },
        None,
        confidence,
        track_id,
    );
    let objects = frame.get_all_objects();
    let obj = objects.iter().find(|o| o.get_id() == obj_id).unwrap();

    label::draw_label(surface.canvas(), r, obj, label_draw, templates, &mut ctx);
    save_surface(&mut surface, name);
}

#[test]
fn label_single_line_top_left_outside() {
    let r = resolved(240.0, 200.0, 200.0, 120.0, 0.0);
    let ld = make_label_draw(
        LabelPositionKind::TopLeftOutside,
        0,
        -2,
        vec!["{label}".into()],
    );
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_single_top_left_outside",
        &ld,
        Some(&tmpl),
        &r,
        "det",
        "car",
        None,
        None,
    );
}

#[test]
fn label_single_line_top_left_inside() {
    let r = resolved(240.0, 200.0, 200.0, 120.0, 0.0);
    let ld = make_label_draw(
        LabelPositionKind::TopLeftInside,
        4,
        4,
        vec!["{label}".into()],
    );
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_single_top_left_inside",
        &ld,
        Some(&tmpl),
        &r,
        "det",
        "car",
        None,
        None,
    );
}

#[test]
fn label_single_line_center() {
    let r = resolved(240.0, 180.0, 200.0, 120.0, 0.0);
    let ld = make_label_draw(LabelPositionKind::Center, 0, 0, vec!["{label}".into()]);
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_single_center",
        &ld,
        Some(&tmpl),
        &r,
        "det",
        "person",
        None,
        None,
    );
}

#[test]
fn label_multiline_top_left_outside() {
    let r = resolved(240.0, 220.0, 200.0, 120.0, 0.0);
    let formats = vec![
        "{namespace}/{label}".into(),
        "id={id}".into(),
        "conf={confidence}".into(),
    ];
    let ld = make_label_draw(LabelPositionKind::TopLeftOutside, 0, -2, formats);
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_multiline_top_left_outside",
        &ld,
        Some(&tmpl),
        &r,
        "detector",
        "truck",
        Some(0.87),
        None,
    );
}

#[test]
fn label_with_template_vars() {
    let r = resolved(240.0, 200.0, 200.0, 120.0, 0.0);
    let formats = vec![
        "{draw_label} #{id}".into(),
        "track={track_id}".into(),
        "det=({det_xc},{det_yc})".into(),
    ];
    let ld = make_label_draw(LabelPositionKind::TopLeftOutside, 0, -2, formats);
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_template_vars",
        &ld,
        Some(&tmpl),
        &r,
        "detector",
        "car",
        Some(0.95),
        Some(42),
    );
}

#[test]
fn label_with_margin_offsets() {
    let r = resolved(240.0, 180.0, 200.0, 120.0, 0.0);
    let ld = make_label_draw(
        LabelPositionKind::TopLeftOutside,
        20,
        -15,
        vec!["{label}".into()],
    );
    let tmpl = ParsedLabelFormats::parse(&ld.format).unwrap();
    draw_label_test(
        "label_margin_offsets",
        &ld,
        Some(&tmpl),
        &r,
        "det",
        "bus",
        None,
        None,
    );
}

// -----------------------------------------------------------------------
// Dot tests
// -----------------------------------------------------------------------

#[test]
fn dot_small() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 160.0, 100.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(255, 255, 255, 80),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    let d = DotDraw::new(color(255, 80, 80, 255), 4).unwrap();
    dot::draw_dot(surface.canvas(), &r, &d, &mut ctx);
    save_surface(&mut surface, "dot_small");
}

#[test]
fn dot_large() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 160.0, 100.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(255, 255, 255, 80),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    let d = DotDraw::new(color(50, 255, 120, 255), 12).unwrap();
    dot::draw_dot(surface.canvas(), &r, &d, &mut ctx);
    save_surface(&mut surface, "dot_large");
}

#[test]
fn dot_semi_transparent() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();
    let r = resolved(240.0, 180.0, 160.0, 100.0, 0.0);
    let bb = BoundingBoxDraw::new(
        color(255, 255, 255, 120),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &r, &bb, &mut ctx);
    let d = DotDraw::new(color(255, 80, 80, 100), 10).unwrap();
    dot::draw_dot(surface.canvas(), &r, &d, &mut ctx);
    save_surface(&mut surface, "dot_semi_transparent");
}

// -----------------------------------------------------------------------
// Blur test
// -----------------------------------------------------------------------

#[test]
fn blur_region() {
    let mut surface = make_surface();
    // Draw some content first so the blur is visible.
    let canvas = surface.canvas();
    draw_background(canvas);
    let mut text_paint = skia_safe::Paint::default();
    text_paint.set_color(skia_safe::Color::WHITE);
    text_paint.set_anti_alias(true);
    let fm = skia_safe::FontMgr::default();
    let tf = fm
        .match_family_style("sans-serif", skia_safe::FontStyle::bold())
        .unwrap();
    let font = skia_safe::Font::from_typeface(tf, 28.0);
    for y in (40..H).step_by(50) {
        canvas.draw_str(
            "BLUR TEST 1234567890 ABCDEFGH",
            (20.0, y as f32),
            &font,
            &text_paint,
        );
    }

    let r = resolved(240.0, 180.0, 200.0, 120.0, 0.0);
    blur::draw_blur(surface.canvas(), &r);
    save_surface(&mut surface, "blur_region");
}

// -----------------------------------------------------------------------
// Combined tests
// -----------------------------------------------------------------------

#[test]
fn combined_full_spec() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();

    let (frame, obj_id) = make_frame_with_object(
        "detector",
        "person",
        240.0,
        200.0,
        180.0,
        130.0,
        None,
        None,
        Some(0.92),
        Some(7),
    );
    let objects = frame.get_all_objects();
    let obj = objects.iter().find(|o| o.get_id() == obj_id).unwrap();

    let formats = vec![
        "{namespace}/{label} #{id}".into(),
        "conf={confidence} trk={track_id}".into(),
    ];
    let tmpl = ParsedLabelFormats::parse(&formats).unwrap();

    let draw_spec = ObjectDraw::with_bbox_source(
        Some(
            BoundingBoxDraw::new(
                color(255, 80, 80, 255),
                color(255, 80, 80, 40),
                2,
                PaddingDraw::new(6, 6, 6, 6).unwrap(),
            )
            .unwrap(),
        ),
        Some(DotDraw::new(color(255, 255, 255, 255), 5).unwrap()),
        Some(
            LabelDraw::new(
                color(0, 0, 0, 255),
                color(255, 80, 80, 200),
                color(200, 40, 40, 255),
                1.2,
                1,
                LabelPosition::new(LabelPositionKind::TopLeftOutside, 0, -2).unwrap(),
                PaddingDraw::new(4, 2, 4, 2).unwrap(),
                formats,
            )
            .unwrap(),
        ),
        false,
        BBoxSource::DetectionBox,
    );

    object::draw_object(surface.canvas(), obj, &draw_spec, Some(&tmpl), &mut ctx);
    save_surface(&mut surface, "combined_full_spec");
}

#[test]
fn combined_full_spec_with_blur() {
    let mut surface = make_surface();
    // Draw text background so blur is visible
    let canvas = surface.canvas();
    draw_background(canvas);
    let mut text_paint = skia_safe::Paint::default();
    text_paint.set_color(skia_safe::Color::from_argb(180, 200, 200, 200));
    text_paint.set_anti_alias(true);
    let fm = skia_safe::FontMgr::default();
    let tf = fm
        .match_family_style("sans-serif", skia_safe::FontStyle::normal())
        .unwrap();
    let font = skia_safe::Font::from_typeface(tf, 14.0);
    for y in (20..H).step_by(20) {
        canvas.draw_str(
            "The quick brown fox jumps over the lazy dog 0123456789",
            (10.0, y as f32),
            &font,
            &text_paint,
        );
    }

    let mut ctx = make_ctx();
    let (frame, obj_id) = make_frame_with_object(
        "detector",
        "face",
        240.0,
        180.0,
        120.0,
        100.0,
        None,
        None,
        Some(0.99),
        None,
    );
    let objects = frame.get_all_objects();
    let obj = objects.iter().find(|o| o.get_id() == obj_id).unwrap();

    let formats = vec!["{label} #{id}".into()];
    let tmpl = ParsedLabelFormats::parse(&formats).unwrap();

    let draw_spec = ObjectDraw::with_bbox_source(
        Some(
            BoundingBoxDraw::new(
                color(255, 200, 50, 255),
                color(0, 0, 0, 0),
                2,
                PaddingDraw::default_padding(),
            )
            .unwrap(),
        ),
        Some(DotDraw::new(color(255, 200, 50, 255), 4).unwrap()),
        Some(
            LabelDraw::new(
                color(0, 0, 0, 255),
                color(255, 200, 50, 200),
                color(0, 0, 0, 0),
                1.0,
                1,
                LabelPosition::new(LabelPositionKind::TopLeftOutside, 0, -2).unwrap(),
                PaddingDraw::new(3, 1, 3, 1).unwrap(),
                formats,
            )
            .unwrap(),
        ),
        true,
        BBoxSource::DetectionBox,
    );

    object::draw_object(surface.canvas(), obj, &draw_spec, Some(&tmpl), &mut ctx);
    save_surface(&mut surface, "combined_full_spec_with_blur");
}

#[test]
fn combined_tracking_box_source() {
    let mut surface = make_surface();
    draw_background(surface.canvas());
    let mut ctx = make_ctx();

    let track_box = RBBox::new(280.0, 160.0, 160.0, 110.0, None);
    let (frame, obj_id) = make_frame_with_object(
        "tracker",
        "car",
        200.0,
        200.0,
        140.0,
        90.0,
        None,
        Some(track_box),
        Some(0.85),
        Some(12),
    );
    let objects = frame.get_all_objects();
    let obj = objects.iter().find(|o| o.get_id() == obj_id).unwrap();

    // Draw detection bbox in white dashed style for reference
    let det_r = resolved(200.0, 200.0, 140.0, 90.0, 0.0);
    let det_bb = BoundingBoxDraw::new(
        color(255, 255, 255, 80),
        color(0, 0, 0, 0),
        1,
        PaddingDraw::default_padding(),
    )
    .unwrap();
    bbox::draw_bounding_box(surface.canvas(), &det_r, &det_bb, &mut ctx);

    let formats = vec![
        "{label} #{track_id}".into(),
        "det=({det_xc},{det_yc})".into(),
        "trk=({track_xc},{track_yc})".into(),
    ];
    let tmpl = ParsedLabelFormats::parse(&formats).unwrap();

    let draw_spec = ObjectDraw::with_bbox_source(
        Some(
            BoundingBoxDraw::new(
                color(80, 200, 255, 255),
                color(80, 200, 255, 30),
                2,
                PaddingDraw::default_padding(),
            )
            .unwrap(),
        ),
        Some(DotDraw::new(color(80, 200, 255, 255), 4).unwrap()),
        Some(
            LabelDraw::new(
                color(255, 255, 255, 255),
                color(80, 200, 255, 200),
                color(0, 0, 0, 0),
                1.0,
                1,
                LabelPosition::new(LabelPositionKind::TopLeftOutside, 0, -2).unwrap(),
                PaddingDraw::new(4, 2, 4, 2).unwrap(),
                formats,
            )
            .unwrap(),
        ),
        false,
        BBoxSource::TrackingBox,
    );

    object::draw_object(surface.canvas(), obj, &draw_spec, Some(&tmpl), &mut ctx);
    save_surface(&mut surface, "combined_tracking_box_source");
}
