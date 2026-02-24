//! Object-level draw orchestrator.
//!
//! Resolves the bounding box source, then delegates to the individual figure
//! renderers in the correct order: blur (background), bbox, dot, label
//! (foreground).

use crate::draw_context::DrawContext;
use crate::render::{bbox, blur, common, dot, label};
use savant_core::draw::ObjectDraw;
use savant_core::label_template::ParsedLabelFormats;
use savant_core::primitives::object::BorrowedVideoObject;

/// Draw all enabled overlays for a single object.
///
/// `templates` should be the pre-parsed label formats for this object's
/// draw spec (if available).  When `None`, the raw `LabelDraw.format`
/// strings are rendered literally (no variable interpolation).
pub fn draw_object(
    canvas: &skia_safe::Canvas,
    obj: &BorrowedVideoObject,
    draw: &ObjectDraw,
    templates: Option<&ParsedLabelFormats>,
    ctx: &mut DrawContext,
) {
    let resolved = common::resolve_bbox(obj, draw.bbox_source);

    // Blur goes first so it affects the underlying image, not our overlays.
    if draw.blur {
        blur::draw_blur(canvas, &resolved);
    }

    if let Some(ref bb) = draw.bounding_box {
        bbox::draw_bounding_box(canvas, &resolved, bb, ctx);
    }

    if let Some(ref dot_draw) = draw.central_dot {
        dot::draw_dot(canvas, &resolved, dot_draw, ctx);
    }

    if let Some(ref label_draw) = draw.label {
        label::draw_label(canvas, &resolved, obj, label_draw, templates, ctx);
    }
}
