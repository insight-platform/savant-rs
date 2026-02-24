//! Bounding box rendering with full `BoundingBoxDraw` spec support:
//! border color, background fill, thickness, padding, and rotation.

use crate::draw_context::DrawContext;
use crate::render::common::{color_to_skia, padded_rect, ResolvedBBox};
use savant_core::draw::BoundingBoxDraw;

/// Draw a bounding box on the canvas.
///
/// The rectangle is expanded outward by `bb.padding` before drawing.
/// If the resolved bbox has a non-zero angle the canvas is rotated
/// around the center.
pub fn draw_bounding_box(
    canvas: &skia_safe::Canvas,
    resolved: &ResolvedBBox,
    bb: &BoundingBoxDraw,
    ctx: &mut DrawContext,
) {
    let hw = resolved.w / 2.0;
    let hh = resolved.h / 2.0;

    canvas.save();
    canvas.translate((resolved.cx, resolved.cy));
    if resolved.angle.abs() > f32::EPSILON {
        canvas.rotate(resolved.angle, None);
    }

    let rect = padded_rect(hw, hh, &bb.padding);

    if bb.background_color.alpha > 0 {
        ctx.fill_paint
            .set_color(color_to_skia(&bb.background_color));
        canvas.draw_rect(rect, &ctx.fill_paint);
    }

    if bb.border_color.alpha > 0 && bb.thickness > 0 {
        ctx.stroke_paint.set_stroke_width(bb.thickness as f32);
        ctx.stroke_paint.set_color(color_to_skia(&bb.border_color));
        canvas.draw_rect(rect, &ctx.stroke_paint);
    }

    canvas.restore();
}
