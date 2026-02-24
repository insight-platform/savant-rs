//! Central dot rendering.

use crate::draw_context::DrawContext;
use crate::render::common::{color_to_skia, ResolvedBBox};
use savant_core::draw::DotDraw;

/// Draw a filled circle at the bbox center.
pub fn draw_dot(
    canvas: &skia_safe::Canvas,
    resolved: &ResolvedBBox,
    dot: &DotDraw,
    ctx: &mut DrawContext,
) {
    if dot.color.alpha > 0 && dot.radius > 0 {
        ctx.fill_paint.set_color(color_to_skia(&dot.color));
        canvas.draw_circle(
            (resolved.cx, resolved.cy),
            dot.radius as f32,
            &ctx.fill_paint,
        );
    }
}
