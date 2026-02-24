//! Multi-line label rendering with full `LabelDraw` spec support:
//! position (TopLeftOutside / TopLeftInside / Center), margin, padding,
//! background, border, font color/scale, and template interpolation.

use crate::draw_context::DrawContext;
use crate::render::common::{color_to_skia, ResolvedBBox};
use savant_core::draw::{LabelDraw, LabelPositionKind};
use savant_core::label_template::ParsedLabelFormats;
use savant_core::primitives::object::{BorrowedVideoObject, ObjectOperations};

/// Draw a multi-line label for an object.
///
/// Each entry in `templates` produces one text line.  Lines are stacked
/// vertically inside a padded background rectangle whose position is
/// determined by `label_draw.position`.
pub fn draw_label(
    canvas: &skia_safe::Canvas,
    resolved: &ResolvedBBox,
    obj: &BorrowedVideoObject,
    label_draw: &LabelDraw,
    templates: Option<&ParsedLabelFormats>,
    ctx: &mut DrawContext,
) {
    let lines: Vec<String> = if let Some(tmpl) = templates {
        tmpl.expand_lines(obj)
    } else if label_draw.format.is_empty() {
        vec![obj.get_label()]
    } else {
        label_draw.format.clone()
    };

    if lines.is_empty() {
        return;
    }

    let font_size = (label_draw.font_scale * 12.0) as f32;
    let font = skia_safe::Font::from_typeface(&ctx.default_typeface, font_size);

    let line_height = font_size * 1.25;
    let text_block_h = line_height * lines.len() as f32;

    let max_text_w = lines
        .iter()
        .map(|l| font.measure_str(l, None).0)
        .fold(0.0f32, f32::max);

    let pad = &label_draw.padding;
    let pad_l = pad.left as f32;
    let pad_t = pad.top as f32;
    let pad_r = pad.right as f32;
    let pad_b = pad.bottom as f32;
    let box_w = max_text_w + pad_l + pad_r;
    let box_h = text_block_h + pad_t + pad_b;

    // Label is always axis-aligned; position it against the wrapping AABB
    // so it stays horizontal even when the object box is rotated.
    let aabb = resolved.wrapping_aabb();
    let hw = aabb.w / 2.0;
    let hh = aabb.h / 2.0;
    let mx = label_draw.position.margin_x as f32;
    let my = label_draw.position.margin_y as f32;

    let (box_x, box_y) = match label_draw.position.position {
        LabelPositionKind::TopLeftOutside => {
            let x = aabb.cx - hw + mx;
            let y = aabb.cy - hh + my - box_h;
            (x, y)
        }
        LabelPositionKind::TopLeftInside => {
            let x = aabb.cx - hw + mx;
            let y = aabb.cy - hh + my;
            (x, y)
        }
        LabelPositionKind::Center => {
            let x = aabb.cx - box_w / 2.0 + mx;
            let y = aabb.cy - box_h / 2.0 + my;
            (x, y)
        }
    };

    let bg_rect = skia_safe::Rect::from_xywh(box_x, box_y, box_w, box_h);

    if label_draw.background_color.alpha > 0 {
        ctx.fill_paint
            .set_color(color_to_skia(&label_draw.background_color));
        canvas.draw_rect(bg_rect, &ctx.fill_paint);
    }

    if label_draw.border_color.alpha > 0 && label_draw.thickness > 0 {
        ctx.stroke_paint
            .set_color(color_to_skia(&label_draw.border_color));
        ctx.stroke_paint
            .set_stroke_width(label_draw.thickness as f32);
        canvas.draw_rect(bg_rect, &ctx.stroke_paint);
    }

    ctx.fill_paint
        .set_color(color_to_skia(&label_draw.font_color));

    let text_x = box_x + pad_l;
    let mut text_y = box_y + pad_t + font_size;

    for line in &lines {
        canvas.draw_str(line, (text_x, text_y), &font, &ctx.fill_paint);
        text_y += line_height;
    }
}
