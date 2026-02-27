//! Region blur rendering.
//!
//! Applies a Gaussian blur to the rectangular area covered by the object's
//! bounding box.  Uses a Skia `save_layer` with a `blur` image filter so
//! that only the clipped region is affected.

use crate::skia::common::ResolvedBBox;

/// Draw a blurred region covering the object's bounding box.
///
/// The blur sigma is derived from the bbox size: `min(w, h) * 0.1`, clamped
/// to `[2.0, 40.0]` for visual sanity.
pub fn draw_blur(canvas: &skia_safe::Canvas, resolved: &ResolvedBBox) {
    let sigma = (resolved.w.min(resolved.h) * 0.1).clamp(2.0, 40.0);

    let hw = resolved.w / 2.0;
    let hh = resolved.h / 2.0;

    canvas.save();
    canvas.translate((resolved.cx, resolved.cy));
    if resolved.angle.abs() > f32::EPSILON {
        canvas.rotate(resolved.angle, None);
    }

    let clip_rect = skia_safe::Rect::from_xywh(-hw, -hh, resolved.w, resolved.h);
    canvas.clip_rect(clip_rect, skia_safe::ClipOp::Intersect, true);

    if let Some(filter) =
        skia_safe::image_filters::blur((sigma, sigma), skia_safe::TileMode::Clamp, None, None)
    {
        let layer_paint = skia_safe::Paint::default();
        let layer_rec = skia_safe::canvas::SaveLayerRec::default()
            .bounds(&clip_rect)
            .backdrop(&filter)
            .paint(&layer_paint);
        canvas.save_layer(&layer_rec);
        canvas.restore();
    }

    canvas.restore();
}
