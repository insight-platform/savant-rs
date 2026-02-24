//! Shared rendering helpers used by all figure modules.

use savant_core::draw::{BBoxSource, ColorDraw, PaddingDraw};
use savant_core::primitives::object::{BorrowedVideoObject, ObjectOperations};
use savant_core::primitives::RBBox;

/// Convert a [`ColorDraw`] to a Skia [`Color`](skia_safe::Color).
#[inline]
pub fn color_to_skia(c: &ColorDraw) -> skia_safe::Color {
    skia_safe::Color::from_argb(c.alpha as u8, c.red as u8, c.green as u8, c.blue as u8)
}

/// Resolved bounding box geometry ready for drawing.
pub struct ResolvedBBox {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

impl ResolvedBBox {
    /// Compute the axis-aligned bounding box that fully encloses this
    /// (possibly rotated) box.  When `angle ≈ 0` returns an identical copy.
    pub fn wrapping_aabb(&self) -> ResolvedBBox {
        if self.angle.abs() <= f32::EPSILON {
            return ResolvedBBox {
                cx: self.cx,
                cy: self.cy,
                w: self.w,
                h: self.h,
                angle: 0.0,
            };
        }
        let rad = self.angle.to_radians();
        let (sin, cos) = rad.sin_cos();
        let hw = self.w / 2.0;
        let hh = self.h / 2.0;
        let aabb_hw = (hw * cos.abs()) + (hh * sin.abs());
        let aabb_hh = (hw * sin.abs()) + (hh * cos.abs());
        ResolvedBBox {
            cx: self.cx,
            cy: self.cy,
            w: aabb_hw * 2.0,
            h: aabb_hh * 2.0,
            angle: 0.0,
        }
    }
}

/// Resolve the bounding box to use for rendering, based on [`BBoxSource`].
///
/// If `TrackingBox` is selected but no tracking box exists, falls back to
/// the detection box.
pub fn resolve_bbox(obj: &BorrowedVideoObject, source: BBoxSource) -> ResolvedBBox {
    let bbox: RBBox = match source {
        BBoxSource::TrackingBox => obj
            .get_track_box()
            .unwrap_or_else(|| obj.get_detection_box()),
        BBoxSource::DetectionBox => obj.get_detection_box(),
    };
    ResolvedBBox {
        cx: bbox.get_xc(),
        cy: bbox.get_yc(),
        w: bbox.get_width(),
        h: bbox.get_height(),
        angle: bbox.get_angle().unwrap_or(0.0),
    }
}

/// Expand a rectangle outward by the given padding amounts.
///
/// The input rect is centered at the origin with half-extents `(hw, hh)`.
/// Returns `(left, top, width, height)` in the local coordinate system.
#[inline]
pub fn padded_rect(hw: f32, hh: f32, pad: &PaddingDraw) -> skia_safe::Rect {
    skia_safe::Rect::from_xywh(
        -hw - pad.left as f32,
        -hh - pad.top as f32,
        2.0 * hw + pad.left as f32 + pad.right as f32,
        2.0 * hh + pad.top as f32 + pad.bottom as f32,
    )
}
