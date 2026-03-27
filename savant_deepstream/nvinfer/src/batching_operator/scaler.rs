//! Coordinate scaler for mapping model-space predictions back to absolute
//! frame coordinates.
//!
//! The transform is a 2D axis-aligned affine: `frame_xy = offset + model_xy * scale`.
//! The four coefficients are precomputed from the ROI crop geometry, model input
//! dimensions, and [`ModelInputScaling`] mode.

use crate::model_input_scaling::ModelInputScaling;
use savant_core::primitives::RBBox;
use std::f32::consts::PI;

/// Precomputed affine coefficients for mapping model-space coordinates to
/// absolute frame coordinates.
///
/// Created via [`CoordinateScaler::new`] from the ROI rectangle (in frame
/// pixels), the model input dimensions, and the scaling mode.  All `scale_*`
/// methods then apply `frame_xy = offset + model_xy * scale`.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateScaler {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}

impl CoordinateScaler {
    /// Compute the inverse-transform coefficients.
    ///
    /// # Arguments
    ///
    /// * `roi_left`, `roi_top`, `roi_w`, `roi_h` — the axis-aligned ROI
    ///   rectangle in frame coordinates (as produced by
    ///   [`crate::batch_meta_builder::rbbox_to_rect_params`]).
    /// * `model_w`, `model_h` — the model's input tensor dimensions.
    /// * `scaling` — how the ROI crop was resized to the model input.
    pub fn new(
        roi_left: f32,
        roi_top: f32,
        roi_w: f32,
        roi_h: f32,
        model_w: f32,
        model_h: f32,
        scaling: ModelInputScaling,
    ) -> Self {
        match scaling {
            ModelInputScaling::Fill => Self {
                scale_x: roi_w / model_w,
                scale_y: roi_h / model_h,
                offset_x: roi_left,
                offset_y: roi_top,
            },
            ModelInputScaling::KeepAspectRatio => {
                let s = (model_w / roi_w).min(model_h / roi_h);
                let inv_s = 1.0 / s;
                Self {
                    scale_x: inv_s,
                    scale_y: inv_s,
                    offset_x: roi_left,
                    offset_y: roi_top,
                }
            }
            ModelInputScaling::KeepAspectRatioSymmetric => {
                let s = (model_w / roi_w).min(model_h / roi_h);
                let inv_s = 1.0 / s;
                let pad_x = (model_w - roi_w * s) * 0.5;
                let pad_y = (model_h - roi_h * s) * 0.5;
                Self {
                    scale_x: inv_s,
                    scale_y: inv_s,
                    offset_x: roi_left - pad_x * inv_s,
                    offset_y: roi_top - pad_y * inv_s,
                }
            }
        }
    }

    /// Transform a single point from model space to frame coordinates.
    #[inline]
    pub fn scale_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.scale_x + self.offset_x,
            y * self.scale_y + self.offset_y,
        )
    }

    /// Batch-transform points from model space to frame coordinates.
    pub fn scale_points(&self, points: &[(f32, f32)]) -> Vec<(f32, f32)> {
        points
            .iter()
            .map(|&(x, y)| self.scale_point(x, y))
            .collect()
    }

    /// Transform a single axis-aligned box given as (left, top, width, height).
    #[inline]
    pub fn scale_ltwh(&self, l: f32, t: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        (
            l * self.scale_x + self.offset_x,
            t * self.scale_y + self.offset_y,
            w * self.scale_x,
            h * self.scale_y,
        )
    }

    /// Batch-transform axis-aligned boxes given as `[left, top, width, height]`.
    pub fn scale_ltwh_batch(&self, boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
        boxes
            .iter()
            .map(|b| {
                let (l, t, w, h) = self.scale_ltwh(b[0], b[1], b[2], b[3]);
                [l, t, w, h]
            })
            .collect()
    }

    /// Transform a single axis-aligned box given as (left, top, right, bottom).
    #[inline]
    pub fn scale_ltrb(&self, l: f32, t: f32, r: f32, b: f32) -> (f32, f32, f32, f32) {
        (
            l * self.scale_x + self.offset_x,
            t * self.scale_y + self.offset_y,
            r * self.scale_x + self.offset_x,
            b * self.scale_y + self.offset_y,
        )
    }

    /// Batch-transform axis-aligned boxes given as `[left, top, right, bottom]`.
    pub fn scale_ltrb_batch(&self, boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
        boxes
            .iter()
            .map(|b| {
                let (l, t, r, bot) = self.scale_ltrb(b[0], b[1], b[2], b[3]);
                [l, t, r, bot]
            })
            .collect()
    }

    /// Transform an [`RBBox`] from model space to frame coordinates.
    ///
    /// For axis-aligned boxes (angle is `None` or a multiple of 90°), center
    /// and dimensions are scaled independently.  For rotated boxes the angle
    /// is recomputed using the same trigonometry as [`RBBox::scale`] to handle
    /// non-uniform scaling correctly.  The result is a fresh `RBBox` built
    /// with a single `RBBox::new` call (no intermediate atomics).
    pub fn scale_rbbox(&self, bbox: &RBBox) -> RBBox {
        let angle = bbox.get_angle().unwrap_or(0.0);
        let xc = bbox.get_xc();
        let yc = bbox.get_yc();
        let width = bbox.get_width();
        let height = bbox.get_height();

        let new_xc = xc * self.scale_x + self.offset_x;
        let new_yc = yc * self.scale_y + self.offset_y;

        if angle % 90.0 == 0.0 {
            RBBox::new(
                new_xc,
                new_yc,
                width * self.scale_x,
                height * self.scale_y,
                bbox.get_angle(),
            )
        } else {
            let sx2 = self.scale_x * self.scale_x;
            let sy2 = self.scale_y * self.scale_y;
            let cotan = (angle * PI / 180.0).tan().powi(-1);
            let cotan_2 = cotan * cotan;
            let scale_angle = (self.scale_x * angle.signum() / (sx2 + sy2 * cotan_2).sqrt()).acos();
            let nscale_height = ((sx2 + sy2 * cotan_2) / (1.0 + cotan_2)).sqrt();
            let ayh = 1.0 / ((90.0 - angle) / 180.0 * PI).tan();
            let nscale_width = ((sx2 + sy2 * ayh * ayh) / (1.0 + ayh * ayh)).sqrt();

            RBBox::new(
                new_xc,
                new_yc,
                width * nscale_width,
                height * nscale_height,
                Some(90.0 - (scale_angle * 180.0 / PI)),
            )
        }
    }

    /// Batch-transform [`RBBox`]es from model space to frame coordinates.
    pub fn scale_rbboxes(&self, boxes: &[RBBox]) -> Vec<RBBox> {
        boxes.iter().map(|b| self.scale_rbbox(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn assert_close(a: f32, b: f32, msg: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{msg}: expected {b}, got {a} (diff={})",
            (a - b).abs()
        );
    }

    // ── Fill mode ──────────────────────────────────────────────────

    #[test]
    fn fill_identity() {
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        let (x, y) = s.scale_point(50.0, 50.0);
        assert_close(x, 50.0, "x");
        assert_close(y, 50.0, "y");
    }

    #[test]
    fn fill_with_roi_offset() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            100.0,
            200.0,
            50.0,
            100.0,
            ModelInputScaling::Fill,
        );
        // scale_x = 100/50 = 2, scale_y = 200/100 = 2
        let (x, y) = s.scale_point(25.0, 50.0);
        assert_close(x, 10.0 + 25.0 * 2.0, "x");
        assert_close(y, 20.0 + 50.0 * 2.0, "y");
    }

    #[test]
    fn fill_non_square_asymmetric() {
        let s = CoordinateScaler::new(0.0, 0.0, 200.0, 100.0, 100.0, 50.0, ModelInputScaling::Fill);
        // scale_x = 200/100 = 2, scale_y = 100/50 = 2
        let (x, y) = s.scale_point(10.0, 10.0);
        assert_close(x, 20.0, "x");
        assert_close(y, 20.0, "y");

        let s2 = CoordinateScaler::new(
            0.0,
            0.0,
            200.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        // scale_x = 2, scale_y = 1
        let (x, y) = s2.scale_point(10.0, 10.0);
        assert_close(x, 20.0, "x");
        assert_close(y, 10.0, "y");
    }

    #[test]
    fn fill_origin_maps_to_roi_left_top() {
        let s = CoordinateScaler::new(
            50.0,
            100.0,
            200.0,
            300.0,
            64.0,
            64.0,
            ModelInputScaling::Fill,
        );
        let (x, y) = s.scale_point(0.0, 0.0);
        assert_close(x, 50.0, "x");
        assert_close(y, 100.0, "y");
    }

    // ── KeepAspectRatio mode ───────────────────────────────────────

    #[test]
    fn keep_ar_wider_roi() {
        // ROI 200x100, model 100x100 → s = min(100/200, 100/100) = 0.5
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            200.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatio,
        );
        // inv_s = 2.0
        let (x, y) = s.scale_point(50.0, 25.0);
        assert_close(x, 100.0, "x");
        assert_close(y, 50.0, "y");
    }

    #[test]
    fn keep_ar_taller_roi() {
        // ROI 100x200, model 100x100 → s = min(100/100, 100/200) = 0.5
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            200.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatio,
        );
        let (x, y) = s.scale_point(50.0, 25.0);
        assert_close(x, 100.0, "x");
        assert_close(y, 50.0, "y");
    }

    #[test]
    fn keep_ar_square() {
        // ROI 100x100, model 100x100 → identity
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            100.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatio,
        );
        let (x, y) = s.scale_point(50.0, 50.0);
        assert_close(x, 60.0, "x");
        assert_close(y, 70.0, "y");
    }

    // ── KeepAspectRatioSymmetric mode ──────────────────────────────

    #[test]
    fn keep_ar_sym_wider_roi() {
        // ROI 200x100, model 100x100
        // s = min(100/200, 100/100) = 0.5, inv_s = 2
        // pad_x = (100 - 200*0.5)/2 = 0, pad_y = (100 - 100*0.5)/2 = 25
        // offset_x = 0, offset_y = 0 - 25*2 = -50
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            200.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatioSymmetric,
        );
        // The image occupies model rows [25, 75] in y (centered)
        // A point at model (50, 50) → frame (100, 100-50=50)?
        // frame_y = 50 * 2 + (-50) = 50.
        let (x, y) = s.scale_point(50.0, 50.0);
        assert_close(x, 100.0, "x");
        assert_close(y, 50.0, "y");

        // Center of the image region in model space is (50, 50) → frame center
        // The image center in model space is at (50, 50). The ROI center is (100, 50).
        let (cx, cy) = s.scale_point(50.0, 50.0);
        assert_close(cx, 100.0, "center_x");
        assert_close(cy, 50.0, "center_y");
    }

    #[test]
    fn keep_ar_sym_taller_roi() {
        // ROI 100x200, model 100x100
        // s = min(100/100, 100/200) = 0.5, inv_s = 2
        // pad_x = (100 - 100*0.5)/2 = 25, pad_y = 0
        // offset_x = 0 - 25*2 = -50, offset_y = 0
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            200.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatioSymmetric,
        );
        let (x, y) = s.scale_point(50.0, 50.0);
        assert_close(x, 50.0, "x");
        assert_close(y, 100.0, "y");
    }

    #[test]
    fn keep_ar_sym_with_offset() {
        // ROI at (100, 200) size 200x100, model 100x100
        let s = CoordinateScaler::new(
            100.0,
            200.0,
            200.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatioSymmetric,
        );
        // s=0.5, inv_s=2, pad_x=0, pad_y=25
        // offset_x = 100, offset_y = 200 - 25*2 = 150
        let (x, y) = s.scale_point(0.0, 25.0);
        assert_close(x, 100.0, "left edge");
        assert_close(y, 200.0, "top edge");
    }

    // ── LTWH ───────────────────────────────────────────────────────

    #[test]
    fn ltwh_fill() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            200.0,
            400.0,
            100.0,
            200.0,
            ModelInputScaling::Fill,
        );
        // scale_x=2, scale_y=2
        let (l, t, w, h) = s.scale_ltwh(5.0, 10.0, 20.0, 30.0);
        assert_close(l, 10.0 + 5.0 * 2.0, "l");
        assert_close(t, 20.0 + 10.0 * 2.0, "t");
        assert_close(w, 20.0 * 2.0, "w");
        assert_close(h, 30.0 * 2.0, "h");
    }

    #[test]
    fn ltwh_batch() {
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        let input = [[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 100.0, 100.0]];
        let result = s.scale_ltwh_batch(&input);
        assert_eq!(result.len(), 2);
        assert_close(result[0][0], 10.0, "b0 l");
        assert_close(result[1][2], 100.0, "b1 w");
    }

    // ── LTRB ───────────────────────────────────────────────────────

    #[test]
    fn ltrb_fill() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            200.0,
            400.0,
            100.0,
            200.0,
            ModelInputScaling::Fill,
        );
        // scale_x=2, scale_y=2
        let (l, t, r, b) = s.scale_ltrb(5.0, 10.0, 25.0, 40.0);
        assert_close(l, 10.0 + 5.0 * 2.0, "l");
        assert_close(t, 20.0 + 10.0 * 2.0, "t");
        assert_close(r, 10.0 + 25.0 * 2.0, "r");
        assert_close(b, 20.0 + 40.0 * 2.0, "b");
    }

    #[test]
    fn ltrb_ltwh_consistency() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            300.0,
            150.0,
            64.0,
            64.0,
            ModelInputScaling::Fill,
        );
        let (ll, lt, lw, lh) = s.scale_ltwh(5.0, 10.0, 20.0, 30.0);
        let (rl, rt, rr, rb) = s.scale_ltrb(5.0, 10.0, 25.0, 40.0);
        assert_close(ll, rl, "left");
        assert_close(lt, rt, "top");
        assert_close(ll + lw, rr, "right");
        assert_close(lt + lh, rb, "bottom");
    }

    #[test]
    fn ltrb_batch() {
        let s = CoordinateScaler::new(0.0, 0.0, 100.0, 100.0, 50.0, 50.0, ModelInputScaling::Fill);
        let input = [[0.0, 0.0, 50.0, 50.0]];
        let result = s.scale_ltrb_batch(&input);
        assert_close(result[0][2], 100.0, "right");
        assert_close(result[0][3], 100.0, "bottom");
    }

    // ── RBBox ──────────────────────────────────────────────────────

    #[test]
    fn rbbox_axis_aligned() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            200.0,
            400.0,
            100.0,
            200.0,
            ModelInputScaling::Fill,
        );
        let bbox = RBBox::new(50.0, 100.0, 20.0, 30.0, None);
        let out = s.scale_rbbox(&bbox);
        assert_close(out.get_xc(), 10.0 + 50.0 * 2.0, "xc");
        assert_close(out.get_yc(), 20.0 + 100.0 * 2.0, "yc");
        assert_close(out.get_width(), 20.0 * 2.0, "w");
        assert_close(out.get_height(), 30.0 * 2.0, "h");
        assert_eq!(out.get_angle(), None);
    }

    #[test]
    fn rbbox_with_zero_angle() {
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        let bbox = RBBox::new(50.0, 50.0, 20.0, 30.0, Some(0.0));
        let out = s.scale_rbbox(&bbox);
        assert_close(out.get_xc(), 50.0, "xc");
        assert_close(out.get_width(), 20.0, "w");
        assert_eq!(out.get_angle(), Some(0.0));
    }

    #[test]
    fn rbbox_rotated_uniform_scale() {
        // Uniform scale: angle should be preserved
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            200.0,
            200.0,
            100.0,
            100.0,
            ModelInputScaling::KeepAspectRatio,
        );
        let bbox = RBBox::new(50.0, 50.0, 20.0, 30.0, Some(45.0));
        let out = s.scale_rbbox(&bbox);
        assert_close(out.get_xc(), 100.0, "xc");
        assert_close(out.get_yc(), 100.0, "yc");
        // Under uniform scale (2x), angle is preserved, dims doubled
        assert_close(out.get_width(), 40.0, "w");
        assert_close(out.get_height(), 60.0, "h");
        assert_close(out.get_angle().unwrap(), 45.0, "angle");
    }

    #[test]
    fn rbbox_rotated_non_uniform_scale() {
        // Non-uniform fill: angle is recomputed via trig
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            200.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        let bbox = RBBox::new(50.0, 50.0, 20.0, 20.0, Some(45.0));
        let out = s.scale_rbbox(&bbox);
        assert_close(out.get_xc(), 100.0, "xc");
        assert_close(out.get_yc(), 50.0, "yc");
        // Angle should differ from 45 under non-uniform scaling
        assert!(
            (out.get_angle().unwrap() - 45.0).abs() > 0.1,
            "angle must change under non-uniform scale"
        );
    }

    #[test]
    fn rbbox_matches_rbbox_scale_shift() {
        let s = CoordinateScaler::new(
            10.0,
            20.0,
            300.0,
            150.0,
            100.0,
            50.0,
            ModelInputScaling::Fill,
        );
        let bbox = RBBox::new(25.0, 12.5, 10.0, 8.0, Some(30.0));
        let our = s.scale_rbbox(&bbox);

        let reference = bbox.copy();
        reference.scale(s.scale_x, s.scale_y);
        reference.shift(s.offset_x, s.offset_y);

        assert_close(our.get_xc(), reference.get_xc(), "xc");
        assert_close(our.get_yc(), reference.get_yc(), "yc");
        assert_close(our.get_width(), reference.get_width(), "w");
        assert_close(our.get_height(), reference.get_height(), "h");
        assert_close(
            our.get_angle().unwrap(),
            reference.get_angle().unwrap(),
            "angle",
        );
    }

    #[test]
    fn rbbox_batch() {
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            100.0,
            100.0,
            100.0,
            100.0,
            ModelInputScaling::Fill,
        );
        let boxes = vec![
            RBBox::new(10.0, 20.0, 5.0, 5.0, None),
            RBBox::new(30.0, 40.0, 10.0, 10.0, Some(45.0)),
        ];
        let result = s.scale_rbboxes(&boxes);
        assert_eq!(result.len(), 2);
        assert_close(result[0].get_xc(), 10.0, "b0 xc");
        assert_close(result[1].get_xc(), 30.0, "b1 xc");
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn full_frame_identity() {
        // Full frame: ROI = (0,0,W,H), model = (W,H) → identity
        let s = CoordinateScaler::new(
            0.0,
            0.0,
            1920.0,
            1080.0,
            1920.0,
            1080.0,
            ModelInputScaling::Fill,
        );
        let (x, y) = s.scale_point(960.0, 540.0);
        assert_close(x, 960.0, "x");
        assert_close(y, 540.0, "y");
    }
}
