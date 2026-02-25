use anyhow::{bail, Result};

use crate::primitives::frame::VideoFrameTransformation;
use crate::primitives::object::VideoObjectBBoxTransformation;

/// Result of [`Affine2D::from_transformations`].
#[derive(Debug, Clone, PartialEq)]
pub struct TransformationChainResult {
    /// The compound affine that maps initial → current coordinates.
    pub affine: Affine2D,
    /// `(width, height)` from the first `InitialSize` entry, if present.
    pub initial_size: Option<(u64, u64)>,
    /// Accumulated `(width, height)` after all transformations.
    pub current_size: Option<(u64, u64)>,
}

/// Axis-aligned 2D affine transform: `(x, y) → (sx*x + tx, sy*y + ty)`.
///
/// Used to compose the chain of [`VideoFrameTransformation`] entries into a
/// single transform that can be applied to all object bboxes in one shot.
///
/// # Examples
///
/// ```
/// use savant_core::geometry::Affine2D;
/// use savant_core::primitives::frame::VideoFrameTransformation;
///
/// let chain = vec![
///     VideoFrameTransformation::InitialSize(1920, 1080),
///     VideoFrameTransformation::LetterBox(660, 500, 10, 10, 10, 10),
///     VideoFrameTransformation::Padding(5, 5, 5, 5),
/// ];
/// let result = Affine2D::from_transformations(&chain);
/// assert_eq!(result.initial_size, Some((1920, 1080)));
/// let inner_w = 660 - 10 - 10;  // 640
/// assert!((result.affine.sx - inner_w as f32 / 1920.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2D {
    pub sx: f32,
    pub sy: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Affine2D {
    /// Identity transform (no change).
    pub const IDENTITY: Affine2D = Affine2D {
        sx: 1.0,
        sy: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    pub fn new(sx: f32, sy: f32, tx: f32, ty: f32) -> Self {
        Self { sx, sy, tx, ty }
    }

    /// Build the compound **initial → current** affine from a frame's
    /// transformation chain.
    pub fn from_transformations(chain: &[VideoFrameTransformation]) -> TransformationChainResult {
        let mut a = Self::IDENTITY;
        let mut initial_size: Option<(u64, u64)> = None;
        let mut current_w: f64 = 0.0;
        let mut current_h: f64 = 0.0;

        for t in chain {
            match t {
                VideoFrameTransformation::InitialSize(w, h) => {
                    initial_size = Some((*w, *h));
                    current_w = *w as f64;
                    current_h = *h as f64;
                    a = Self::IDENTITY;
                }
                VideoFrameTransformation::LetterBox(
                    outer_w,
                    outer_h,
                    pad_l,
                    pad_t,
                    pad_r,
                    pad_b,
                ) => {
                    let inner_w = (*outer_w - pad_l - pad_r) as f64;
                    let inner_h = (*outer_h - pad_t - pad_b) as f64;
                    if current_w > 0.0 && current_h > 0.0 && inner_w > 0.0 && inner_h > 0.0 {
                        let kx = inner_w / current_w;
                        let ky = inner_h / current_h;
                        a.sx *= kx as f32;
                        a.sy *= ky as f32;
                        a.tx *= kx as f32;
                        a.ty *= ky as f32;
                        a.tx += *pad_l as f32;
                        a.ty += *pad_t as f32;
                        current_w = *outer_w as f64;
                        current_h = *outer_h as f64;
                    }
                }
                VideoFrameTransformation::Padding(l, t_pad, r, b) => {
                    a.tx += *l as f32;
                    a.ty += *t_pad as f32;
                    current_w += (*l + r) as f64;
                    current_h += (*t_pad + b) as f64;
                }
                VideoFrameTransformation::Crop(l, t_crop, r, b) => {
                    a.tx -= *l as f32;
                    a.ty -= *t_crop as f32;
                    current_w -= (*l + r) as f64;
                    current_h -= (*t_crop + b) as f64;
                }
            }
        }

        let current_size = if current_w > 0.0 && current_h > 0.0 {
            Some((current_w as u64, current_h as u64))
        } else {
            None
        };

        TransformationChainResult {
            affine: a,
            initial_size,
            current_size,
        }
    }

    /// Invert the affine: maps **current → initial**.
    ///
    /// # Panics
    ///
    /// Panics if `sx` or `sy` is zero (degenerate transform).
    pub fn inverse(&self) -> Self {
        assert!(
            self.sx.abs() > f32::EPSILON && self.sy.abs() > f32::EPSILON,
            "Cannot invert degenerate affine (sx={}, sy={})",
            self.sx,
            self.sy
        );
        Self {
            sx: 1.0 / self.sx,
            sy: 1.0 / self.sy,
            tx: -self.tx / self.sx,
            ty: -self.ty / self.sy,
        }
    }

    /// Compose `self` with a subsequent uniform scale that maps
    /// `initial_size → target_size`.
    pub fn then_scale_to(
        &self,
        target_w: f32,
        target_h: f32,
        initial_w: f32,
        initial_h: f32,
    ) -> Self {
        let kx = target_w / initial_w;
        let ky = target_h / initial_h;
        Self {
            sx: self.sx * kx,
            sy: self.sy * ky,
            tx: self.tx * kx,
            ty: self.ty * ky,
        }
    }

    /// Compose two affines: apply `self` first, then `other`.
    ///
    /// `other(self(x)) = other.sx * (self.sx * x + self.tx) + other.tx`
    pub fn then(&self, other: &Self) -> Self {
        Self {
            sx: other.sx * self.sx,
            sy: other.sy * self.sy,
            tx: other.sx * self.tx + other.tx,
            ty: other.sy * self.ty + other.ty,
        }
    }

    /// Convert to [`VideoObjectBBoxTransformation`] ops that can be passed
    /// to [`crate::primitives::frame::VideoFrameProxy::transform_geometry()`].
    pub fn to_bbox_ops(&self) -> Vec<VideoObjectBBoxTransformation> {
        vec![
            VideoObjectBBoxTransformation::Scale(self.sx, self.sy),
            VideoObjectBBoxTransformation::Shift(self.tx, self.ty),
        ]
    }
}

/// How the source image is placed within the destination rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LetterBoxKind {
    /// Stretch the source to fill the destination (no padding).
    Stretch,
    /// Maintain aspect ratio; center the image with equal padding on both
    /// sides.
    Symmetric,
    /// Maintain aspect ratio; place the image at top-left, padding goes to
    /// right and bottom.
    RightBottom,
}

/// Crop rectangle within the source frame, specified as
/// `(left, top, width, height)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    pub left: u64,
    pub top: u64,
    pub width: u64,
    pub height: u64,
}

/// Describes how a source frame is cropped and scaled into a destination
/// rectangle.
///
/// Use [`ScaleSpec::to_transformations`] to produce the corresponding
/// [`VideoFrameTransformation`] chain.
///
/// # Examples
///
/// ```
/// use savant_core::geometry::{ScaleSpec, CropRect, LetterBoxKind};
/// use savant_core::primitives::frame::VideoFrameTransformation;
///
/// // No crop, symmetric letterbox
/// let spec = ScaleSpec {
///     source_width: 1920,
///     source_height: 1080,
///     dest_width: 800,
///     dest_height: 800,
///     letterbox: LetterBoxKind::Symmetric,
///     crop: None,
/// };
/// let chain = spec.to_transformations().unwrap();
/// assert_eq!(chain.len(), 1);
/// assert!(matches!(chain[0], VideoFrameTransformation::LetterBox(..)));
///
/// // With crop
/// let spec = ScaleSpec {
///     source_width: 1920,
///     source_height: 1080,
///     dest_width: 800,
///     dest_height: 600,
///     letterbox: LetterBoxKind::Stretch,
///     crop: Some(CropRect { left: 100, top: 50, width: 800, height: 600 }),
/// };
/// let chain = spec.to_transformations().unwrap();
/// assert_eq!(chain.len(), 2);
/// assert!(matches!(chain[0], VideoFrameTransformation::Crop(..)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleSpec {
    /// Source frame width (must be > 0).
    pub source_width: u64,
    /// Source frame height (must be > 0).
    pub source_height: u64,
    /// Target output width (must be > 0).
    pub dest_width: u64,
    /// Target output height (must be > 0).
    pub dest_height: u64,
    /// Letterbox placement variant.
    pub letterbox: LetterBoxKind,
    /// Optional crop within the source frame.  Must lie entirely within the
    /// source dimensions and have non-zero width/height.
    pub crop: Option<CropRect>,
}

impl ScaleSpec {
    /// Build the [`VideoFrameTransformation`] chain described by this spec.
    ///
    /// Returns an optional `Crop` followed by a `LetterBox`.
    ///
    /// # Errors
    ///
    /// Returns an error when any dimension is zero or the crop rectangle
    /// extends beyond the source frame.
    pub fn to_transformations(&self) -> Result<Vec<VideoFrameTransformation>> {
        let (src_w, src_h) = (self.source_width, self.source_height);
        let (dst_w, dst_h) = (self.dest_width, self.dest_height);

        if src_w == 0 || src_h == 0 {
            bail!("source dimensions must be > 0, got {src_w}x{src_h}");
        }
        if dst_w == 0 || dst_h == 0 {
            bail!("destination dimensions must be > 0, got {dst_w}x{dst_h}");
        }

        let mut out = Vec::with_capacity(2);

        let (lb_src_w, lb_src_h) = if let Some(c) = self.crop {
            if c.width == 0 || c.height == 0 {
                bail!("crop dimensions must be > 0, got {}x{}", c.width, c.height);
            }
            if c.left + c.width > src_w {
                bail!(
                    "crop right edge ({}) exceeds source width ({src_w})",
                    c.left + c.width
                );
            }
            if c.top + c.height > src_h {
                bail!(
                    "crop bottom edge ({}) exceeds source height ({src_h})",
                    c.top + c.height
                );
            }

            let crop_r = src_w - c.left - c.width;
            let crop_b = src_h - c.top - c.height;
            out.push(VideoFrameTransformation::Crop(
                c.left, c.top, crop_r, crop_b,
            ));
            (c.width, c.height)
        } else {
            (src_w, src_h)
        };

        let (pad_l, pad_t, pad_r, pad_b) =
            compute_letterbox_padding(lb_src_w, lb_src_h, dst_w, dst_h, self.letterbox);
        out.push(VideoFrameTransformation::LetterBox(
            dst_w, dst_h, pad_l, pad_t, pad_r, pad_b,
        ));

        Ok(out)
    }
}

/// Compute the letterbox padding given a source size, destination size, and
/// placement variant.
///
/// Returns `(pad_left, pad_top, pad_right, pad_bottom)`.
fn compute_letterbox_padding(
    src_w: u64,
    src_h: u64,
    dst_w: u64,
    dst_h: u64,
    kind: LetterBoxKind,
) -> (u64, u64, u64, u64) {
    if kind == LetterBoxKind::Stretch {
        return (0, 0, 0, 0);
    }

    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;

    let (scaled_w, scaled_h) = if src_aspect > dst_aspect {
        (dst_w, (dst_w as f64 / src_aspect).round() as u64)
    } else {
        ((dst_h as f64 * src_aspect).round() as u64, dst_h)
    };

    match kind {
        LetterBoxKind::Symmetric => {
            let pad_l = (dst_w - scaled_w) / 2;
            let pad_t = (dst_h - scaled_h) / 2;
            let pad_r = dst_w - scaled_w - pad_l;
            let pad_b = dst_h - scaled_h - pad_t;
            (pad_l, pad_t, pad_r, pad_b)
        }
        LetterBoxKind::RightBottom => {
            let pad_r = dst_w - scaled_w;
            let pad_b = dst_h - scaled_h;
            (0, 0, pad_r, pad_b)
        }
        LetterBoxKind::Stretch => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::frame::VideoFrameTransformation;

    #[test]
    fn identity() {
        let a = Affine2D::IDENTITY;
        assert_eq!(a.sx, 1.0);
        assert_eq!(a.sy, 1.0);
        assert_eq!(a.tx, 0.0);
        assert_eq!(a.ty, 0.0);
    }

    #[test]
    fn from_initial_only() {
        let chain = vec![VideoFrameTransformation::InitialSize(1920, 1080)];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let initial = r.initial_size;
        assert_eq!(a, Affine2D::IDENTITY);
        assert_eq!(initial, Some((1920, 1080)));
    }

    #[test]
    fn from_initial_and_letterbox_no_padding() {
        // LetterBox with zero padding is equivalent to the old Scale
        let chain = vec![
            VideoFrameTransformation::InitialSize(1920, 1080),
            VideoFrameTransformation::LetterBox(960, 540, 0, 0, 0, 0),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let initial = r.initial_size;
        assert_eq!(initial, Some((1920, 1080)));
        assert!((a.sx - 0.5).abs() < 1e-6);
        assert!((a.sy - 0.5).abs() < 1e-6);
        assert!((a.tx - 0.0).abs() < 1e-6);
        assert!((a.ty - 0.0).abs() < 1e-6);
    }

    #[test]
    fn from_letterbox_with_padding() {
        // LetterBox(660, 500, 10, 10, 10, 10) → inner 640×480, then padded
        let chain = vec![
            VideoFrameTransformation::InitialSize(1920, 1080),
            VideoFrameTransformation::LetterBox(660, 500, 10, 10, 10, 10),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let expected_sx = 640.0 / 1920.0;
        let expected_sy = 480.0 / 1080.0;
        assert!((a.sx - expected_sx).abs() < 1e-6);
        assert!((a.sy - expected_sy).abs() < 1e-6);
        assert!((a.tx - 10.0).abs() < 1e-6);
        assert!((a.ty - 10.0).abs() < 1e-6);
    }

    #[test]
    fn from_letterbox_and_extra_padding() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(1920, 1080),
            VideoFrameTransformation::LetterBox(660, 500, 10, 10, 10, 10),
            VideoFrameTransformation::Padding(5, 5, 5, 5),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let expected_sx = 640.0 / 1920.0;
        let expected_sy = 480.0 / 1080.0;
        assert!((a.sx - expected_sx).abs() < 1e-6);
        assert!((a.sy - expected_sy).abs() < 1e-6);
        assert!((a.tx - 15.0).abs() < 1e-6);
        assert!((a.ty - 15.0).abs() < 1e-6);
    }

    #[test]
    fn crop_basic() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(1000, 1000),
            VideoFrameTransformation::Crop(100, 50, 100, 50),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        assert!((a.sx - 1.0).abs() < 1e-6);
        assert!((a.sy - 1.0).abs() < 1e-6);
        assert!((a.tx - (-100.0)).abs() < 1e-6);
        assert!((a.ty - (-50.0)).abs() < 1e-6);
    }

    #[test]
    fn crop_then_letterbox() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(1920, 1080),
            VideoFrameTransformation::Crop(160, 40, 160, 40),
            VideoFrameTransformation::LetterBox(800, 500, 0, 0, 0, 0),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        // After crop: 1600×1000, then scale to 800×500
        let kx = 800.0 / 1600.0;
        let ky = 500.0 / 1000.0;
        assert!((a.sx - kx).abs() < 1e-6);
        assert!((a.sy - ky).abs() < 1e-6);
        assert!((a.tx - (-160.0 * kx)).abs() < 1e-3);
        assert!((a.ty - (-40.0 * ky)).abs() < 1e-3);
    }

    #[test]
    fn letterbox_then_crop() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(1000, 1000),
            VideoFrameTransformation::LetterBox(520, 520, 10, 10, 10, 10),
            VideoFrameTransformation::Crop(10, 10, 10, 10),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        // LetterBox inner: 500×500, kx=ky=0.5, tx=ty=10
        // Crop: tx -= 10, ty -= 10
        assert!((a.sx - 0.5).abs() < 1e-6);
        assert!((a.sy - 0.5).abs() < 1e-6);
        assert!(a.tx.abs() < 1e-6);
        assert!(a.ty.abs() < 1e-6);
    }

    #[test]
    fn padding_then_crop_cancel() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(800, 600),
            VideoFrameTransformation::Padding(20, 20, 20, 20),
            VideoFrameTransformation::Crop(20, 20, 20, 20),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        assert!((a.sx - 1.0).abs() < 1e-6);
        assert!((a.sy - 1.0).abs() < 1e-6);
        assert!(a.tx.abs() < 1e-6);
        assert!(a.ty.abs() < 1e-6);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = Affine2D::new(0.5, 0.25, 10.0, 20.0);
        let inv = a.inverse();
        let roundtrip = a.then(&inv);
        assert!((roundtrip.sx - 1.0).abs() < 1e-5);
        assert!((roundtrip.sy - 1.0).abs() < 1e-5);
        assert!(roundtrip.tx.abs() < 1e-4);
        assert!(roundtrip.ty.abs() < 1e-4);
    }

    #[test]
    fn then_composition() {
        let a = Affine2D::new(2.0, 3.0, 10.0, 20.0);
        let b = Affine2D::new(0.5, 0.5, 5.0, 5.0);
        let c = a.then(&b);
        assert!((c.sx - 1.0).abs() < 1e-6);
        assert!((c.sy - 1.5).abs() < 1e-6);
        assert!((c.tx - 10.0).abs() < 1e-6);
        assert!((c.ty - 15.0).abs() < 1e-6);
    }

    #[test]
    fn then_scale_to() {
        // LetterBox(660, 500, 10, 10, 10, 10) is equivalent to
        // Scale(640, 480) + Padding(10, 10, 10, 10)
        let chain = vec![
            VideoFrameTransformation::InitialSize(1920, 1080),
            VideoFrameTransformation::LetterBox(660, 500, 10, 10, 10, 10),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let initial = r.initial_size;
        let (iw, ih) = initial.unwrap();
        let inv = a.inverse();
        let target = inv.then_scale_to(800.0, 600.0, iw as f32, ih as f32);
        let x = target.sx * 0.0 + target.tx;
        let y = target.sy * 0.0 + target.ty;
        let inv_x = -10.0 / a.sx;
        let inv_y = -10.0 / a.sy;
        let expected_x = inv_x * (800.0 / 1920.0);
        let expected_y = inv_y * (600.0 / 1080.0);
        assert!((x - expected_x).abs() < 1e-3);
        assert!((y - expected_y).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "Cannot invert degenerate")]
    fn inverse_degenerate_panics() {
        let a = Affine2D::new(0.0, 1.0, 0.0, 0.0);
        let _ = a.inverse();
    }

    #[test]
    fn to_bbox_ops() {
        let a = Affine2D::new(2.0, 3.0, 10.0, 20.0);
        let ops = a.to_bbox_ops();
        assert_eq!(ops.len(), 2);
        match ops[0] {
            VideoObjectBBoxTransformation::Scale(sx, sy) => {
                assert!((sx - 2.0).abs() < 1e-6);
                assert!((sy - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected Scale"),
        }
        match ops[1] {
            VideoObjectBBoxTransformation::Shift(tx, ty) => {
                assert!((tx - 10.0).abs() < 1e-6);
                assert!((ty - 20.0).abs() < 1e-6);
            }
            _ => panic!("Expected Shift"),
        }
    }

    #[test]
    fn double_letterbox() {
        // Two successive LetterBox with no padding = two scales
        let chain = vec![
            VideoFrameTransformation::InitialSize(1000, 1000),
            VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0),
            VideoFrameTransformation::LetterBox(250, 250, 0, 0, 0, 0),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        assert!((a.sx - 0.25).abs() < 1e-6);
        assert!((a.sy - 0.25).abs() < 1e-6);
    }

    #[test]
    fn letterbox_padding_letterbox() {
        // LetterBox(no pad) + Padding + LetterBox(no pad) combo
        let chain = vec![
            VideoFrameTransformation::InitialSize(1000, 1000),
            VideoFrameTransformation::LetterBox(500, 500, 0, 0, 0, 0),
            VideoFrameTransformation::Padding(50, 50, 50, 50),
            VideoFrameTransformation::LetterBox(300, 300, 0, 0, 0, 0),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        // After LetterBox(500,500): kx=ky=0.5, current=500×500
        // After Padding(50): tx=ty=50, current=600×600
        // After LetterBox(300,300): kx=ky=300/600=0.5
        //   sx = 0.5*0.5 = 0.25, tx = 50*0.5 = 25
        assert!((a.sx - 0.25).abs() < 1e-6);
        assert!((a.sy - 0.25).abs() < 1e-6);
        assert!((a.tx - 25.0).abs() < 1e-6);
        assert!((a.ty - 25.0).abs() < 1e-6);
    }

    #[test]
    fn complex_chain_crop_letterbox_padding() {
        let chain = vec![
            VideoFrameTransformation::InitialSize(2000, 1000),
            VideoFrameTransformation::Crop(200, 100, 200, 100),
            VideoFrameTransformation::LetterBox(840, 420, 20, 10, 20, 10),
            VideoFrameTransformation::Padding(5, 5, 5, 5),
        ];
        let r = Affine2D::from_transformations(&chain);
        let a = r.affine;
        let initial = r.initial_size;
        assert_eq!(initial, Some((2000, 1000)));
        // After crop: 1600×800, tx=-200, ty=-100
        // LetterBox inner: 800×400, kx=800/1600=0.5, ky=400/800=0.5
        //   sx=0.5, sy=0.5, tx=-200*0.5+20=-80, ty=-100*0.5+10=-40
        // Padding(5): tx=-80+5=-75, ty=-40+5=-35
        assert!((a.sx - 0.5).abs() < 1e-5);
        assert!((a.sy - 0.5).abs() < 1e-5);
        assert!((a.tx - (-75.0)).abs() < 1e-3);
        assert!((a.ty - (-35.0)).abs() < 1e-3);
    }

    // ---------------------------------------------------------------
    // ScaleSpec::to_transformations tests
    // ---------------------------------------------------------------

    mod scale_spec {
        use super::*;

        fn spec(
            sw: u64,
            sh: u64,
            dw: u64,
            dh: u64,
            lb: LetterBoxKind,
            crop: Option<CropRect>,
        ) -> ScaleSpec {
            ScaleSpec {
                source_width: sw,
                source_height: sh,
                dest_width: dw,
                dest_height: dh,
                letterbox: lb,
                crop,
            }
        }

        // ---- error cases ----

        #[test]
        fn zero_source_width() {
            let r = spec(0, 1080, 800, 600, LetterBoxKind::Stretch, None).to_transformations();
            assert!(r.is_err());
            assert!(r.unwrap_err().to_string().contains("source"));
        }

        #[test]
        fn zero_source_height() {
            let r = spec(1920, 0, 800, 600, LetterBoxKind::Stretch, None).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn zero_dest_width() {
            let r = spec(1920, 1080, 0, 600, LetterBoxKind::Stretch, None).to_transformations();
            assert!(r.is_err());
            assert!(r.unwrap_err().to_string().contains("destination"));
        }

        #[test]
        fn zero_dest_height() {
            let r = spec(1920, 1080, 800, 0, LetterBoxKind::Stretch, None).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn crop_zero_width() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 0,
                height: 600,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
            assert!(r.unwrap_err().to_string().contains("crop dimensions"));
        }

        #[test]
        fn crop_zero_height() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 800,
                height: 0,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn crop_right_edge_exceeds_source() {
            let c = CropRect {
                left: 1000,
                top: 0,
                width: 1000,
                height: 1080,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
            assert!(r.unwrap_err().to_string().contains("right edge"));
        }

        #[test]
        fn crop_bottom_edge_exceeds_source() {
            let c = CropRect {
                left: 0,
                top: 500,
                width: 1920,
                height: 600,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
            assert!(r.unwrap_err().to_string().contains("bottom edge"));
        }

        #[test]
        fn crop_left_equals_source_width() {
            let c = CropRect {
                left: 1920,
                top: 0,
                width: 1,
                height: 1080,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn crop_top_equals_source_height() {
            let c = CropRect {
                left: 0,
                top: 1080,
                width: 1920,
                height: 1,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn crop_right_edge_exactly_at_boundary() {
            let c = CropRect {
                left: 920,
                top: 0,
                width: 1000,
                height: 1080,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_ok());
        }

        #[test]
        fn crop_bottom_edge_exactly_at_boundary() {
            let c = CropRect {
                left: 0,
                top: 480,
                width: 1920,
                height: 600,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_ok());
        }

        // ---- no crop, stretch ----

        #[test]
        fn no_crop_stretch() {
            let chain = spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, None)
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 1);
            assert_eq!(
                chain[0],
                VideoFrameTransformation::LetterBox(800, 600, 0, 0, 0, 0)
            );
        }

        // ---- no crop, symmetric ----

        #[test]
        fn no_crop_symmetric_wider_source() {
            let chain = spec(1920, 1080, 800, 800, LetterBoxKind::Symmetric, None)
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 1);
            match chain[0] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (800, 800));
                    assert_eq!(pl, pr);
                    assert_eq!((pl, pr), (0, 0));
                    assert_eq!(pt + pb, 800 - 450);
                }
                _ => panic!("expected LetterBox"),
            }
        }

        #[test]
        fn no_crop_symmetric_taller_source() {
            let chain = spec(600, 800, 800, 800, LetterBoxKind::Symmetric, None)
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 1);
            match chain[0] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (800, 800));
                    assert_eq!(pt, pb);
                    assert_eq!((pt, pb), (0, 0));
                    assert_eq!(pl + pr, 200);
                }
                _ => panic!("expected LetterBox"),
            }
        }

        #[test]
        fn no_crop_symmetric_same_aspect() {
            let chain = spec(1280, 720, 1920, 1080, LetterBoxKind::Symmetric, None)
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 1);
            assert_eq!(
                chain[0],
                VideoFrameTransformation::LetterBox(1920, 1080, 0, 0, 0, 0)
            );
        }

        // ---- no crop, right-bottom ----

        #[test]
        fn no_crop_right_bottom() {
            let chain = spec(1920, 1080, 800, 600, LetterBoxKind::RightBottom, None)
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 1);
            match chain[0] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (800, 600));
                    assert_eq!((pl, pt), (0, 0));
                    assert_eq!(pr, 0);
                    assert_eq!(pb, 150);
                }
                _ => panic!("expected LetterBox"),
            }
        }

        // ---- with crop, stretch ----

        #[test]
        fn crop_center_stretch() {
            let c = CropRect {
                left: 480,
                top: 270,
                width: 960,
                height: 540,
            };
            let chain = spec(1920, 1080, 960, 540, LetterBoxKind::Stretch, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], VideoFrameTransformation::Crop(480, 270, 480, 270));
            assert_eq!(
                chain[1],
                VideoFrameTransformation::LetterBox(960, 540, 0, 0, 0, 0)
            );
        }

        #[test]
        fn crop_top_left_stretch() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 960,
                height: 540,
            };
            let chain = spec(1920, 1080, 480, 270, LetterBoxKind::Stretch, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], VideoFrameTransformation::Crop(0, 0, 960, 540));
            assert_eq!(
                chain[1],
                VideoFrameTransformation::LetterBox(480, 270, 0, 0, 0, 0)
            );
        }

        #[test]
        fn crop_bottom_right_corner_stretch() {
            let c = CropRect {
                left: 960,
                top: 540,
                width: 960,
                height: 540,
            };
            let chain = spec(1920, 1080, 960, 540, LetterBoxKind::Stretch, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], VideoFrameTransformation::Crop(960, 540, 0, 0));
            assert_eq!(
                chain[1],
                VideoFrameTransformation::LetterBox(960, 540, 0, 0, 0, 0)
            );
        }

        // ---- with crop, symmetric letterbox ----

        #[test]
        fn crop_then_symmetric_letterbox() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 1600,
                height: 900,
            };
            let chain = spec(1920, 1080, 800, 800, LetterBoxKind::Symmetric, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], VideoFrameTransformation::Crop(0, 0, 320, 180));
            match chain[1] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (800, 800));
                    assert_eq!((pl, pr), (0, 0));
                    assert_eq!(pt, 175);
                    assert_eq!(pb, 175);
                }
                _ => panic!("expected LetterBox"),
            }
        }

        // ---- full-frame crop (crop = entire source) ----

        #[test]
        fn full_frame_crop_is_noop_crop() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 1920,
                height: 1080,
            };
            let chain = spec(1920, 1080, 960, 540, LetterBoxKind::Stretch, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], VideoFrameTransformation::Crop(0, 0, 0, 0));
            assert_eq!(
                chain[1],
                VideoFrameTransformation::LetterBox(960, 540, 0, 0, 0, 0)
            );
        }

        // ---- 1-pixel crop (minimum valid) ----

        #[test]
        fn minimal_1x1_crop() {
            let c = CropRect {
                left: 100,
                top: 200,
                width: 1,
                height: 1,
            };
            let chain = spec(1920, 1080, 100, 100, LetterBoxKind::Stretch, Some(c))
                .to_transformations()
                .unwrap();
            assert_eq!(chain.len(), 2);
            assert_eq!(
                chain[0],
                VideoFrameTransformation::Crop(100, 200, 1819, 879)
            );
        }

        // ---- affine consistency ----

        #[test]
        fn affine_consistency_no_crop_stretch() {
            let chain = spec(1920, 1080, 640, 480, LetterBoxKind::Stretch, None)
                .to_transformations()
                .unwrap();
            let mut full = vec![VideoFrameTransformation::InitialSize(1920, 1080)];
            full.extend(chain);
            let r = Affine2D::from_transformations(&full);
            assert_eq!(r.current_size, Some((640, 480)));
        }

        #[test]
        fn affine_consistency_crop_symmetric() {
            let c = CropRect {
                left: 200,
                top: 100,
                width: 1520,
                height: 880,
            };
            let chain = spec(1920, 1080, 800, 600, LetterBoxKind::Symmetric, Some(c))
                .to_transformations()
                .unwrap();
            let mut full = vec![VideoFrameTransformation::InitialSize(1920, 1080)];
            full.extend(chain);
            let r = Affine2D::from_transformations(&full);
            assert_eq!(r.current_size, Some((800, 600)));
            assert_eq!(r.initial_size, Some((1920, 1080)));
        }

        #[test]
        fn affine_consistency_crop_right_bottom() {
            let c = CropRect {
                left: 0,
                top: 0,
                width: 960,
                height: 540,
            };
            let chain = spec(1920, 1080, 1280, 720, LetterBoxKind::RightBottom, Some(c))
                .to_transformations()
                .unwrap();
            let mut full = vec![VideoFrameTransformation::InitialSize(1920, 1080)];
            full.extend(chain);
            let r = Affine2D::from_transformations(&full);
            assert_eq!(r.current_size, Some((1280, 720)));
        }

        // ---- crop exceeding source by 1 pixel ----

        #[test]
        fn crop_right_exceeds_by_one() {
            let c = CropRect {
                left: 960,
                top: 0,
                width: 961,
                height: 1080,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
        }

        #[test]
        fn crop_bottom_exceeds_by_one() {
            let c = CropRect {
                left: 0,
                top: 540,
                width: 1920,
                height: 541,
            };
            let r =
                spec(1920, 1080, 800, 600, LetterBoxKind::Stretch, Some(c)).to_transformations();
            assert!(r.is_err());
        }

        // ---- non-standard aspect ratios ----

        #[test]
        fn tall_source_to_wide_dest_symmetric() {
            let chain = spec(100, 1000, 1000, 100, LetterBoxKind::Symmetric, None)
                .to_transformations()
                .unwrap();
            match chain[0] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (1000, 100));
                    let scaled_w = 10u64;
                    assert_eq!(pl + pr, 1000 - scaled_w);
                    assert_eq!((pt, pb), (0, 0));
                }
                _ => panic!("expected LetterBox"),
            }
        }

        #[test]
        fn square_to_non_square_right_bottom() {
            let chain = spec(500, 500, 1000, 500, LetterBoxKind::RightBottom, None)
                .to_transformations()
                .unwrap();
            match chain[0] {
                VideoFrameTransformation::LetterBox(ow, oh, pl, pt, pr, pb) => {
                    assert_eq!((ow, oh), (1000, 500));
                    assert_eq!((pl, pt), (0, 0));
                    assert_eq!(pr, 500);
                    assert_eq!(pb, 0);
                }
                _ => panic!("expected LetterBox"),
            }
        }
    }
}
