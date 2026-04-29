//! Transform helpers for the Picasso pipeline.
//!
//! Provides [`compute_letterbox_params`] to convert a GPU letterbox/resize
//! operation into `VideoFrameTransformation::LetterBox` parameters.

use crate::error::PicassoError;
use deepstream_buffers::{DstPadding, Padding, MIN_EFFECTIVE_DIM};

/// Result of a letterbox parameter computation.
///
/// Describes the padding applied around the scaled image within the
/// destination surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LetterboxParams {
    /// Full destination width (unchanged from input `dst_w`).
    pub outer_width: u64,
    /// Full destination height (unchanged from input `dst_h`).
    pub outer_height: u64,
    /// Left padding in pixels.
    pub pad_left: u64,
    /// Top padding in pixels.
    pub pad_top: u64,
    /// Right padding in pixels.
    pub pad_right: u64,
    /// Bottom padding in pixels.
    pub pad_bottom: u64,
}

/// Compute `VideoFrameTransformation::LetterBox` parameters that describe
/// the GPU letterbox/resize from `(src_w, src_h)` to `(dst_w, dst_h)` with the
/// given padding mode and optional destination padding.
///
/// # Errors
///
/// Returns [`PicassoError::InvalidLetterboxParams`] if `dst_padding` reduces
/// the effective width or height below [`MIN_EFFECTIVE_DIM`].
pub fn compute_letterbox_params(
    src_w: u64,
    src_h: u64,
    dst_w: u64,
    dst_h: u64,
    padding: Padding,
    dst_padding: Option<DstPadding>,
) -> Result<LetterboxParams, PicassoError> {
    let (eff_w, eff_h, offset_left, offset_top) = match dst_padding {
        Some(p) => (
            dst_w
                .saturating_sub(p.left as u64)
                .saturating_sub(p.right as u64),
            dst_h
                .saturating_sub(p.top as u64)
                .saturating_sub(p.bottom as u64),
            p.left as u64,
            p.top as u64,
        ),
        None => (dst_w, dst_h, 0u64, 0u64),
    };

    if eff_w < MIN_EFFECTIVE_DIM as u64 {
        return Err(PicassoError::InvalidLetterboxParams(format!(
            "effective width after dst_padding must be >= {MIN_EFFECTIVE_DIM} px, got {eff_w}"
        )));
    }
    if eff_h < MIN_EFFECTIVE_DIM as u64 {
        return Err(PicassoError::InvalidLetterboxParams(format!(
            "effective height after dst_padding must be >= {MIN_EFFECTIVE_DIM} px, got {eff_h}"
        )));
    }

    if padding == Padding::None {
        return Ok(LetterboxParams {
            outer_width: dst_w,
            outer_height: dst_h,
            pad_left: offset_left,
            pad_top: offset_top,
            pad_right: dst_padding.map_or(0u64, |p| p.right as u64),
            pad_bottom: dst_padding.map_or(0u64, |p| p.bottom as u64),
        });
    }

    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = eff_w as f64 / eff_h as f64;

    let (scaled_w, scaled_h) = if src_aspect > dst_aspect {
        let w = eff_w;
        let h = (eff_w as f64 / src_aspect).round() as u64;
        (w, h)
    } else {
        let h = eff_h;
        let w = (eff_h as f64 * src_aspect).round() as u64;
        (w, h)
    };

    Ok(match padding {
        Padding::Symmetric => {
            let pad_left = (eff_w - scaled_w) / 2;
            let pad_top = (eff_h - scaled_h) / 2;
            let pad_right = eff_w - scaled_w - pad_left;
            let pad_bottom = eff_h - scaled_h - pad_top;
            LetterboxParams {
                outer_width: dst_w,
                outer_height: dst_h,
                pad_left: offset_left + pad_left,
                pad_top: offset_top + pad_top,
                pad_right: dst_padding.map_or(pad_right, |p| p.right as u64 + pad_right),
                pad_bottom: dst_padding.map_or(pad_bottom, |p| p.bottom as u64 + pad_bottom),
            }
        }
        Padding::RightBottom => {
            let pad_right = eff_w - scaled_w;
            let pad_bottom = eff_h - scaled_h;
            LetterboxParams {
                outer_width: dst_w,
                outer_height: dst_h,
                pad_left: offset_left,
                pad_top: offset_top,
                pad_right: dst_padding.map_or(pad_right, |p| p.right as u64 + pad_right),
                pad_bottom: dst_padding.map_or(pad_bottom, |p| p.bottom as u64 + pad_bottom),
            }
        }
        Padding::None => unreachable!(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_params_no_padding() {
        let lp = compute_letterbox_params(800, 600, 1600, 1200, Padding::None, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (1600, 1200));
        assert_eq!(
            (lp.pad_left, lp.pad_top, lp.pad_right, lp.pad_bottom),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn letterbox_params_symmetric_800x600_to_800x800() {
        let lp = compute_letterbox_params(800, 600, 800, 800, Padding::Symmetric, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (800, 800));
        assert_eq!((lp.pad_left, lp.pad_right), (0, 0));
        assert_eq!(lp.pad_top + lp.pad_bottom, 200);
        assert_eq!(lp.pad_top, 100);
        assert_eq!(lp.pad_bottom, 100);
    }

    #[test]
    fn letterbox_params_right_bottom_1920x1080_to_800x600() {
        let lp =
            compute_letterbox_params(1920, 1080, 800, 600, Padding::RightBottom, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (800, 600));
        assert_eq!((lp.pad_left, lp.pad_top), (0, 0));
        assert_eq!(lp.pad_right, 0);
        assert_eq!(lp.pad_bottom, 150);
    }

    #[test]
    fn letterbox_params_symmetric_pillarbox() {
        let lp = compute_letterbox_params(600, 800, 800, 800, Padding::Symmetric, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (800, 800));
        assert_eq!(lp.pad_left + lp.pad_right, 200);
        assert_eq!(lp.pad_left, 100);
        assert_eq!(lp.pad_right, 100);
        assert_eq!((lp.pad_top, lp.pad_bottom), (0, 0));
    }

    #[test]
    fn letterbox_params_no_padding_different_aspect() {
        let lp = compute_letterbox_params(800, 600, 800, 800, Padding::None, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (800, 800));
        assert_eq!(
            (lp.pad_left, lp.pad_top, lp.pad_right, lp.pad_bottom),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn letterbox_params_same_aspect_symmetric() {
        let lp = compute_letterbox_params(1280, 720, 1920, 1080, Padding::Symmetric, None).unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (1920, 1080));
        assert_eq!(
            (lp.pad_left, lp.pad_top, lp.pad_right, lp.pad_bottom),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn letterbox_params_rejects_degenerate_dst_padding_width() {
        let dst_pad = DstPadding {
            left: 400,
            top: 0,
            right: 400,
            bottom: 0,
        };
        let err = compute_letterbox_params(800, 600, 800, 600, Padding::Symmetric, Some(dst_pad))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("effective width"),
            "expected 'effective width' in: {msg}"
        );
    }

    #[test]
    fn letterbox_params_rejects_degenerate_dst_padding_height() {
        let dst_pad = DstPadding {
            left: 0,
            top: 300,
            right: 0,
            bottom: 300,
        };
        let err = compute_letterbox_params(800, 600, 800, 600, Padding::Symmetric, Some(dst_pad))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("effective height"),
            "expected 'effective height' in: {msg}"
        );
    }

    #[test]
    fn letterbox_params_with_dst_padding() {
        let dst_pad = DstPadding {
            left: 10,
            top: 20,
            right: 10,
            bottom: 20,
        };
        let lp = compute_letterbox_params(800, 600, 820, 640, Padding::Symmetric, Some(dst_pad))
            .unwrap();
        assert_eq!((lp.outer_width, lp.outer_height), (820, 640));
        assert_eq!(lp.pad_left, 10);
        assert_eq!(lp.pad_top, 20);
        assert_eq!(lp.pad_right, 10);
        assert_eq!(lp.pad_bottom, 20);
    }
}
