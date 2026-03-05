//! Transform helpers for the Picasso pipeline.
//!
//! Provides [`compute_letterbox_params`] to convert a GPU letterbox/resize
//! operation into [`VideoFrameTransformation::LetterBox`] parameters.

use deepstream_nvbufsurface::{DstPadding, Padding};

/// Compute [`VideoFrameTransformation::LetterBox`] parameters that describe
/// the GPU letterbox/resize from `(src_w, src_h)` to `(dst_w, dst_h)` with the
/// given padding mode and optional destination padding.
///
/// Returns `(outer_w, outer_h, pad_left, pad_top, pad_right, pad_bottom)`.
pub fn compute_letterbox_params(
    src_w: u64,
    src_h: u64,
    dst_w: u64,
    dst_h: u64,
    padding: Padding,
    dst_padding: Option<DstPadding>,
) -> (u64, u64, u64, u64, u64, u64) {
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

    if padding == Padding::None {
        return (
            dst_w,
            dst_h,
            offset_left,
            offset_top,
            dst_padding.map_or(0u64, |p| p.right as u64),
            dst_padding.map_or(0u64, |p| p.bottom as u64),
        );
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

    match padding {
        Padding::Symmetric => {
            let pad_left = (eff_w - scaled_w) / 2;
            let pad_top = (eff_h - scaled_h) / 2;
            let pad_right = eff_w - scaled_w - pad_left;
            let pad_bottom = eff_h - scaled_h - pad_top;
            (
                dst_w,
                dst_h,
                offset_left + pad_left,
                offset_top + pad_top,
                dst_padding.map_or(pad_right, |p| p.right as u64 + pad_right),
                dst_padding.map_or(pad_bottom, |p| p.bottom as u64 + pad_bottom),
            )
        }
        Padding::RightBottom => {
            let pad_right = eff_w - scaled_w;
            let pad_bottom = eff_h - scaled_h;
            (
                dst_w,
                dst_h,
                offset_left,
                offset_top,
                dst_padding.map_or(pad_right, |p| p.right as u64 + pad_right),
                dst_padding.map_or(pad_bottom, |p| p.bottom as u64 + pad_bottom),
            )
        }
        Padding::None => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_params_no_padding() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(800, 600, 1600, 1200, Padding::None, None);
        assert_eq!((ow, oh), (1600, 1200));
        assert_eq!((pl, pt, pr, pb), (0, 0, 0, 0));
    }

    #[test]
    fn letterbox_params_symmetric_800x600_to_800x800() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(800, 600, 800, 800, Padding::Symmetric, None);
        assert_eq!((ow, oh), (800, 800));
        assert_eq!((pl, pr), (0, 0));
        assert_eq!(pt + pb, 200);
        assert_eq!(pt, 100);
        assert_eq!(pb, 100);
    }

    #[test]
    fn letterbox_params_right_bottom_1920x1080_to_800x600() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(1920, 1080, 800, 600, Padding::RightBottom, None);
        assert_eq!((ow, oh), (800, 600));
        assert_eq!((pl, pt), (0, 0));
        assert_eq!(pr, 0);
        assert_eq!(pb, 150);
    }

    #[test]
    fn letterbox_params_symmetric_pillarbox() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(600, 800, 800, 800, Padding::Symmetric, None);
        assert_eq!((ow, oh), (800, 800));
        assert_eq!(pl + pr, 200);
        assert_eq!(pl, 100);
        assert_eq!(pr, 100);
        assert_eq!((pt, pb), (0, 0));
    }

    #[test]
    fn letterbox_params_no_padding_different_aspect() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(800, 600, 800, 800, Padding::None, None);
        assert_eq!((ow, oh), (800, 800));
        assert_eq!((pl, pt, pr, pb), (0, 0, 0, 0));
    }

    #[test]
    fn letterbox_params_same_aspect_symmetric() {
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(1280, 720, 1920, 1080, Padding::Symmetric, None);
        assert_eq!((ow, oh), (1920, 1080));
        assert_eq!((pl, pt, pr, pb), (0, 0, 0, 0));
    }

    #[test]
    fn letterbox_params_with_dst_padding() {
        let dst_pad = DstPadding {
            left: 10,
            top: 20,
            right: 10,
            bottom: 20,
        };
        let (ow, oh, pl, pt, pr, pb) =
            compute_letterbox_params(800, 600, 820, 640, Padding::Symmetric, Some(dst_pad));
        assert_eq!((ow, oh), (820, 640));
        // Effective area 800x600, same aspect as source -> no letterbox bars
        assert_eq!(pl, 10);
        assert_eq!(pt, 20);
        assert_eq!(pr, 10);
        assert_eq!(pb, 20);
    }
}
