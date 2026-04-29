//! Convert Savant [`VideoFrame`](super::frame::VideoFrame) timestamps to **GStreamer clock scale**
//! (unsigned nanoseconds).

use super::frame::VideoFrame;
use log::warn;

/// GStreamer clock: one nanosecond per unit (`ClockTime` tick).
pub const GST_TIME_BASE: (i64, i64) = (1, 1_000_000_000);

/// Map `value` expressed in `time_base = num/den` seconds per unit to nanoseconds.
///
/// GStreamer [`ClockTime`](https://gstreamer.freedesktop.org/documentation/gstreamer/gstclock.html?gi-language=c)
/// uses an unsigned nanosecond timeline. For a **valid** positive rational time base, a negative `value`
/// yields a negative intermediate result and clamps to `0`; overflow clamps to `u64::MAX`.
///
/// Returns [`None`] when `num <= 0` or `den <= 0` (invalid time base; negative components must not be
/// silently folded into `0` nanoseconds).
pub fn time_base_to_ns(time_base: (i64, i64), value: i64) -> Option<u64> {
    let (num, den) = time_base;
    if num <= 0 || den <= 0 {
        return None;
    }
    let v = (value as i128)
        .saturating_mul(1_000_000_000)
        .saturating_mul(num as i128)
        / (den as i128);
    if v <= 0 {
        Some(0)
    } else if v > u64::MAX as i128 {
        Some(u64::MAX)
    } else {
        Some(v as u64)
    }
}

/// Timestamps in GStreamer **nanosecond** clock scale for one compressed packet.
#[derive(Debug, Clone, Copy)]
pub struct FrameClockNs {
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    /// Ordering key for decode submission: `DTS` when set, else presentation `PTS`.
    /// Must be strictly increasing for ordered hardware decode pipelines.
    pub submission_order_ns: u64,
}

/// Build GStreamer-scale timestamps from a frame proxy.
///
/// If the frame's time base is invalid (`num <= 0` or `den <= 0`), every field that depends on
/// rational conversion is set as if timestamps were `0` / unset — do not treat raw PTS/DTS/duration
/// integers as nanoseconds.
pub fn frame_clock_ns(frame: &VideoFrame) -> FrameClockNs {
    let tb = frame.get_time_base();
    let pts = frame.get_pts();
    let pts_ns = time_base_to_ns(tb, pts).unwrap_or(0);
    let dts_ns = frame.get_dts().and_then(|dts| time_base_to_ns(tb, dts));
    let duration_ns = frame.get_duration().and_then(|d| time_base_to_ns(tb, d));
    let submission_order_ns = dts_ns.unwrap_or(pts_ns);
    FrameClockNs {
        dts_ns,
        duration_ns,
        submission_order_ns,
    }
}

/// Clamp unsigned nanoseconds to a non-negative `i64` PTS/DTS field.
#[inline]
fn ns_to_i64(ns: u64) -> i64 {
    ns.min(i64::MAX as u64) as i64
}

/// Rewrite [`VideoFrame`] `pts` / `dts` / `duration` / `time_base` to GStreamer nanosecond scale.
///
/// If `time_base` is already [`GST_TIME_BASE`], returns immediately. If `time_base` is invalid
/// (`num <= 0` or `den <= 0`), logs a warning and leaves the frame unchanged.
pub fn normalize_frame_to_gst_ns(frame: &mut VideoFrame) {
    let (n, d) = frame.get_time_base();
    if n == GST_TIME_BASE.0 && d == GST_TIME_BASE.1 {
        return;
    }
    let tb_i64 = (n, d);
    if tb_i64.0 <= 0 || tb_i64.1 <= 0 {
        warn!("normalize_frame_to_gst_ns: invalid time_base ({n}, {d}), leaving frame unchanged");
        return;
    }
    let Some(pts_ns) = time_base_to_ns(tb_i64, frame.get_pts()) else {
        warn!(
            "normalize_frame_to_gst_ns: cannot convert PTS with time_base ({n}, {d}), leaving frame unchanged"
        );
        return;
    };
    if let Err(e) = frame.set_pts(ns_to_i64(pts_ns)) {
        warn!("normalize_frame_to_gst_ns: set_pts failed: {e}");
        return;
    }

    if let Some(dts) = frame.get_dts() {
        let dts_ns = time_base_to_ns(tb_i64, dts).unwrap_or(0);
        if let Err(e) = frame.set_dts(Some(ns_to_i64(dts_ns))) {
            warn!("normalize_frame_to_gst_ns: set_dts failed: {e}");
            return;
        }
    }

    if let Some(dur) = frame.get_duration() {
        let dur_ns = time_base_to_ns(tb_i64, dur).unwrap_or(0);
        if let Err(e) = frame.set_duration(Some(ns_to_i64(dur_ns))) {
            warn!("normalize_frame_to_gst_ns: set_duration failed: {e}");
            return;
        }
    }

    if let Err(e) = frame.set_time_base(GST_TIME_BASE) {
        warn!("normalize_frame_to_gst_ns: set_time_base failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::frame::{
        VideoFrame, VideoFrameContent, VideoFrameInner, VideoFrameTranscodingMethod,
    };
    use crate::primitives::video_codec::VideoCodec;

    fn test_frame(
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> VideoFrame {
        VideoFrame::new(
            "src",
            (30, 1),
            320,
            240,
            VideoFrameContent::None,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::Jpeg),
            None,
            time_base,
            pts,
            dts,
            duration,
        )
        .expect("test frame")
    }

    #[test]
    fn rational_1_90000_to_ns() {
        // 90000 Hz tick, 1 tick ≈ 11111 ns
        let ns = time_base_to_ns((1, 90_000), 1).unwrap();
        assert!((11_111..=11_112).contains(&ns));
    }

    #[test]
    fn invalid_tb_fallback_none() {
        assert!(time_base_to_ns((0, 1), 5).is_none());
        assert!(time_base_to_ns((1, 0), 5).is_none());
        assert!(time_base_to_ns((-1, 90_000), 1).is_none());
        assert!(time_base_to_ns((1, -90_000), 1).is_none());
    }

    #[test]
    fn invalid_time_base_no_raw_pts_as_ns() {
        // `VideoFrame::new` rejects non-positive time_base; malformed values can still
        // appear via `from_inner` / deserialization — `frame_clock_ns` must not treat PTS as ns.
        let f = VideoFrame::from_inner(VideoFrameInner {
            time_base: (-1, 90_000),
            pts: 90_000,
            dts: Some(90_000),
            duration: Some(1),
            ..Default::default()
        });
        let clk = frame_clock_ns(&f);
        assert_eq!(clk.submission_order_ns, 0);
        assert_eq!(clk.dts_ns, None);
        assert_eq!(clk.duration_ns, None);
    }

    #[test]
    fn normalize_noop_for_gst_timebase() {
        let mut f = test_frame((1, 1_000_000_000), 1_500, Some(1_400), Some(33_333));
        normalize_frame_to_gst_ns(&mut f);
        assert_eq!(f.get_time_base(), (1, 1_000_000_000));
        assert_eq!(f.get_pts(), 1_500);
        assert_eq!(f.get_dts(), Some(1_400));
        assert_eq!(f.get_duration(), Some(33_333));
    }

    #[test]
    fn normalize_90khz_to_ns() {
        let mut f = test_frame((1, 90_000), 90_000, None, None);
        normalize_frame_to_gst_ns(&mut f);
        assert_eq!(f.get_time_base(), GST_TIME_BASE);
        assert_eq!(f.get_pts(), 1_000_000_000);
        assert_eq!(f.get_dts(), None);
        assert_eq!(f.get_duration(), None);
    }

    #[test]
    fn normalize_millisecond_to_ns() {
        let mut f = test_frame((1, 1000), 1000, Some(1000), Some(33));
        normalize_frame_to_gst_ns(&mut f);
        assert_eq!(f.get_time_base(), GST_TIME_BASE);
        assert_eq!(f.get_pts(), 1_000_000_000);
        assert_eq!(f.get_dts(), Some(1_000_000_000));
        assert_eq!(f.get_duration(), Some(33_000_000));
    }

    #[test]
    fn normalize_preserves_none_dts_duration() {
        let mut f = test_frame((1, 90_000), 180_000, None, None);
        normalize_frame_to_gst_ns(&mut f);
        assert_eq!(f.get_pts(), 2_000_000_000);
        assert_eq!(f.get_dts(), None);
        assert_eq!(f.get_duration(), None);
    }

    #[test]
    fn normalize_huge_pts_clamps_via_time_base_to_ns() {
        // 1 tick == 1 second; PTS = i64::MAX → nanoseconds overflow `u64::MAX`, then clamp to `i64::MAX`.
        let mut f = test_frame((1, 1), i64::MAX, None, None);
        normalize_frame_to_gst_ns(&mut f);
        assert_eq!(f.get_time_base(), GST_TIME_BASE);
        assert_eq!(f.get_pts(), i64::MAX);
    }
}
