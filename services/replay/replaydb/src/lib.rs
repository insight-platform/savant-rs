use savant_core::primitives::frame::VideoFrame;
use savant_core::utils::video_id::VideoId;

pub mod job;
pub mod service;
pub mod store;
pub mod stream_processor;

pub type ParkingLotMutex<T> = parking_lot::Mutex<T>;

const NS_PER_SEC: u64 = 1_000_000_000;

/// Inclusive lower bound of a keyframe range scan at `ts_secs`.
/// Returns a u128 in the video_id encoding suitable for direct
/// comparison with stored frame uuids. Source identity is carried by
/// the surrounding `(source_md5, keyframe_uuid)` storage key, not by
/// the boundary itself.
pub fn keyframe_lower_bound(ts_secs: u64) -> u128 {
    VideoId::lower_bound(ts_secs.saturating_mul(NS_PER_SEC)).as_u128()
}

/// Inclusive upper bound of a keyframe range scan at `ts_secs`.
/// Saturates at the last nanosecond of the second so a query at
/// second N includes every keyframe whose `ts_ns` falls in
/// `[N * 1e9, N * 1e9 + 999_999_999]`.
pub fn keyframe_upper_bound(ts_secs: u64) -> u128 {
    let end_of_second_ns = ts_secs
        .saturating_mul(NS_PER_SEC)
        .saturating_add(NS_PER_SEC - 1);
    VideoId::upper_bound(end_of_second_ns).as_u128()
}

pub(crate) fn best_ts(f: &VideoFrame) -> i64 {
    let dts_opt = f.get_dts();
    if let Some(dts) = dts_opt {
        dts
    } else {
        f.get_pts()
    }
}
