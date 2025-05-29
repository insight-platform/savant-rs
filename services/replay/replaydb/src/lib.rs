use savant_core::primitives::frame::VideoFrameProxy;
use uuid::{NoContext, Timestamp, Uuid};

pub mod job;
pub mod service;
pub mod store;
pub mod stream_processor;

pub type ParkingLotMutex<T> = parking_lot::Mutex<T>;

pub fn get_keyframe_boundary(v: Option<u64>, default: u64) -> Uuid {
    let ts = v.unwrap_or(default);
    Uuid::new_v7(Timestamp::from_unix(NoContext, ts, 0))
}

pub(crate) fn best_ts(f: &VideoFrameProxy) -> i64 {
    let dts_opt = f.get_dts();
    if let Some(dts) = dts_opt {
        dts
    } else {
        f.get_pts()
    }
}
