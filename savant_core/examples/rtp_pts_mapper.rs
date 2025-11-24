use std::error::Error;
use std::thread::sleep;
use std::time::{Duration, UNIX_EPOCH};

use savant_core::utils::rtp_pts_mapper::RtpPtsMapper;

const VIDEO_CLOCK_RATE: i64 = 900_000_000;
const TARGET_FPS: u32 = 1;
const SEED_RTP: u32 = u32::MAX - 1_000_000;
const SEED_TS: Duration = Duration::from_secs(0);

fn main() -> Result<(), Box<dyn Error>> {
    let mut mapper = RtpPtsMapper::with_seed(
        SEED_RTP,
        SEED_TS,
        (1, VIDEO_CLOCK_RATE),
        (1, 1_000_000), // 1us ticks for PTS
    )?;

    let frame_interval_ticks = (VIDEO_CLOCK_RATE as u32) / TARGET_FPS;
    let frame_interval_sleep = Duration::from_micros(1_000_000 / TARGET_FPS as u64);

    let mut rtp = SEED_RTP;
    let mut last_pts: Option<i64> = None;
    let mut last_ts: Option<Duration> = None;
    loop {
        rtp = rtp.wrapping_add(frame_interval_ticks);
        let mapping = mapper.map(rtp)?;

        let wall_clock = UNIX_EPOCH + mapping.ts;
        let pts_delta = last_pts.map(|p| mapping.pts - p).unwrap_or(0);
        let ts_delta = last_ts
            .map(|ts| mapping.ts.checked_sub(ts).unwrap_or_default())
            .unwrap_or_default();

        println!(
            "RTP {:>12} -> PTS {:>12} (ΔPTS = {:>6}, Δt = {:?}, ts = {:?})",
            rtp, mapping.pts, pts_delta, ts_delta, wall_clock
        );

        last_pts = Some(mapping.pts);
        last_ts = Some(mapping.ts);

        sleep(frame_interval_sleep);
    }
}
