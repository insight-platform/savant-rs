use std::time::Duration;

use anyhow::bail;
use retina::NtpTimestamp;

pub const ONE_NS: f64 = 1_000_000_000.0;

pub fn ts2epoch_duration(ts: NtpTimestamp, skew_millis: i64) -> Duration {
    let since_epoch = ts.0.wrapping_sub(retina::UNIX_EPOCH.0);
    let sec_since_epoch = (since_epoch >> 32) as u32;
    let ns = u32::try_from(((since_epoch & 0xFFFF_FFFF) * 1_000_000_000) >> 32)
        .expect("should be < 1_000_000_000");
    if skew_millis > 0 {
        Duration::new(sec_since_epoch as u64, ns as u32)
            + Duration::from_millis(skew_millis.abs() as u64)
    } else {
        Duration::new(sec_since_epoch as u64, ns as u32)
            - Duration::from_millis(skew_millis.abs() as u64)
    }
}

pub fn convert_to_annexb(frame: retina::codec::VideoFrame) -> anyhow::Result<Vec<u8>> {
    let mut data = frame.into_data();
    let mut i = 0;
    while i < data.len() - 3 {
        // Replace each NAL's length with the Annex B start code b"\x00\x00\x00\x01".
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 1;
        i += 4 + len;
        if i > data.len() {
            bail!("partial NAL body");
        }
    }
    if i < data.len() {
        bail!("partial NAL length");
    }
    Ok(data)
}
