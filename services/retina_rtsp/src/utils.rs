use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::bail;
use log::debug;

use crate::service::StreamInfo;
use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h264::parser::NaluType::AuDelimiter as H264AuDelimiter;
use cros_codecs::codec::h264::parser::NaluType::SliceIdr as H264Idr;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::codec::h265::parser::NaluType::AudNut as HEVCAuDelimiter;
use cros_codecs::codec::h265::parser::NaluType::CraNut as H265IdrCraNut;
use cros_codecs::codec::h265::parser::NaluType::IdrNLp as H265IdrNLp;
use cros_codecs::codec::h265::parser::NaluType::IdrWRadl as H265IdrWRadl;

pub const ONE_NS: f64 = 1_000_000_000.0;

/// NTP epoch (1900-01-01) to UNIX epoch (1970-01-01) offset in NTP 32.32 fixed-point format.
const NTP_UNIX_EPOCH_OFFSET: u64 = 0x83AA7E80_00000000;

pub const H264_AU_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x09, 0xF0];
pub const HEVC_AU_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x23, 0xF0];

/// Cooperatively wait until `shutdown` is set (shared by GStreamer and Retina `run_group` paths).
pub(crate) async fn wait_for_shutdown_flag(shutdown: &Arc<AtomicBool>) {
    const POLL_MS: u64 = 50;
    while !shutdown.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(POLL_MS)).await;
    }
}

pub fn ts2epoch_duration(ntp_raw: u64, skew_millis: i64) -> Duration {
    let since_epoch = ntp_raw.wrapping_sub(NTP_UNIX_EPOCH_OFFSET);
    let sec_since_epoch = (since_epoch >> 32) as u32;
    let ns = u32::try_from(((since_epoch & 0xFFFF_FFFF) * 1_000_000_000) >> 32)
        .expect("should be < 1_000_000_000");
    if skew_millis > 0 {
        Duration::new(sec_since_epoch as u64, ns)
            + Duration::from_millis(skew_millis.unsigned_abs())
    } else {
        Duration::new(sec_since_epoch as u64, ns)
            - Duration::from_millis(skew_millis.unsigned_abs())
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

/// Check whether the first NALU is an AU delimiter and prepend one if missing.
/// Works with Annex B byte-stream data. `encoding` must be `"h264"` or `"hevc"`.
pub fn ensure_au_delimiter(data: Vec<u8>, encoding: &str) -> Vec<u8> {
    let has_aud = has_au_delimiter_nalu(&data, encoding);
    if has_aud {
        return data;
    }
    let delimiter: &[u8] = match encoding {
        "h264" => &H264_AU_DELIMITER,
        "hevc" => &HEVC_AU_DELIMITER,
        _ => return data,
    };
    let mut out = Vec::with_capacity(delimiter.len() + data.len());
    out.extend_from_slice(delimiter);
    out.extend_from_slice(&data);
    out
}

fn has_au_delimiter_nalu(frame_data: &[u8], encoding: &str) -> bool {
    let mut cursor = Cursor::new(frame_data);
    match encoding {
        "h264" => {
            if let Ok(nal) = H264Nalu::next(&mut cursor) {
                return matches!(nal.header.type_, H264AuDelimiter);
            }
        }
        "hevc" => {
            if let Ok(nal) = H265Nalu::next(&mut cursor) {
                return matches!(nal.header.type_, HEVCAuDelimiter);
            }
        }
        _ => {}
    }
    false
}

pub fn is_keyframe(
    frame_data: &[u8],
    source_id: &str,
    rtp_time: i64,
    stream_info: &StreamInfo,
) -> bool {
    let mut kf = false;
    let mut cursor = Cursor::new(frame_data);
    debug!(
        target: "retina_rtsp::service::parser",
        "Stream_id: {}, RTP time: {}, Encoding: {}",
        source_id, rtp_time, stream_info.encoding
    );
    if matches!(stream_info.encoding.as_str(), "h264") {
        while let Ok(nal) = H264Nalu::next(&mut cursor) {
            debug!(
                target: "retina_rtsp::service::parser",
                "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                source_id, rtp_time, nal.header, nal.offset
            );
            if matches!(nal.header.type_, H264Idr) {
                kf = true;
            }
        }
    } else if matches!(stream_info.encoding.as_str(), "hevc") {
        while let Ok(nal) = H265Nalu::next(&mut cursor) {
            debug!(
                target: "retina_rtsp::service::parser",
                "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                source_id, rtp_time, nal.header, nal.offset
            );
            if matches!(nal.header.type_, H265IdrWRadl | H265IdrNLp | H265IdrCraNut) {
                kf = true;
            }
        }
    } else {
        // MJPEG frames are always keyframes
        kf = true;
    }
    kf
}
