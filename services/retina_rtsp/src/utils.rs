use std::io::Cursor;
use std::time::Duration;

use anyhow::bail;
use log::debug;
use retina::NtpTimestamp;

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

pub const H264_AU_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x09, 0xF0];
pub const HEVC_AU_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x23, 0xF0];

pub fn ts2epoch_duration(ts: NtpTimestamp, skew_millis: i64) -> Duration {
    let since_epoch = ts.0.wrapping_sub(retina::UNIX_EPOCH.0);
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

pub fn check_contains_au_delimiter(
    frame_data: &[u8],
    source_id: &str,
    rtp_time: i64,
    stream_info: &StreamInfo,
) -> bool {
    let mut cursor = Cursor::new(frame_data);
    debug!(
        target: "retina_rtsp::service::parser",
        "Stream_id: {}, RTP time: {}, Encoding: {}",
        source_id, rtp_time, stream_info.encoding
    );
    let mut aud = false;
    if matches!(stream_info.encoding.as_str(), "h264") {
        if let Ok(nal) = H264Nalu::next(&mut cursor) {
            debug!(
                target: "retina_rtsp::service::parser::check_contains_au_delimiter",
                "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                source_id, rtp_time, nal.header, nal.offset
            );
            if matches!(nal.header.type_, H264AuDelimiter) {
                aud = true;
            }
        }
    } else if matches!(stream_info.encoding.as_str(), "hevc") {
        if let Ok(nal) = H265Nalu::next(&mut cursor) {
            debug!(
                target: "retina_rtsp::service::parser::check_contains_au_delimiter",
                "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                source_id, rtp_time, nal.header, nal.offset
            );
            if matches!(nal.header.type_, HEVCAuDelimiter) {
                aud = true;
            }
        }
    }
    debug!(
        target: "retina_rtsp::service::parser::check_contains_au_delimiter",
        "Stream_id: {}, RTP time: {}, AU delimiter: {}",
        source_id, rtp_time, aud
    );
    aud
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
