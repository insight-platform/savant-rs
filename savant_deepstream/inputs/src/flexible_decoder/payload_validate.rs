//! Pre-submission payload sanity checks.
//!
//! These functions perform lightweight structural validation **without**
//! decoding pixel data.  The goal is to catch corrupt / misidentified payloads
//! before they enter the GStreamer pipeline, where silent drops desynchronise
//! PTS-keyed tracking structures.

use jfifdump::{JfifError, Reader, SegmentKind};
use savant_core::primitives::video_codec::VideoCodec;
use std::io::Cursor;

/// 8-byte PNG file signature (RFC 2083 §3.1).
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Validate that `data` is structurally plausible for the given codec.
///
/// Returns `Ok(())` when the payload passes, or `Err(reason)` with a
/// human-readable explanation when it does not.
///
/// Only codecs with known container signatures are checked; all others
/// (H.264, HEVC, VP8, VP9, AV1, raw, …) pass unconditionally.
pub(crate) fn validate_payload(codec: VideoCodec, data: &[u8]) -> Result<(), String> {
    match codec {
        VideoCodec::Jpeg | VideoCodec::SwJpeg => validate_jpeg(data),
        VideoCodec::Png => validate_png(data),
        _ => Ok(()),
    }
}

/// Validate JPEG structure via [`jfifdump`].
///
/// Walks the JFIF segment chain looking for at least one SOF (Start of Frame)
/// marker.  Fails on truncated data, missing SOI, or broken segment lengths.
fn validate_jpeg(data: &[u8]) -> Result<(), String> {
    if data.len() < 2 {
        return Err("payload too short for JPEG (< 2 bytes)".into());
    }
    let cursor = Cursor::new(data);
    let mut reader = Reader::new(cursor).map_err(|e| format!("JPEG header rejected: {e}"))?;

    let mut found_sof = false;
    loop {
        match reader.next_segment() {
            Ok(seg) => match seg.kind {
                SegmentKind::Eoi => break,
                SegmentKind::Frame(_) => {
                    found_sof = true;
                    break;
                }
                _ => {}
            },
            Err(JfifError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err("JPEG truncated before SOF/EOI".into());
            }
            Err(e) => {
                return Err(format!("JPEG structure invalid: {e}"));
            }
        }
    }
    if !found_sof {
        return Err("JPEG contains no SOF (Start of Frame) marker".into());
    }
    Ok(())
}

/// Validate PNG: 8-byte signature + first chunk must be IHDR.
fn validate_png(data: &[u8]) -> Result<(), String> {
    if data.len() < 16 {
        return Err(format!(
            "payload too short for PNG ({} bytes, need >= 16)",
            data.len()
        ));
    }
    if data[..8] != PNG_SIGNATURE {
        return Err("PNG signature mismatch (first 8 bytes)".into());
    }
    // First chunk: bytes 8..12 = length (big-endian u32), bytes 12..16 = type.
    if &data[12..16] != b"IHDR" {
        return Err(format!(
            "first PNG chunk is {:?}, expected IHDR",
            std::str::from_utf8(&data[12..16]).unwrap_or("<non-utf8>")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_garbage_rejected() {
        let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        assert!(validate_jpeg(&garbage).is_err());
    }

    #[test]
    fn jpeg_too_short() {
        assert!(validate_jpeg(&[0xFF]).is_err());
        assert!(validate_jpeg(&[]).is_err());
    }

    #[test]
    fn jpeg_soi_only_no_sof() {
        // Valid SOI but no SOF marker — not a real image.
        let data = [0xFF, 0xD8, 0xFF, 0xD9]; // SOI + EOI
        assert!(validate_jpeg(&data).is_err());
    }

    #[test]
    fn png_garbage_rejected() {
        let garbage = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF, 0x42, 0x13, 0x00].repeat(3);
        assert!(validate_png(&garbage).is_err());
    }

    #[test]
    fn png_too_short() {
        assert!(validate_png(&[137, 80, 78, 71]).is_err());
    }

    #[test]
    fn png_valid_signature_wrong_chunk() {
        let mut data = PNG_SIGNATURE.to_vec();
        data.extend_from_slice(&[0, 0, 0, 13]); // length
        data.extend_from_slice(b"tEXt"); // wrong chunk type
        assert!(validate_png(&data).is_err());
    }

    #[test]
    fn png_valid_header() {
        let mut data = PNG_SIGNATURE.to_vec();
        data.extend_from_slice(&[0, 0, 0, 13]); // IHDR length = 13
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&[0; 13]); // IHDR body
        data.extend_from_slice(&[0; 4]); // CRC
        assert!(validate_png(&data).is_ok());
    }

    #[test]
    fn passthrough_for_unchecked_codecs() {
        let garbage = vec![0x00; 4];
        assert!(validate_payload(VideoCodec::H264, &garbage).is_ok());
        assert!(validate_payload(VideoCodec::Hevc, &garbage).is_ok());
        assert!(validate_payload(VideoCodec::Vp8, &garbage).is_ok());
        assert!(validate_payload(VideoCodec::Av1, &garbage).is_ok());
    }

    #[test]
    fn real_jpeg_accepted() {
        let img = image::RgbaImage::from_pixel(64, 64, image::Rgba([128, 64, 32, 255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
        let jpeg_bytes = buf.into_inner();
        assert!(validate_jpeg(&jpeg_bytes).is_ok());
        assert!(validate_payload(VideoCodec::Jpeg, &jpeg_bytes).is_ok());
        assert!(validate_payload(VideoCodec::SwJpeg, &jpeg_bytes).is_ok());
    }

    #[test]
    fn real_png_accepted() {
        let img = image::RgbaImage::from_pixel(64, 64, image::Rgba([64, 128, 255, 255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let png_bytes = buf.into_inner();
        assert!(validate_png(&png_bytes).is_ok());
        assert!(validate_payload(VideoCodec::Png, &png_bytes).is_ok());
    }
}
