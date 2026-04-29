//! Detect H.264 / HEVC stream packaging from one access unit and build [`DecoderConfig`].
//!
//! Annex-B (byte-stream) is detected when the buffer starts with a 4-byte start code
//! (`00 00 00 01`) or a 3-byte start code (`00 00 01`) that is **not** a false positive.
//! The prefix `00 00 01 XX` is ambiguous: it is either a 3-byte start code (NAL begins at
//! `XX`) or the first four bytes of a 4-byte big-endian length in **256..=511** (NAL begins
//! after the length). In the ambiguous case we prefer AVCC/HVCC only when the buffer fully
//! splits as length-prefixed NALs and the first NAL parses for the codec.
//! Otherwise the buffer is treated as 4-byte big-endian length–prefixed NAL units (AVCC/HVCC).
//!
//! [`is_random_access_point`] checks whether a single access unit is a valid decode entry point
//! (parameter sets + IRAP/IDR for HEVC/H264).

use std::io::Cursor;

use cros_codecs::codec::h264::nalu::Header;
use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h264::parser::NaluType as H264NaluType;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::codec::h265::parser::NaluType as H265NaluType;
use cros_codecs::codec::h265::parser::Parser as H265Parser;

use crate::config::{
    DecoderConfig, H264DecoderConfig, H264StreamFormat, HevcDecoderConfig, HevcStreamFormat,
};
use savant_core::primitives::video_codec::VideoCodec;

type H264ParamSets = (Vec<Vec<u8>>, Vec<Vec<u8>>);
type HevcParamSets = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Inspect one access unit (packet) to determine the H264/HEVC stream format.
///
/// Returns `Some(DecoderConfig)` with the correct stream format and, for length-prefixed
/// formats (AVCC / `hvc1`), the constructed `codec_data`. Returns [`None`] when:
/// - `codec` is not [`VideoCodec::H264`] or [`VideoCodec::Hevc`]
/// - the prefix is neither Annex-B nor a valid 4-byte length-prefixed layout
/// - length-prefixed H.264 without both SPS and PPS
/// - length-prefixed HEVC without VPS, SPS, and PPS, or SPS parsing fails
///
/// # Examples
///
/// Annex-B H.264 (start code at the beginning):
///
/// ```no_run
/// use deepstream_decoders::{detect_stream_config, DecoderConfig, H264StreamFormat};
/// use savant_core::primitives::video_codec::VideoCodec;
///
/// let au: &[u8] = &[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x0A]; // SPS-like NAL after SC
/// let cfg = detect_stream_config(VideoCodec::H264, au).expect("annex-b");
/// match cfg {
///     DecoderConfig::H264(c) => assert_eq!(c.stream_format, H264StreamFormat::ByteStream),
///     _ => panic!("expected H264"),
/// }
/// ```
///
/// Unsupported codec:
///
/// ```
/// use deepstream_decoders::detect_stream_config;
/// use savant_core::primitives::video_codec::VideoCodec;
///
/// assert!(detect_stream_config(VideoCodec::Vp9, &[0, 0, 0, 1, 0x09]).is_none());
/// ```
///
/// Length-prefixed H.264 with SPS and PPS (4-byte big-endian lengths):
///
/// ```no_run
/// use deepstream_decoders::{detect_stream_config, DecoderConfig, H264StreamFormat};
/// use savant_core::primitives::video_codec::VideoCodec;
///
/// let sps = [0x67u8, 0x42, 0x00, 0x0A, 0xFF];
/// let pps = [0x68u8, 0xCE, 0x3C, 0x80];
/// let mut au = Vec::new();
/// for nal in [&sps[..], &pps[..]] {
///     au.extend_from_slice(&(nal.len() as u32).to_be_bytes());
///     au.extend_from_slice(nal);
/// }
/// let cfg = detect_stream_config(VideoCodec::H264, &au).unwrap();
/// match cfg {
///     DecoderConfig::H264(c) => {
///         assert_eq!(c.stream_format, H264StreamFormat::Avc);
///         assert!(c.codec_data.is_some());
///     }
///     _ => panic!("expected H264"),
/// }
/// ```
pub fn detect_stream_config(codec: VideoCodec, data: &[u8]) -> Option<DecoderConfig> {
    match codec {
        VideoCodec::H264 => detect_h264(data),
        VideoCodec::Hevc => detect_hevc(data),
        _ => None,
    }
}

/// Returns `true` when the access unit is a valid random access point from which a decoder
/// can start producing pictures.
///
/// Framing matches [`detect_stream_config`]: Annex-B if `data` starts with an unambiguous
/// start code or an ambiguous `00 00 01` prefix that does not resolve as a valid
/// length-prefixed AU; otherwise 4-byte big-endian length–prefixed NAL units.
///
/// - **H.264**: requires at least one SPS, one PPS, and one IDR slice (NAL type 5) in the same
///   access unit.
/// - **HEVC**: requires at least one VPS, SPS, PPS, and one IRAP VCL NAL (IDR, CRA, BLA, etc.;
///   matches `cros_codecs` `NaluType::is_irap()`).
/// - **JPEG, PNG, RawRgba, RawRgb, RawNv12**: `true` for non-empty data (every frame is
///   independently decodable). Empty buffer returns `false`.
/// - **VP8, VP9, AV1**: always `false` (keyframe detection not implemented).
///
/// # Examples
///
/// H.264 Annex-B access unit with SPS, PPS, and IDR:
///
/// ```no_run
/// use deepstream_decoders::is_random_access_point;
/// use savant_core::primitives::video_codec::VideoCodec;
///
/// let sps = [0x67u8, 0x42, 0x00, 0x0A, 0xFF];
/// let pps = [0x68u8, 0xCE, 0x3C, 0x80];
/// let idr = [0x65u8, 0x88, 0x84, 0x00];
/// let mut au = Vec::new();
/// for nal in [&sps[..], &pps[..], &idr[..]] {
///     au.extend_from_slice(&[0, 0, 0, 1]);
///     au.extend_from_slice(nal);
/// }
/// assert!(is_random_access_point(VideoCodec::H264, &au));
/// ```
///
/// Unsupported video codecs:
///
/// ```
/// use deepstream_decoders::is_random_access_point;
/// use savant_core::primitives::video_codec::VideoCodec;
///
/// assert!(!is_random_access_point(VideoCodec::Vp9, &[0, 0, 0, 1, 0x9A]));
/// ```
pub fn is_random_access_point(codec: VideoCodec, data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    match codec {
        VideoCodec::H264 => is_h264_rap(data),
        VideoCodec::Hevc => is_hevc_rap(data),
        VideoCodec::Jpeg
        | VideoCodec::SwJpeg
        | VideoCodec::Png
        | VideoCodec::RawRgba
        | VideoCodec::RawRgb
        | VideoCodec::RawNv12 => true,
        VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Av1 => false,
    }
}

fn is_h264_rap(data: &[u8]) -> bool {
    if has_annexb_prefix(data, true) {
        scan_h264_annexb_rap(data)
    } else {
        scan_h264_length_prefixed_rap(data)
    }
}

fn scan_h264_annexb_rap(data: &[u8]) -> bool {
    let mut cur = Cursor::new(data);
    let mut has_sps = false;
    let mut has_pps = false;
    let mut has_idr = false;
    while let Ok(nalu) = H264Nalu::next(&mut cur) {
        match nalu.header.type_ {
            H264NaluType::Sps => has_sps = true,
            H264NaluType::Pps => has_pps = true,
            H264NaluType::SliceIdr => has_idr = true,
            _ => {}
        }
    }
    has_sps && has_pps && has_idr
}

fn scan_h264_length_prefixed_rap(data: &[u8]) -> bool {
    let Some(nals) = split_length_prefixed_nalus(data) else {
        return false;
    };
    let mut has_sps = false;
    let mut has_pps = false;
    let mut has_idr = false;
    for nal in nals {
        let wrapped = prepend_start_code(nal);
        let mut cur = Cursor::new(wrapped.as_slice());
        let Ok(nalu) = H264Nalu::next(&mut cur) else {
            return false;
        };
        match nalu.header.type_ {
            H264NaluType::Sps => has_sps = true,
            H264NaluType::Pps => has_pps = true,
            H264NaluType::SliceIdr => has_idr = true,
            _ => {}
        }
    }
    has_sps && has_pps && has_idr
}

fn is_hevc_rap(data: &[u8]) -> bool {
    if has_annexb_prefix(data, false) {
        scan_hevc_annexb_rap(data)
    } else {
        scan_hevc_length_prefixed_rap(data)
    }
}

fn scan_hevc_annexb_rap(data: &[u8]) -> bool {
    let mut cur = Cursor::new(data);
    let mut has_vps = false;
    let mut has_sps = false;
    let mut has_pps = false;
    let mut has_irap = false;
    while let Ok(nalu) = H265Nalu::next(&mut cur) {
        match nalu.header.type_ {
            H265NaluType::VpsNut => has_vps = true,
            H265NaluType::SpsNut => has_sps = true,
            H265NaluType::PpsNut => has_pps = true,
            t if t.is_irap() => has_irap = true,
            _ => {}
        }
    }
    has_vps && has_sps && has_pps && has_irap
}

fn scan_hevc_length_prefixed_rap(data: &[u8]) -> bool {
    let Some(nals) = split_length_prefixed_nalus(data) else {
        return false;
    };
    let mut has_vps = false;
    let mut has_sps = false;
    let mut has_pps = false;
    let mut has_irap = false;
    for nal in nals {
        let wrapped = prepend_start_code(nal);
        let mut cur = Cursor::new(wrapped.as_slice());
        let Ok(nalu) = H265Nalu::next(&mut cur) else {
            return false;
        };
        match nalu.header.type_ {
            H265NaluType::VpsNut => has_vps = true,
            H265NaluType::SpsNut => has_sps = true,
            H265NaluType::PpsNut => has_pps = true,
            t if t.is_irap() => has_irap = true,
            _ => {}
        }
    }
    has_vps && has_sps && has_pps && has_irap
}

fn detect_h264(data: &[u8]) -> Option<DecoderConfig> {
    if data.is_empty() {
        return None;
    }
    if has_annexb_prefix(data, true) {
        return Some(DecoderConfig::H264(H264DecoderConfig::new(
            H264StreamFormat::ByteStream,
        )));
    }
    let (sps, pps) = collect_h264_param_nalus_length_prefixed(data)?;
    let codec_data = build_avcc_record(&sps, &pps)?;
    Some(DecoderConfig::H264(
        H264DecoderConfig::new(H264StreamFormat::Avc).codec_data(codec_data),
    ))
}

fn detect_hevc(data: &[u8]) -> Option<DecoderConfig> {
    if data.is_empty() {
        return None;
    }
    if has_annexb_prefix(data, false) {
        return Some(DecoderConfig::Hevc(HevcDecoderConfig::new(
            HevcStreamFormat::ByteStream,
        )));
    }
    let (vps, sps, pps) = collect_hevc_param_nalus_length_prefixed(data)?;
    let codec_data = build_hvcc_record(&vps, &sps, &pps)?;
    Some(DecoderConfig::Hevc(
        HevcDecoderConfig::new(HevcStreamFormat::Hvc1).codec_data(codec_data),
    ))
}

/// True if `data` begins with `00 00 00 01`, or with `00 00 01` as a 3-byte Annex-B start code
/// (not as the first four bytes of a valid length-prefixed AU for this codec).
fn has_annexb_prefix(data: &[u8], is_h264: bool) -> bool {
    if data.len() >= 4 && data[..4] == [0, 0, 0, 1] {
        return true;
    }
    if data.len() < 3 || data[..3] != [0, 0, 1] {
        return false;
    }
    // `00 00 01` + next byte: either 3-byte start code or AVCC/HVCC length in 256..=511.
    if data.len() >= 4 {
        let first_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() >= 4 + first_len && length_prefixed_au_first_nal_parses(data, is_h264) {
            return false;
        }
    }
    true
}

/// `data` fully splits as 4-byte length–prefixed NALs and the first NAL decodes for `is_h264`.
fn length_prefixed_au_first_nal_parses(data: &[u8], is_h264: bool) -> bool {
    let Some(nals) = split_length_prefixed_nalus(data) else {
        return false;
    };
    let Some(first) = nals.first().copied() else {
        return false;
    };
    let wrapped = prepend_start_code(first);
    let mut cur = Cursor::new(wrapped.as_slice());
    if is_h264 {
        H264Nalu::next(&mut cur).is_ok()
    } else {
        H265Nalu::next(&mut cur).is_ok()
    }
}

/// Split `data` into NAL units using 4-byte big-endian length prefixes; returns [`None`] if truncated or malformed.
fn split_length_prefixed_nalus(data: &[u8]) -> Option<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + 4 > data.len() {
            return None;
        }
        let len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + len > data.len() {
            return None;
        }
        out.push(&data[pos..pos + len]);
        pos += len;
    }
    Some(out)
}

fn collect_h264_param_nalus_length_prefixed(data: &[u8]) -> Option<H264ParamSets> {
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    let nals = split_length_prefixed_nalus(data)?;
    for nal in nals {
        let wrapped = prepend_start_code(nal);
        let mut cur = Cursor::new(wrapped.as_slice());
        let nalu = H264Nalu::next(&mut cur).ok()?;
        match nalu.header.type_ {
            H264NaluType::Sps => sps.push(nalu.as_ref().to_vec()),
            H264NaluType::Pps => pps.push(nalu.as_ref().to_vec()),
            _ => {}
        }
    }
    if sps.is_empty() || pps.is_empty() {
        return None;
    }
    Some((sps, pps))
}

fn collect_hevc_param_nalus_length_prefixed(data: &[u8]) -> Option<HevcParamSets> {
    let mut vps = Vec::new();
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    let nals = split_length_prefixed_nalus(data)?;
    for nal in nals {
        let wrapped = prepend_start_code(nal);
        let mut cur = Cursor::new(wrapped.as_slice());
        let nalu = H265Nalu::next(&mut cur).ok()?;
        match nalu.header.type_ {
            H265NaluType::VpsNut => vps.push(nalu.as_ref().to_vec()),
            H265NaluType::SpsNut => sps.push(nalu.as_ref().to_vec()),
            H265NaluType::PpsNut => pps.push(nalu.as_ref().to_vec()),
            _ => {}
        }
    }
    if vps.is_empty() || sps.is_empty() || pps.is_empty() {
        return None;
    }
    Some((vps, sps, pps))
}

fn prepend_start_code(nal: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + nal.len());
    v.extend_from_slice(&[0, 0, 0, 1]);
    v.extend_from_slice(nal);
    v
}

/// ISO/IEC 14496-15 AVCDecoderConfigurationRecord.
fn build_avcc_record(sps_list: &[Vec<u8>], pps_list: &[Vec<u8>]) -> Option<Vec<u8>> {
    if sps_list.is_empty() || pps_list.is_empty() {
        return None;
    }
    let first = sps_list.first()?;
    if first.len() < 4 {
        return None;
    }
    let profile = first[1];
    let compat = first[2];
    let level = first[3];

    let mut out = vec![1, profile, compat, level, 0xFF];
    // 3 bits reserved (1) + numOfSequenceParameterSets (5 bits)
    let numsps = sps_list.len().min(31);
    out.push(0xE0u8 | numsps as u8);
    for sps in sps_list.iter().take(numsps) {
        let len = u16::try_from(sps.len()).ok()?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(sps);
    }
    let numpps = pps_list.len().min(255);
    out.push(numpps as u8);
    for pps in pps_list.iter().take(numpps) {
        let len = u16::try_from(pps.len()).ok()?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(pps);
    }
    Some(out)
}

/// ISO/IEC 14496-15 HEVCDecoderConfigurationRecord (version 1).
fn build_hvcc_record(
    vps_list: &[Vec<u8>],
    sps_list: &[Vec<u8>],
    pps_list: &[Vec<u8>],
) -> Option<Vec<u8>> {
    let first_sps = sps_list.first()?;
    let wrapped = prepend_start_code(first_sps);
    let mut cur = Cursor::new(wrapped.as_slice());
    let sps_nalu = H265Nalu::next(&mut cur).ok()?;
    if !matches!(sps_nalu.header.type_, H265NaluType::SpsNut) {
        return None;
    }
    let mut h265p = H265Parser::default();
    let sps_parsed = h265p.parse_sps(&sps_nalu).ok()?;

    let sps_bytes = sps_nalu.as_ref();
    if sps_bytes.len() < 3 {
        return None;
    }
    let hdr_len = sps_nalu.header.len();
    let rbsp = &sps_bytes[hdr_len..];
    let rbsp = strip_emulation_prevention(rbsp);
    let general = read_hvcc_general_from_sps_rbsp(&rbsp)?;

    let mut out = Vec::new();
    out.push(1u8); // configurationVersion
    out.extend_from_slice(&general); // 12 bytes: profile through general_level_idc
                                     // min_spatial_segmentation_idc: 4 reserved 1s + 12 bits 0
    out.extend_from_slice(&[0xF0, 0x00]);
    // parallelismType: 6 reserved 1s + 2 bits 0
    out.push(0xFC);
    let chroma = sps_parsed.chroma_format_idc.min(3);
    out.push(0xFCu8 | chroma);
    let luma = sps_parsed.bit_depth_luma_minus8.min(7);
    let chroma_bd = sps_parsed.bit_depth_chroma_minus8.min(7);
    out.push(0xF8u8 | luma);
    out.push(0xF8u8 | chroma_bd);
    // avgFrameRate
    out.extend_from_slice(&[0, 0]);
    // constantFrameRate=0, numTemporalLayers=1, temporalIdNested=1, lengthSizeMinusOne=3
    out.push(0x0F);
    let num_arrays = 3u8;
    out.push(num_arrays);

    push_hvcc_array(&mut out, 32, vps_list)?;
    push_hvcc_array(&mut out, 33, sps_list)?;
    push_hvcc_array(&mut out, 34, pps_list)?;
    Some(out)
}

fn push_hvcc_array(out: &mut Vec<u8>, nal_type: u8, nals: &[Vec<u8>]) -> Option<()> {
    if nals.is_empty() {
        return None;
    }
    // array_completeness (1) + reserved (1) + NAL_unit_type (6)
    out.push(0x80u8 | (nal_type & 0x3F));
    let n = nals.len().min(u16::MAX as usize) as u16;
    out.extend_from_slice(&n.to_be_bytes());
    for nal in nals.iter().take(n as usize) {
        let len = u16::try_from(nal.len()).ok()?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nal);
    }
    Some(())
}

/// First 12 bytes of hvcC after `configurationVersion`: general profile through `general_level_idc`.
fn read_hvcc_general_from_sps_rbsp(rbsp: &[u8]) -> Option<[u8; 12]> {
    let mut bc = BitCursor::new(rbsp);
    bc.skip(4)?; // video_parameter_set_id
    bc.skip(3)?; // max_sub_layers_minus1
    bc.skip(1)?; // temporal_id_nesting_flag
    let profile_space = bc.read_bits(2)?;
    let tier = bc.read_bits(1)?;
    let profile_idc = bc.read_bits(5)?;
    let mut compat = 0u32;
    for j in 0..32 {
        let b = bc.read_bits(1)?;
        compat |= b << (31 - j);
    }
    let mut constraints = 0u64;
    for _ in 0..48 {
        let b = bc.read_bits(1)?;
        constraints = (constraints << 1) | u64::from(b);
    }
    let level = bc.read_bits(8)? as u8;

    let mut out = [0u8; 12];
    out[0] =
        ((profile_space & 3) << 6) as u8 | ((tier & 1) << 5) as u8 | (profile_idc as u8 & 0x1F);
    out[1..5].copy_from_slice(&compat.to_be_bytes());
    out[5..11].copy_from_slice(&constraints.to_be_bytes()[2..8]); // high 48 bits of u64
    out[11] = level;
    Some(out)
}

fn strip_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

struct BitCursor<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.bit
    }

    fn skip(&mut self, n: u32) -> Option<()> {
        self.read_bits(n)?;
        Some(())
    }

    fn read_bits(&mut self, n: u32) -> Option<u32> {
        if n > 32 || n == 0 {
            return None;
        }
        if self.bits_left() < n as usize {
            return None;
        }
        let mut v = 0u32;
        for _ in 0..n {
            let byte_idx = self.bit / 8;
            let bit_idx = 7 - (self.bit % 8);
            let b = (self.data[byte_idx] >> bit_idx) & 1;
            v = (v << 1) | u32::from(b);
            self.bit += 1;
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_h264_sps() -> Vec<u8> {
        // 1-byte NAL header (type 7 SPS) + minimal RBSP: profile, constraint, level, seq_parameter_set_id ue(0)
        vec![
            0x67, 0x42, 0x00, 0x0A, 0xFF, 0xE1, 0x00, 0x27, 0x9A, 0x88, 0x80,
        ]
    }

    fn minimal_h264_pps() -> Vec<u8> {
        vec![0x68, 0xCE, 0x3C, 0x80]
    }

    fn prepend_sc(nal: &[u8]) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1];
        v.extend_from_slice(nal);
        v
    }

    fn len_prefixed(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            let l = n.len() as u32;
            v.extend_from_slice(&l.to_be_bytes());
            v.extend_from_slice(n);
        }
        v
    }

    #[test]
    fn test_annexb_h264_4byte_sc() {
        let au = prepend_sc(&minimal_h264_sps());
        let c = detect_stream_config(VideoCodec::H264, &au).unwrap();
        match c {
            DecoderConfig::H264(cfg) => {
                assert_eq!(cfg.stream_format, H264StreamFormat::ByteStream);
                assert!(cfg.codec_data.is_none());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_annexb_h264_3byte_sc() {
        let mut au = vec![0, 0, 1];
        au.extend_from_slice(&minimal_h264_sps());
        let c = detect_stream_config(VideoCodec::H264, &au).unwrap();
        match c {
            DecoderConfig::H264(cfg) => assert_eq!(cfg.stream_format, H264StreamFormat::ByteStream),
            _ => panic!(),
        }
    }

    #[test]
    fn test_annexb_h264_idr_only() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = prepend_sc(&idr);
        let c = detect_stream_config(VideoCodec::H264, &au).unwrap();
        match c {
            DecoderConfig::H264(cfg) => assert_eq!(cfg.stream_format, H264StreamFormat::ByteStream),
            _ => panic!(),
        }
    }

    /// H.264 filler NAL (type 12): 1-byte header + 255 `0xFF` RBSP bytes → 256-byte NAL.
    /// Prefix `00 00 01 00` is length **256** in big-endian and must not be mistaken for a
    /// 3-byte Annex-B start code.
    fn h264_filler_nalu_256() -> Vec<u8> {
        let mut nal = vec![0x0Cu8];
        nal.extend(std::iter::repeat_n(0xFFu8, 255));
        assert_eq!(nal.len(), 256);
        nal
    }

    #[test]
    fn test_avcc_h264_first_nal_len_256_detects_avc() {
        let au = len_prefixed(&[
            h264_filler_nalu_256().as_slice(),
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
        ]);
        let c = detect_stream_config(VideoCodec::H264, &au).expect("avc");
        match c {
            DecoderConfig::H264(cfg) => assert_eq!(cfg.stream_format, H264StreamFormat::Avc),
            _ => panic!(),
        }
    }

    #[test]
    fn test_avcc_h264_first_nal_len_256_rap_length_prefixed_path() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = len_prefixed(&[
            h264_filler_nalu_256().as_slice(),
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
            idr.as_slice(),
        ]);
        assert!(is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_avcc_h264_sps_pps_idr() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = len_prefixed(&[
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
            idr.as_slice(),
        ]);
        let c = detect_stream_config(VideoCodec::H264, &au).unwrap();
        match c {
            DecoderConfig::H264(cfg) => {
                assert_eq!(cfg.stream_format, H264StreamFormat::Avc);
                let cd = cfg.codec_data.expect("codec_data");
                assert_eq!(cd[0], 1);
                assert_eq!(cd[1], 0x42);
                assert_eq!(cd[2], 0x00);
                assert_eq!(cd[3], 0x0A);
                assert_eq!(cd[4], 0xFF);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_avcc_h264_sps_only_no_pps() {
        let au = len_prefixed(&[minimal_h264_sps().as_slice()]);
        assert!(detect_stream_config(VideoCodec::H264, &au).is_none());
    }

    #[test]
    fn test_avcc_h264_pps_only_no_sps() {
        let au = len_prefixed(&[minimal_h264_pps().as_slice()]);
        assert!(detect_stream_config(VideoCodec::H264, &au).is_none());
    }

    #[test]
    fn test_avcc_h264_idr_only() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = len_prefixed(&[idr.as_slice()]);
        assert!(detect_stream_config(VideoCodec::H264, &au).is_none());
    }

    #[test]
    fn test_avcc_h264_multiple_sps_pps() {
        let sps2 = vec![0x67, 0x4D, 0x00, 0x0A, 0xFF, 0xE1, 0x00];
        let pps2 = vec![0x68, 0xEE, 0x06, 0xF2, 0xC0];
        let au = len_prefixed(&[
            minimal_h264_sps().as_slice(),
            sps2.as_slice(),
            minimal_h264_pps().as_slice(),
            pps2.as_slice(),
        ]);
        let c = detect_stream_config(VideoCodec::H264, &au).unwrap();
        match c {
            DecoderConfig::H264(cfg) => {
                let cd = cfg.codec_data.unwrap();
                assert_eq!(cd[5], 0xE2); // 0xE0 | 2 SPS
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_unsupported_codec_vp8() {
        assert!(detect_stream_config(VideoCodec::Vp8, &[0, 0, 0, 1, 0x9A]).is_none());
    }

    #[test]
    fn test_unsupported_codec_vp9() {
        assert!(detect_stream_config(VideoCodec::Vp9, &[0, 0, 0, 1]).is_none());
    }

    #[test]
    fn test_unsupported_codec_jpeg() {
        assert!(detect_stream_config(VideoCodec::Jpeg, &[0xFF, 0xD8]).is_none());
    }

    #[test]
    fn test_empty_data() {
        assert!(detect_stream_config(VideoCodec::H264, &[]).is_none());
        assert!(detect_stream_config(VideoCodec::Hevc, &[]).is_none());
    }

    #[test]
    fn test_one_byte() {
        assert!(detect_stream_config(VideoCodec::H264, &[0x42]).is_none());
    }

    #[test]
    fn test_two_bytes() {
        assert!(detect_stream_config(VideoCodec::H264, &[0, 0]).is_none());
    }

    #[test]
    fn test_three_bytes_not_sc() {
        assert!(detect_stream_config(VideoCodec::H264, &[0, 0, 2]).is_none());
    }

    #[test]
    fn test_four_bytes_almost_sc() {
        assert!(detect_stream_config(VideoCodec::H264, &[0, 0, 0, 2]).is_none());
    }

    #[test]
    fn test_length_prefix_zero() {
        assert!(detect_stream_config(VideoCodec::H264, &[0, 0, 0, 0]).is_none());
    }

    #[test]
    fn test_length_prefix_exceeds_data() {
        let au = vec![0, 0, 0, 10, 1, 2, 3];
        assert!(detect_stream_config(VideoCodec::H264, &au).is_none());
    }

    #[test]
    fn test_garbage_data() {
        assert!(detect_stream_config(VideoCodec::H264, &[0xDE, 0xAD, 0xBE, 0xEF]).is_none());
    }

    #[test]
    fn test_avcc_record_binary_layout() {
        let sps = vec![vec![0x67, 0x42, 0x00, 0x1E, 0xFF]];
        let pps = vec![vec![0x68, 0xCE, 0x3C, 0x80]];
        let r = build_avcc_record(&sps, &pps).unwrap();
        assert_eq!(r[0], 1);
        assert_eq!(r[1], 0x42);
        assert_eq!(r[2], 0x00);
        assert_eq!(r[3], 0x1E);
        assert_eq!(r[4], 0xFF);
        assert_eq!(r[5], 0xE1);
        assert_eq!(&r[6..8], &(5u16).to_be_bytes());
        assert_eq!(&r[8..13], sps[0].as_slice());
        assert_eq!(r[13], 1);
        assert_eq!(&r[14..16], &(4u16).to_be_bytes());
        assert_eq!(&r[16..20], pps[0].as_slice());
    }

    // VPS / SPS / PPS from cros-codecs `test_data/bear.h265` (Chromium bear clip).
    fn hevc_vps() -> Vec<u8> {
        vec![
            0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x3C, 0x95, 0xC0, 0x90,
        ]
    }

    fn hevc_sps() -> Vec<u8> {
        vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x03, 0x00, 0x3C, 0xA0, 0x0A, 0x08, 0x0B, 0x9F, 0x79, 0x65, 0x79, 0x24, 0xCA,
            0xE0, 0x10, 0x00, 0x00, 0x06, 0x40, 0x00, 0x00, 0xBB, 0x50, 0x80,
        ]
    }

    fn hevc_pps() -> Vec<u8> {
        vec![0x44, 0x01, 0xC1, 0x73, 0xD1, 0x89]
    }

    #[test]
    fn test_annexb_hevc_4byte_sc() {
        let mut au = prepend_sc(&hevc_vps());
        au.extend_from_slice(&prepend_sc(&hevc_sps()));
        let c = detect_stream_config(VideoCodec::Hevc, &au).unwrap();
        match c {
            DecoderConfig::Hevc(cfg) => assert_eq!(cfg.stream_format, HevcStreamFormat::ByteStream),
            _ => panic!(),
        }
    }

    #[test]
    fn test_annexb_hevc_3byte_sc() {
        let mut au = vec![0, 0, 1];
        au.extend_from_slice(&hevc_sps());
        let c = detect_stream_config(VideoCodec::Hevc, &au).unwrap();
        match c {
            DecoderConfig::Hevc(cfg) => assert_eq!(cfg.stream_format, HevcStreamFormat::ByteStream),
            _ => panic!(),
        }
    }

    #[test]
    fn test_annexb_hevc_idr_only() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let au = prepend_sc(&idr);
        let c = detect_stream_config(VideoCodec::Hevc, &au).unwrap();
        match c {
            DecoderConfig::Hevc(cfg) => assert_eq!(cfg.stream_format, HevcStreamFormat::ByteStream),
            _ => panic!(),
        }
    }

    /// Pad VPS RBSP so the first length-prefixed NAL is exactly **256** bytes (`00 00 01 00`).
    fn hevc_vps_nalu_len_256() -> Vec<u8> {
        let mut v = hevc_vps();
        assert!(v.len() < 256);
        v.resize(256, 0);
        v
    }

    #[test]
    fn test_hvcc_hevc_first_nal_len_256_detects_hvc1() {
        let au = len_prefixed(&[
            hevc_vps_nalu_len_256().as_slice(),
            hevc_vps().as_slice(),
            hevc_sps().as_slice(),
            hevc_pps().as_slice(),
        ]);
        let c = detect_stream_config(VideoCodec::Hevc, &au).expect("hvc1");
        match c {
            DecoderConfig::Hevc(cfg) => assert_eq!(cfg.stream_format, HevcStreamFormat::Hvc1),
            _ => panic!(),
        }
    }

    #[test]
    fn test_hvcc_hevc_vps_sps_pps_idr() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let au = len_prefixed(&[
            hevc_vps().as_slice(),
            hevc_sps().as_slice(),
            hevc_pps().as_slice(),
            idr.as_slice(),
        ]);
        let c = detect_stream_config(VideoCodec::Hevc, &au).expect("hvcc");
        match c {
            DecoderConfig::Hevc(cfg) => {
                assert_eq!(cfg.stream_format, HevcStreamFormat::Hvc1);
                let cd = cfg.codec_data.expect("codec_data");
                assert_eq!(cd[0], 1);
                assert!(cd.len() > 23);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_hvcc_hevc_missing_vps() {
        let au = len_prefixed(&[
            hevc_sps().as_slice(),
            hevc_pps().as_slice(),
            vec![0x26, 0x01, 0x02].as_slice(),
        ]);
        assert!(detect_stream_config(VideoCodec::Hevc, &au).is_none());
    }

    #[test]
    fn test_hvcc_hevc_missing_sps() {
        let au = len_prefixed(&[
            hevc_vps().as_slice(),
            hevc_pps().as_slice(),
            vec![0x26, 0x01, 0x02].as_slice(),
        ]);
        assert!(detect_stream_config(VideoCodec::Hevc, &au).is_none());
    }

    #[test]
    fn test_hvcc_hevc_missing_pps() {
        let au = len_prefixed(&[
            hevc_vps().as_slice(),
            hevc_sps().as_slice(),
            vec![0x26, 0x01, 0x02].as_slice(),
        ]);
        assert!(detect_stream_config(VideoCodec::Hevc, &au).is_none());
    }

    #[test]
    fn test_hvcc_hevc_idr_only() {
        let au = len_prefixed(&[vec![0x26, 0x01, 0x02, 0x03].as_slice()]);
        assert!(detect_stream_config(VideoCodec::Hevc, &au).is_none());
    }

    #[test]
    fn test_malformed_sps_avcc() {
        // Type SPS (0x67) but truncated RBSP — still enough for profile/level bytes in AVCC
        let bad_sps = vec![0x67, 0x42, 0x00];
        let au = len_prefixed(&[bad_sps.as_slice(), minimal_h264_pps().as_slice()]);
        let c = detect_stream_config(VideoCodec::H264, &au);
        assert!(c.is_none());
    }

    #[test]
    fn test_hvcc_record_binary_layout_smoke() {
        let vps = vec![hevc_vps()];
        let sps = vec![hevc_sps()];
        let pps = vec![hevc_pps()];
        let r = build_hvcc_record(&vps, &sps, &pps).expect("hvcc");
        assert_eq!(r[0], 1);
        // Byte after avgFrameRate: constantFrameRate + numTemporalLayers + temporalIdNested + lengthSizeMinusOne
        assert_eq!(r[21], 0x0F);
        assert_eq!(r[22], 3); // numOfArrays (VPS, SPS, PPS)
    }

    // ── is_random_access_point ─────────────────────────────────────

    fn h264_au_annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut au = Vec::new();
        for nal in nals {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        au
    }

    /// HEVC CRA slice NAL (type 21), 2-byte header + minimal payload (parses as IRAP).
    fn hevc_cra_slice() -> Vec<u8> {
        vec![0x2A, 0x01, 0xAF, 0x09, 0x40]
    }

    #[test]
    fn test_rap_h264_annexb_sps_pps_idr() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = h264_au_annexb(&[
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
            idr.as_slice(),
        ]);
        assert!(is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_rap_h264_annexb_idr_only() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = prepend_sc(&idr);
        assert!(!is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_rap_h264_annexb_sps_pps_non_idr_slice() {
        let slice = vec![0x41, 0x9A, 0x26, 0xE0, 0x3F]; // type 1 non-IDR slice-like
        let au = h264_au_annexb(&[
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
            slice.as_slice(),
        ]);
        assert!(!is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_rap_h264_len_prefixed_sps_pps_idr() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = len_prefixed(&[
            minimal_h264_sps().as_slice(),
            minimal_h264_pps().as_slice(),
            idr.as_slice(),
        ]);
        assert!(is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_rap_h264_len_prefixed_idr_only() {
        let idr = vec![0x65, 0x88, 0x84, 0x00];
        let au = len_prefixed(&[idr.as_slice()]);
        assert!(!is_random_access_point(VideoCodec::H264, &au));
    }

    #[test]
    fn test_rap_hevc_annexb_vps_sps_pps_idr() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let mut au = prepend_sc(&hevc_vps());
        au.extend_from_slice(&prepend_sc(&hevc_sps()));
        au.extend_from_slice(&prepend_sc(&hevc_pps()));
        au.extend_from_slice(&prepend_sc(&idr));
        assert!(is_random_access_point(VideoCodec::Hevc, &au));
    }

    #[test]
    fn test_rap_hevc_annexb_idr_only() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let au = prepend_sc(&idr);
        assert!(!is_random_access_point(VideoCodec::Hevc, &au));
    }

    #[test]
    fn test_rap_hevc_annexb_vps_sps_pps_cra() {
        let cra = hevc_cra_slice();
        let mut au = prepend_sc(&hevc_vps());
        au.extend_from_slice(&prepend_sc(&hevc_sps()));
        au.extend_from_slice(&prepend_sc(&hevc_pps()));
        au.extend_from_slice(&prepend_sc(&cra));
        assert!(is_random_access_point(VideoCodec::Hevc, &au));
    }

    #[test]
    fn test_rap_hevc_missing_vps() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let mut au = prepend_sc(&hevc_sps());
        au.extend_from_slice(&prepend_sc(&hevc_pps()));
        au.extend_from_slice(&prepend_sc(&idr));
        assert!(!is_random_access_point(VideoCodec::Hevc, &au));
    }

    #[test]
    fn test_rap_jpeg_png_raw_nonempty() {
        assert!(is_random_access_point(
            VideoCodec::Jpeg,
            &[0xFF, 0xD8, 0xFF]
        ));
        assert!(is_random_access_point(
            VideoCodec::Png,
            &[0x89, 0x50, 0x4E, 0x47]
        ));
        assert!(is_random_access_point(VideoCodec::RawRgba, &[0; 4]));
        assert!(is_random_access_point(VideoCodec::RawRgb, &[0; 3]));
        assert!(is_random_access_point(VideoCodec::RawNv12, &[0; 2]));
    }

    #[test]
    fn test_rap_jpeg_png_empty() {
        assert!(!is_random_access_point(VideoCodec::Jpeg, &[]));
        assert!(!is_random_access_point(VideoCodec::Png, &[]));
        assert!(!is_random_access_point(VideoCodec::RawRgba, &[]));
    }

    #[test]
    fn test_rap_vp8_vp9_av1_false() {
        assert!(!is_random_access_point(
            VideoCodec::Vp8,
            &[0, 0, 0, 1, 0x9A]
        ));
        assert!(!is_random_access_point(VideoCodec::Vp9, &[0, 0, 0, 1]));
        assert!(!is_random_access_point(VideoCodec::Av1, &[0, 0, 0, 1]));
    }

    #[test]
    fn test_rap_empty_h264_hevc() {
        assert!(!is_random_access_point(VideoCodec::H264, &[]));
        assert!(!is_random_access_point(VideoCodec::Hevc, &[]));
    }

    #[test]
    fn test_rap_hevc_len_prefixed_vps_sps_pps_idr() {
        let idr = vec![0x26, 0x01, 0xAF, 0x09, 0x40];
        let au = len_prefixed(&[
            hevc_vps().as_slice(),
            hevc_sps().as_slice(),
            hevc_pps().as_slice(),
            idr.as_slice(),
        ]);
        assert!(is_random_access_point(VideoCodec::Hevc, &au));
    }
}
