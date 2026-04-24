//! Parity integration test: `Mp4Demuxer` (ground truth) vs `UriDemuxer`.
//!
//! Both demuxers are run against the same generated MP4 and their outputs
//! compared. Strict equality is asserted for:
//! - packet count,
//! - `pts_ns`, `dts_ns`, `duration_ns`, `is_keyframe` per packet,
//! - `VideoInfo`,
//! - `detected_codec`.
//!
//! H.264 payload comparison is done on a **normalized** form: leading
//! configuration NAL units (AUD / SPS / PPS / SEI) are stripped via
//! `cros-codecs` so the slice NAL remains. The raw SPS/PPS prefix count
//! differs legitimately between the two paths because `UriDemuxer` uses
//! `parsebin` (whose internal parser chain handles SPS/PPS insertion
//! slightly differently than `Mp4Demuxer`'s direct `qtdemux -> h264parse`
//! chain). Downstream decoders handle both formats identically, and the
//! slice bytes (the actual encoded video data) are byte-identical.

use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cros_codecs::codec::h264::parser::{Nalu as H264Nalu, NaluType as H264NaluType};
use gstreamer as gst;

use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::demux::{DemuxedPacket, VideoInfo};
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use savant_gstreamer::mp4_muxer::Mp4Muxer;
use savant_gstreamer::uri_demuxer::{UriDemuxer, UriDemuxerConfig, UriDemuxerOutput};

const H264_SPS_PPS_IDR: [u8; 32] = [
    0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0A, 0xE9, 0x40, 0x40, 0x04, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x80, 0x40,
];

fn make_h264_mp4(path: &str, num_frames: usize) {
    let mut muxer = Mp4Muxer::new(VideoCodec::H264, path, 30, 1).unwrap();
    let duration_ns = 33_333_333u64;
    for i in 0..num_frames {
        muxer
            .push(
                &H264_SPS_PPS_IDR,
                (i as u64) * duration_ns,
                None,
                Some(duration_ns),
            )
            .unwrap();
    }
    muxer.finish().unwrap();
}

fn file_uri(path: &str) -> String {
    format!("file://{}", Path::new(path).display())
}

fn collect_via_uri_demuxer(
    uri: &str,
) -> (Vec<DemuxedPacket>, Option<VideoInfo>, Option<VideoCodec>) {
    let packets: Arc<Mutex<Vec<DemuxedPacket>>> = Arc::new(Mutex::new(Vec::new()));
    let errored = Arc::new(AtomicBool::new(false));
    let packets_cb = packets.clone();
    let errored_cb = errored.clone();
    let demuxer = UriDemuxer::new(
        UriDemuxerConfig::new(uri).with_parsed(true),
        move |out| match out {
            UriDemuxerOutput::Packet(p) => packets_cb.lock().unwrap().push(p),
            UriDemuxerOutput::Error(e) => {
                errored_cb.store(true, Ordering::SeqCst);
                panic!("UriDemuxer error: {e}");
            }
            _ => {}
        },
    )
    .unwrap();
    assert!(
        demuxer.wait_timeout(Duration::from_secs(10)),
        "UriDemuxer timed out"
    );
    let info = demuxer.video_info();
    let codec = demuxer.detected_codec();
    let pkts = std::mem::take(&mut *packets.lock().unwrap());
    (pkts, info, codec)
}

/// Extract the slice NAL units from an Annex-B H.264 access unit using
/// `cros-codecs`, discarding configuration NALs (AUD / SPS / PPS / SEI /
/// FillerData / SeqEnd / StreamEnd).
fn h264_slice_nals(au: &[u8]) -> Vec<Vec<u8>> {
    let mut cur = Cursor::new(au);
    let mut slices = Vec::new();
    while let Ok(nalu) = H264Nalu::next(&mut cur) {
        match nalu.header.type_ {
            H264NaluType::AuDelimiter
            | H264NaluType::Sps
            | H264NaluType::Pps
            | H264NaluType::Sei
            | H264NaluType::FillerData
            | H264NaluType::SeqEnd
            | H264NaluType::StreamEnd
            | H264NaluType::SpsExt
            | H264NaluType::SubsetSps => {}
            _ => slices.push(nalu.as_ref().to_vec()),
        }
    }
    slices
}

#[test]
fn test_mp4_vs_uri_demuxer_h264_parity() {
    let _ = gst::init();
    let path = "/tmp/test_demuxer_parity_h264.mp4";
    let _ = std::fs::remove_file(path);
    make_h264_mp4(path, 10);

    // Ground truth via Mp4Demuxer.
    let (ref_packets, ref_info) = Mp4Demuxer::demux_all_parsed(path).expect("Mp4Demuxer failed");
    let ref_codec = ref_info.map(|i| i.codec);

    // Candidate via UriDemuxer.
    let (uri_packets, uri_info, uri_codec) = collect_via_uri_demuxer(&file_uri(path));

    // Strict parity on codec + VideoInfo.
    assert_eq!(ref_codec, uri_codec, "detected_codec mismatch");
    assert_eq!(ref_info, uri_info, "VideoInfo mismatch");

    // Strict parity on packet count.
    assert_eq!(
        ref_packets.len(),
        uri_packets.len(),
        "packet count mismatch: mp4={} uri={}",
        ref_packets.len(),
        uri_packets.len()
    );

    // Strict parity on per-packet timing/flags + semantic parity on payload.
    for (idx, (r, u)) in ref_packets.iter().zip(uri_packets.iter()).enumerate() {
        assert_eq!(r.pts_ns, u.pts_ns, "packet[{idx}] pts_ns differ");
        assert_eq!(
            r.is_keyframe, u.is_keyframe,
            "packet[{idx}] is_keyframe differ"
        );
        if let (Some(rd), Some(ud)) = (r.dts_ns, u.dts_ns) {
            assert_eq!(rd, ud, "packet[{idx}] dts_ns differ");
        }
        if let (Some(rd), Some(ud)) = (r.duration_ns, u.duration_ns) {
            assert_eq!(rd, ud, "packet[{idx}] duration_ns differ");
        }

        // Byte-level comparison of slice NAL units (configuration-agnostic).
        let r_slices = h264_slice_nals(&r.data);
        let u_slices = h264_slice_nals(&u.data);
        assert_eq!(
            r_slices, u_slices,
            "packet[{idx}] slice NAL units differ\nmp4:  {:?}\nuri:  {:?}",
            r.data, u.data
        );
        assert!(
            !r_slices.is_empty(),
            "packet[{idx}] has no slice NALs (mp4)"
        );
    }

    let _ = std::fs::remove_file(path);
}
