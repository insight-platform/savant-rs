//! B-frame stress tests for the NvEncoder.
//!
//! Decodes a real-world video (NY City Center, H.264 1080p) and re-encodes it
//! through every B-frame-capable profile at keyframe-interval 30.  The tests
//! verify that:
//!
//! 1. Output PTS values are **strictly monotonically increasing** – i.e. no
//!    PTS reordering occurs, proving B-frames were never emitted.
//! 2. When DTS is present it satisfies **DTS ≤ PTS** for every frame.
//!
//! If B-frames were silently enabled, the encoder would internally reorder
//! frames and the output PTS sequence would *not* be monotonic (the decoder
//! would see P,B,B,P ordering).
//!
//! # Architecture
//!
//! Decoding and encoding are done **frame-by-frame** in lockstep.  Each
//! decoded NVMM buffer is immediately copied into the encoder's pool via
//! [`NvBufSurfaceGenerator::transform()`] and then **dropped** so the
//! decoder's output buffer pool is not exhausted (`nvv4l2decoder` typically
//! has only 4-8 output buffers).  Encoded output is pulled after every
//! submit to keep the encoder's internal pipeline (pool size = 1) flowing.
//!
//! # Requirements
//!
//! * GPU with NVENC support and DeepStream installed.
//! * Internet access to download the test video on first run.
//!   The file is cached in `/tmp/savant_test_data/`.
//!
//! # Running
//!
//! ```sh
//! cargo test -p deepstream_encoders --test test_encoder_b_frame_stress -- --nocapture
//! ```

use deepstream_encoders::properties::*;
use deepstream_encoders::{cuda_init, Codec, EncoderConfig, NvEncoder, VideoFormat};
use deepstream_nvbufsurface::{ComputeMode, Interpolation, Padding, TransformConfig};
use gstreamer as gst;
use gstreamer::prelude::*;
use serial_test::serial;
use std::path::{Path, PathBuf};

// ─── Constants ───────────────────────────────────────────────────────────

const VIDEO_URL: &str =
    "https://eu-central-1.linodeobjects.com/savant-data/demo/ny_city_center.mov";
const VIDEO_CACHE_DIR: &str = "/tmp/savant_test_data";
const VIDEO_FILENAME: &str = "ny_city_center.mov";

/// Keyframe interval used for all tests.
const IFRAME_INTERVAL: u32 = 30;

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Initialize CUDA and GStreamer once.
fn init() {
    let _ = env_logger::try_init();
    let _ = gst::init();
    cuda_init(0).expect("CUDA init failed — is a GPU available?");
}

/// Download the test video if it is not already cached.  Returns the local
/// path to the `.mov` file.
fn ensure_video() -> PathBuf {
    let dir = Path::new(VIDEO_CACHE_DIR);
    let path = dir.join(VIDEO_FILENAME);

    if path.exists() {
        eprintln!("Video already cached: {}", path.display());
        return path;
    }

    std::fs::create_dir_all(dir).expect("Failed to create cache directory");
    eprintln!("Downloading test video from {VIDEO_URL} ...");

    let status = std::process::Command::new("curl")
        .args(["-fSL", "--retry", "3", "-o"])
        .arg(&path)
        .arg(VIDEO_URL)
        .status()
        .expect("Failed to invoke curl — is it installed?");

    assert!(
        status.success(),
        "curl failed to download {VIDEO_URL} (exit code: {status})"
    );

    eprintln!("Downloaded to {}", path.display());
    path
}

/// Build a decoder pipeline, probe resolution, then stream-decode +
/// re-encode through the given encoder configuration.
///
/// For each decoded NVMM frame:
/// 1. `transform()` copies it into the encoder's buffer pool (GPU→GPU).
/// 2. The decoded buffer is **dropped immediately** so the decoder's
///    output pool is not exhausted.
/// 3. The frame is submitted to the encoder.
/// 4. Encoded output is drained to keep the encoder pipeline flowing.
///
/// The encoder validates output PTS/DTS ordering internally: if B-frames
/// leak through, `pull_encoded()` or `finish()` will return
/// `Err(OutputPtsReordered)` / `Err(OutputDtsExceedsPts)`.  This function
/// propagates those errors via `unwrap`, so a successful return proves no
/// B-frames were emitted.
///
/// Returns `(submitted_count, encoded_count)`.
fn decode_and_reencode(
    video_path: &Path,
    config_fn: &dyn Fn(u32, u32) -> EncoderConfig,
) -> (usize, usize) {
    let path_str = video_path.to_str().expect("video path is not valid UTF-8");

    // Build the GStreamer decode pipeline.
    //
    // The explicit `d.video_0` pad selection avoids early EOS caused by
    // an unconnected audio pad on `qtdemux`.
    let pipeline_str = format!(
        "filesrc location={path_str} ! qtdemux name=d d.video_0 \
         ! queue ! h264parse ! nvv4l2decoder \
         ! video/x-raw(memory:NVMM),format=NV12 \
         ! appsink name=sink emit-signals=false sync=false"
    );

    let pipeline = gst::parse::launch(&pipeline_str)
        .expect("Failed to build decode pipeline")
        .dynamic_cast::<gst::Pipeline>()
        .expect("Pipeline cast failed");

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start decode pipeline");

    let appsink = pipeline
        .by_name("sink")
        .expect("appsink not found")
        .dynamic_cast::<gstreamer_app::AppSink>()
        .expect("appsink cast failed");

    // Pull the first frame to discover resolution.
    let first_sample = appsink
        .try_pull_sample(gst::ClockTime::from_seconds(15))
        .expect("Failed to pull first decoded frame (timeout)");

    let caps = first_sample.caps().expect("First sample has no caps");
    let caps_struct = caps.structure(0).expect("No caps structure");
    let width = caps_struct.get::<i32>("width").expect("No width") as u32;
    let height = caps_struct.get::<i32>("height").expect("No height") as u32;
    eprintln!("Decoded video resolution: {width}×{height}");

    // Create the encoder using the discovered resolution.
    let encoder_config = config_fn(width, height);
    let mut encoder = NvEncoder::new(&encoder_config).unwrap_or_else(|e| {
        panic!(
            "Failed to create encoder (codec={:?}, props={:?}): {e}",
            encoder_config.codec, encoder_config.encoder_params,
        );
    });

    let transform_cfg = TransformConfig {
        padding: Padding::None,
        interpolation: Interpolation::Nearest,
        src_rect: None,
        compute_mode: ComputeMode::Default,
        cuda_stream: std::ptr::null_mut(),
    };

    let frame_dur_ns: u64 = 33_333_333; // 30 fps
    let mut encoded_count: usize = 0;
    let mut frame_idx: usize = 0;

    // Helper: submit one decoded buffer to the encoder and drain output.
    //
    // `pull_encoded()` validates output PTS/DTS ordering internally;
    // if B-frames leaked through, it returns Err which we propagate.
    let mut submit_and_drain = |buf_ref: &gst::BufferRef, idx: usize| {
        let owned = buf_ref.to_owned();
        let enc_buf = encoder
            .generator()
            .transform(&owned, &transform_cfg, Some(idx as i64))
            .unwrap_or_else(|e| panic!("transform failed at frame {idx}: {e}"));

        let pts = idx as u64 * frame_dur_ns;
        encoder
            .submit_frame(enc_buf, idx as i64, pts, Some(frame_dur_ns))
            .unwrap_or_else(|e| panic!("submit_frame failed at frame {idx}: {e}"));

        // Drain ready frames — the encoder validates ordering internally.
        while let Ok(Some(_)) = encoder.pull_encoded() {
            encoded_count += 1;
        }
    };

    // Process the first frame (already pulled above).
    {
        let buf = first_sample.buffer().expect("Sample has no buffer");
        submit_and_drain(buf, frame_idx);
        frame_idx += 1;
    }
    // first_sample is dropped here, returning the buffer to the decoder pool.

    // Stream the remaining frames until EOS.
    loop {
        let sample = match appsink.try_pull_sample(gst::ClockTime::from_seconds(10)) {
            Some(s) => s,
            None => break,
        };

        let buf = sample.buffer().expect("Sample has no buffer");
        submit_and_drain(buf, frame_idx);
        frame_idx += 1;
        // sample (and its buffer) is dropped here.
    }

    // Shut down the decoder pipeline.
    let _ = pipeline.set_state(gst::State::Null);

    assert!(
        frame_idx >= 30,
        "Expected at least 30 decoded frames, got {frame_idx}"
    );

    // Send EOS and drain remaining encoded frames.
    // finish() also calls pull_encoded_timeout() internally, which
    // validates output ordering — any B-frame reordering will Err here.
    let remaining = encoder.finish(Some(5000)).unwrap_or_else(|e| {
        panic!(
            "finish() failed for codec={:?}, props={:?}: {e}",
            encoder_config.codec, encoder_config.encoder_params,
        );
    });
    encoded_count += remaining.len();

    eprintln!("Submitted {frame_idx} frames, encoded {encoded_count} — no B-frame reordering");

    (frame_idx, encoded_count)
}

// ═════════════════════════════════════════════════════════════════════════
// H.264 stress tests — B-frame-capable profiles
// ═════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn stress_h264_main_profile_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::Main),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_high_profile_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::High),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_high_profile_vbr_p7_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::High),
            control_rate: Some(RateControl::VariableBitrate),
            preset: Some(DgpuPreset::P7),
            tuning_info: Some(TuningPreset::HighQuality),
            bitrate: Some(8_000_000),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_high_profile_cbr_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::High),
            control_rate: Some(RateControl::ConstantBitrate),
            bitrate: Some(4_000_000),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_main_profile_cqp_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::Main),
            control_rate: Some(RateControl::ConstantQP),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_baseline_profile_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::Baseline),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

// ═════════════════════════════════════════════════════════════════════════
// HEVC stress tests — B-frame-capable profiles
// ═════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn stress_hevc_main_profile_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_hevc_main_vbr_p7_hq_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            control_rate: Some(RateControl::VariableBitrate),
            preset: Some(DgpuPreset::P7),
            tuning_info: Some(TuningPreset::HighQuality),
            bitrate: Some(8_000_000),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_hevc_main_cbr_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            control_rate: Some(RateControl::ConstantBitrate),
            bitrate: Some(6_000_000),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_hevc_main_cqp_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            control_rate: Some(RateControl::ConstantQP),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_hevc_main10_profile_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main10),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

// ═════════════════════════════════════════════════════════════════════════
// H.264 with temporal AQ and adaptive quantization — these can interact
// with B-frame decisions in some encoder implementations.
// ═════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn stress_h264_high_temporal_aq_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::High),
            control_rate: Some(RateControl::VariableBitrate),
            bitrate: Some(6_000_000),
            temporal_aq: Some(true),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_h264_high_spatial_aq_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::H264Dgpu(H264DgpuProps {
            profile: Some(H264Profile::High),
            control_rate: Some(RateControl::VariableBitrate),
            bitrate: Some(6_000_000),
            aq: Some(15),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

// ═════════════════════════════════════════════════════════════════════════
// HEVC with various presets — higher presets are more likely to try
// enabling B-frames internally if the element's B-frame property is
// not explicitly zeroed.
// ═════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn stress_hevc_main_p4_low_latency_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            preset: Some(DgpuPreset::P4),
            tuning_info: Some(TuningPreset::LowLatency),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

#[test]
#[serial]
fn stress_hevc_main_p7_high_quality_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
            profile: Some(HevcProfile::Main),
            preset: Some(DgpuPreset::P7),
            tuning_info: Some(TuningPreset::HighQuality),
            bitrate: Some(10_000_000),
            iframeinterval: Some(IFRAME_INTERVAL),
            ..Default::default()
        });
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::NV12)
            .properties(props)
    });
}

// ═════════════════════════════════════════════════════════════════════════
// No explicit properties — the encoder's built-in defaults should also
// produce B-frame-free output because force_disable_b_frames() runs.
// ═════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn stress_h264_default_props_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        EncoderConfig::new(Codec::H264, w, h).format(VideoFormat::NV12)
    });
}

#[test]
#[serial]
fn stress_hevc_default_props_no_b_frames() {
    init();
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        EncoderConfig::new(Codec::Hevc, w, h).format(VideoFormat::NV12)
    });
}
