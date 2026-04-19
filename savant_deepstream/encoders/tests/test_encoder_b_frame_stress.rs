//! B-frame stress tests for [`NvEncoder`].
//!
//! Decodes a real-world 1080p H.264 video (NY City Center) and re-encodes
//! every decoded frame through every B-frame-capable profile.  The
//! encoder validates output ordering internally (strict PTS monotonicity
//! and DTS ≤ PTS); any B-frame leakage surfaces as an
//! [`EncoderError::OutputPtsReordered`] or
//! [`EncoderError::OutputDtsExceedsPts`] either during `submit_frame` or
//! `graceful_shutdown`.
//!
//! # Running
//! ```sh
//! cargo test -p savant-deepstream-encoders --test test_encoder_b_frame_stress -- --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{has_nvenc, init, is_jetson};
use deepstream_buffers::{ComputeMode, Interpolation, Padding, TransformConfig};
use deepstream_encoders::prelude::*;
use gstreamer as gst;
use gstreamer::prelude::*;
use serial_test::serial;

// ─── Constants ───────────────────────────────────────────────────────

const VIDEO_URL: &str =
    "https://eu-central-1.linodeobjects.com/savant-data/demo/ny_city_center.mov";
const VIDEO_CACHE_DIR: &str = "/tmp/savant_test_data";
const VIDEO_FILENAME: &str = "ny_city_center.mov";
const IFRAME_INTERVAL: u32 = 30;
const FRAME_DUR_NS: u64 = 33_333_333;

// ─── Asset management ────────────────────────────────────────────────

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
    assert!(status.success(), "curl failed to download {VIDEO_URL}");
    eprintln!("Downloaded to {}", path.display());
    path
}

// ─── Decode + re-encode driver ───────────────────────────────────────

/// Build the GStreamer decode pipeline, probe the video resolution, and
/// re-encode each decoded frame through an [`NvEncoder`] created from
/// `config_fn(width, height)`.  Panics on any PTS/DTS ordering
/// violation.
fn decode_and_reencode(video_path: &Path, config_fn: &dyn Fn(u32, u32) -> EncoderConfig) {
    let path_str = video_path.to_str().expect("video path is not valid UTF-8");

    // dGPU gets NV12 directly; Jetson needs an extra nvvideoconvert to
    // turn NV12 into pitch-linear RGBA so SurfaceView + EGL can register
    // the slot.
    let after_decoder = if is_jetson() {
        "nvv4l2decoder ! nvvideoconvert compute-hw=1 ! \
         video/x-raw(memory:NVMM),format=RGBA"
    } else {
        "nvv4l2decoder ! video/x-raw(memory:NVMM),format=NV12"
    };
    let pipeline_str = format!(
        "filesrc location={path_str} ! qtdemux name=d d.video_0 \
         ! queue ! h264parse ! {after_decoder} \
         ! appsink name=sink emit-signals=false sync=false"
    );

    let pipeline = gst::parse::launch(&pipeline_str)
        .expect("Failed to build decode pipeline")
        .dynamic_cast::<gst::Pipeline>()
        .expect("Pipeline cast failed");
    pipeline.set_state(gst::State::Playing).unwrap();

    let appsink = pipeline
        .by_name("sink")
        .expect("appsink not found")
        .dynamic_cast::<gstreamer_app::AppSink>()
        .expect("appsink cast failed");

    // Probe resolution from the first sample.
    let first_sample = appsink
        .try_pull_sample(gst::ClockTime::from_seconds(15))
        .expect("Failed to pull first decoded frame (timeout)");
    let caps = first_sample.caps().expect("First sample has no caps");
    let caps_struct = caps.structure(0).expect("No caps structure");
    let width = caps_struct.get::<i32>("width").expect("No width") as u32;
    let height = caps_struct.get::<i32>("height").expect("No height") as u32;
    eprintln!("Decoded video resolution: {width}×{height}");

    let encoder_config = config_fn(width, height);
    let nv_config = NvEncoderConfig::new(0, encoder_config)
        .operation_timeout(Duration::from_secs(5))
        .name("b_frame_stress");
    let encoder = NvEncoder::new(nv_config).expect("NvEncoder::new failed");

    let transform_cfg = TransformConfig {
        padding: Padding::None,
        dst_padding: None,
        interpolation: Interpolation::Nearest,
        compute_mode: ComputeMode::Default,
        cuda_stream: deepstream_buffers::CudaStream::default(),
    };

    let encoded_count = AtomicUsize::new(0);
    let mut frame_idx: usize = 0;

    // Submit one decoded sample into the encoder, transforming NVMM
    // → encoder-owned NVMM so the decoder's output pool is not held.
    let submit_sample = |sample: &gst::Sample, idx: usize| {
        let buf_ref = sample.buffer().expect("Sample has no buffer");
        let owned = buf_ref.to_owned();
        let src_view = deepstream_buffers::SurfaceView::from_gst_buffer(owned, 0)
            .unwrap_or_else(|e| panic!("SurfaceView failed at frame {idx}: {e}"));
        let enc_buf = encoder
            .generator()
            .lock()
            .transform_to_buffer(&src_view, &transform_cfg, None)
            .unwrap_or_else(|e| panic!("transform failed at frame {idx}: {e}"));
        let pts = idx as u64 * FRAME_DUR_NS;
        encoder
            .submit_frame(enc_buf, idx as u128, pts, Some(FRAME_DUR_NS))
            .unwrap_or_else(|e| panic!("submit_frame failed at frame {idx}: {e}"));
        // Drain ready frames non-blockingly so pool slots are freed.
        while let Ok(Some(out)) = encoder.try_recv() {
            match out {
                NvEncoderOutput::Frame(_) => {
                    encoded_count.fetch_add(1, Ordering::Relaxed);
                }
                NvEncoderOutput::Error(e) => {
                    panic!("encoder reported error during drain at frame {idx}: {e}")
                }
                NvEncoderOutput::Event(_)
                | NvEncoderOutput::SourceEos { .. }
                | NvEncoderOutput::Eos => {}
            }
        }
    };

    submit_sample(&first_sample, frame_idx);
    frame_idx += 1;
    drop(first_sample);

    loop {
        let Some(sample) = appsink.try_pull_sample(gst::ClockTime::from_seconds(10)) else {
            break;
        };
        submit_sample(&sample, frame_idx);
        frame_idx += 1;
    }

    let _ = pipeline.set_state(gst::State::Null);
    assert!(
        frame_idx >= 30,
        "Expected at least 30 decoded frames, got {frame_idx}"
    );

    // Final drain — graceful_shutdown also validates ordering.
    encoder
        .graceful_shutdown(Some(Duration::from_secs(10)), |out| match out {
            NvEncoderOutput::Frame(_) => {
                encoded_count.fetch_add(1, Ordering::Relaxed);
            }
            NvEncoderOutput::Error(e) => panic!("encoder error during shutdown: {e}"),
            _ => {}
        })
        .expect("graceful_shutdown failed");

    eprintln!(
        "Submitted {frame_idx} frames, encoded {} — no B-frame reordering",
        encoded_count.load(Ordering::Relaxed)
    );
}

// ─── Config helpers ──────────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
fn h264_config(w: u32, h: u32, props: H264DgpuProps) -> EncoderConfig {
    EncoderConfig::H264(
        H264EncoderConfig::new(w, h)
            .format(VideoFormat::NV12)
            .props(props),
    )
}

#[cfg(target_arch = "aarch64")]
fn h264_config(w: u32, h: u32, props: H264JetsonProps) -> EncoderConfig {
    EncoderConfig::H264(
        H264EncoderConfig::new(w, h)
            .format(VideoFormat::RGBA)
            .props(props),
    )
}

#[cfg(not(target_arch = "aarch64"))]
fn hevc_config(w: u32, h: u32, props: HevcDgpuProps) -> EncoderConfig {
    EncoderConfig::Hevc(
        HevcEncoderConfig::new(w, h)
            .format(VideoFormat::NV12)
            .props(props),
    )
}

#[cfg(target_arch = "aarch64")]
fn hevc_config(w: u32, h: u32, props: HevcJetsonProps) -> EncoderConfig {
    EncoderConfig::Hevc(
        HevcEncoderConfig::new(w, h)
            .format(VideoFormat::RGBA)
            .props(props),
    )
}

// ─── H.264 stress ────────────────────────────────────────────────────

#[test]
#[serial]
fn stress_h264_main_profile_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            h264_config(
                w,
                h,
                H264DgpuProps {
                    profile: Some(H264Profile::Main),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            h264_config(
                w,
                h,
                H264JetsonProps {
                    profile: Some(H264Profile::Main),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_h264_high_profile_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            h264_config(
                w,
                h,
                H264DgpuProps {
                    profile: Some(H264Profile::High),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            h264_config(
                w,
                h,
                H264JetsonProps {
                    profile: Some(H264Profile::High),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_h264_high_vbr_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            h264_config(
                w,
                h,
                H264DgpuProps {
                    profile: Some(H264Profile::High),
                    control_rate: Some(RateControl::VariableBitrate),
                    preset: Some(DgpuPreset::P7),
                    tuning_info: Some(TuningPreset::HighQuality),
                    bitrate: Some(8_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            h264_config(
                w,
                h,
                H264JetsonProps {
                    profile: Some(H264Profile::High),
                    control_rate: Some(RateControl::VariableBitrate),
                    bitrate: Some(8_000_000),
                    preset_level: Some(JetsonPresetLevel::Slow),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_h264_high_cbr_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            h264_config(
                w,
                h,
                H264DgpuProps {
                    profile: Some(H264Profile::High),
                    control_rate: Some(RateControl::ConstantBitrate),
                    bitrate: Some(4_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            h264_config(
                w,
                h,
                H264JetsonProps {
                    profile: Some(H264Profile::High),
                    control_rate: Some(RateControl::ConstantBitrate),
                    bitrate: Some(4_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_h264_baseline_profile_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            h264_config(
                w,
                h,
                H264DgpuProps {
                    profile: Some(H264Profile::Baseline),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            h264_config(
                w,
                h,
                H264JetsonProps {
                    profile: Some(H264Profile::Baseline),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

// ─── HEVC stress ─────────────────────────────────────────────────────

#[test]
#[serial]
fn stress_hevc_main_profile_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            hevc_config(
                w,
                h,
                HevcDgpuProps {
                    profile: Some(HevcProfile::Main),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            hevc_config(
                w,
                h,
                HevcJetsonProps {
                    profile: Some(HevcProfile::Main),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_hevc_main_vbr_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            hevc_config(
                w,
                h,
                HevcDgpuProps {
                    profile: Some(HevcProfile::Main),
                    control_rate: Some(RateControl::VariableBitrate),
                    preset: Some(DgpuPreset::P7),
                    tuning_info: Some(TuningPreset::HighQuality),
                    bitrate: Some(8_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            hevc_config(
                w,
                h,
                HevcJetsonProps {
                    profile: Some(HevcProfile::Main),
                    control_rate: Some(RateControl::VariableBitrate),
                    bitrate: Some(8_000_000),
                    preset_level: Some(JetsonPresetLevel::Slow),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_hevc_main_cbr_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            hevc_config(
                w,
                h,
                HevcDgpuProps {
                    profile: Some(HevcProfile::Main),
                    control_rate: Some(RateControl::ConstantBitrate),
                    bitrate: Some(6_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            hevc_config(
                w,
                h,
                HevcJetsonProps {
                    profile: Some(HevcProfile::Main),
                    control_rate: Some(RateControl::ConstantBitrate),
                    bitrate: Some(6_000_000),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

#[test]
#[serial]
fn stress_hevc_main10_profile_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        #[cfg(not(target_arch = "aarch64"))]
        {
            hevc_config(
                w,
                h,
                HevcDgpuProps {
                    profile: Some(HevcProfile::Main10),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            hevc_config(
                w,
                h,
                HevcJetsonProps {
                    profile: Some(HevcProfile::Main10),
                    iframeinterval: Some(IFRAME_INTERVAL),
                    ..Default::default()
                },
            )
        }
    });
}

// ─── dGPU-only AQ variants ───────────────────────────────────────────

#[test]
#[serial]
fn stress_h264_high_temporal_aq_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    if is_jetson() {
        eprintln!("temporal_aq is dGPU-only — skipping on Jetson");
        return;
    }
    let video = ensure_video();
    #[cfg(not(target_arch = "aarch64"))]
    decode_and_reencode(&video, &|w, h| {
        h264_config(
            w,
            h,
            H264DgpuProps {
                profile: Some(H264Profile::High),
                control_rate: Some(RateControl::VariableBitrate),
                bitrate: Some(6_000_000),
                temporal_aq: Some(true),
                iframeinterval: Some(IFRAME_INTERVAL),
                ..Default::default()
            },
        )
    });
    #[cfg(target_arch = "aarch64")]
    let _ = video;
}

// ─── No explicit properties ──────────────────────────────────────────

#[test]
#[serial]
fn stress_h264_default_props_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        if is_jetson() {
            EncoderConfig::H264(H264EncoderConfig::new(w, h).format(VideoFormat::RGBA))
        } else {
            EncoderConfig::H264(H264EncoderConfig::new(w, h).format(VideoFormat::NV12))
        }
    });
}

#[test]
#[serial]
fn stress_hevc_default_props_no_b_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping");
        return;
    }
    let video = ensure_video();
    decode_and_reencode(&video, &|w, h| {
        if is_jetson() {
            EncoderConfig::Hevc(HevcEncoderConfig::new(w, h).format(VideoFormat::RGBA))
        } else {
            EncoderConfig::Hevc(HevcEncoderConfig::new(w, h).format(VideoFormat::NV12))
        }
    });
}
