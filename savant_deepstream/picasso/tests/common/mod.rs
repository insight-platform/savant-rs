//! Shared test helpers for picasso integration tests.
//!
//! Based on patterns from `kb/picasso-rust/patterns.md`. Use `mod common;` in each test file
//! and then `use common::*` or `use common::{make_frame, make_gst_buffer, ...}`.

#![allow(dead_code)] // Helpers used by various test files

use deepstream_buffers::BufferGenerator;
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use savant_core::primitives::WithAttributes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Default frame duration in nanoseconds (33.33ms at 30fps).
pub const FRAME_DUR_NS: u64 = 33_333_333;

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

/// Creates a VideoFrameProxy with default 320x240 size.
pub fn make_frame(source_id: &str) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        320,
        240,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap()
}

/// Creates a VideoFrameProxy with custom width and height.
pub fn make_frame_sized(source_id: &str, w: i64, h: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap()
}

/// Creates a GStreamer buffer (stub for NOGPU tests).
/// Call `gstreamer::init().unwrap()` before first use.
pub fn make_gst_buffer() -> gstreamer::Buffer {
    gstreamer::Buffer::new()
}

/// Creates a SurfaceView wrapping a plain GStreamer buffer (stub for NOGPU tests).
pub fn make_surface_view() -> deepstream_buffers::SurfaceView {
    deepstream_buffers::SurfaceView::wrap(make_gst_buffer())
}

/// Creates a frame with PTS, duration, and a persistent attribute.
pub fn make_frame_with_attr(source_id: &str, idx: u64, ns: &str, name: &str) -> VideoFrameProxy {
    let frame = make_frame(source_id);
    let mut fm = frame.clone();
    fm.set_pts((idx * FRAME_DUR_NS) as i64).unwrap();
    fm.set_duration(Some(FRAME_DUR_NS as i64)).unwrap();
    fm.set_persistent_attribute(ns, name, &None, false, vec![]);
    frame
}

/// Adds a detection object to the frame. Returns the object ID.
pub fn add_object(frame: &VideoFrameProxy, cx: f32, cy: f32, w: f32, h: f32) -> i64 {
    add_object_with_label(frame, "det", "car", cx, cy, w, h)
}

/// Adds a detection object with custom namespace and label.
pub fn add_object_with_label(
    frame: &VideoFrameProxy,
    ns: &str,
    label: &str,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
) -> i64 {
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace(ns.to_string())
        .label(label.to_string())
        .detection_box(RBBox::new(cx, cy, w, h, None))
        .build()
        .unwrap();
    frame
        .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
        .unwrap()
        .get_id()
}

// ---------------------------------------------------------------------------
// Callback implementations
// ---------------------------------------------------------------------------

/// Counting bypass callback.
pub struct CountingBypassCb {
    pub count: Arc<AtomicUsize>,
}

impl OnBypassFrame for CountingBypassCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(_) => {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {}
        }
    }
}

/// Counting encoded callback (VideoFrame and EndOfStream separately).
pub struct CountingEncodedCb {
    pub count: Arc<AtomicUsize>,
    pub eos_count: Arc<AtomicUsize>,
}

impl OnEncodedFrame for CountingEncodedCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::VideoFrame(_) => {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

/// Eviction callback that terminates immediately.
pub struct TerminateEviction;

impl OnEviction for TerminateEviction {
    fn call(&self, _source_id: &str) -> EvictionDecision {
        EvictionDecision::TerminateImmediately
    }
}

// ---------------------------------------------------------------------------
// GPU helpers (require cuda_init, deepstream_encoders)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn make_gpu_buffer(
    gen: &BufferGenerator,
    idx: u64,
    _dur_ns: u64,
) -> deepstream_buffers::SharedBuffer {
    gen.acquire(Some(idx as i64)).unwrap()
}

/// Creates a SurfaceView from a single-frame GPU buffer (BufferGenerator).
#[cfg(test)]
pub fn make_gpu_surface_view(
    gen: &BufferGenerator,
    idx: u64,
    dur_ns: u64,
) -> deepstream_buffers::SurfaceView {
    let buf = make_gpu_buffer(gen, idx, dur_ns);
    deepstream_buffers::SurfaceView::from_buffer(&buf, 0).unwrap()
}

/// Creates a SurfaceView from a single-frame GPU buffer (BufferGenerator).
#[cfg(test)]
pub fn make_gpu_surface_view_uniform(
    gen: &BufferGenerator,
    idx: u64,
    dur_ns: u64,
) -> deepstream_buffers::SurfaceView {
    let shared = gen.acquire(Some(idx as i64)).unwrap();
    shared.set_pts_ns(idx * dur_ns);
    shared.set_duration_ns(dur_ns);
    deepstream_buffers::SurfaceView::from_buffer(&shared, 0).unwrap()
}

#[cfg(test)]
pub fn jpeg_encoder_config(w: u32, h: u32) -> deepstream_encoders::EncoderConfig {
    use deepstream_encoders::prelude::*;
    EncoderConfig::new(Codec::Jpeg, w, h)
        .format(VideoFormat::RGBA)
        .fps(30, 1)
        .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(80) }))
}

// ---------------------------------------------------------------------------
// Platform-aware encoder helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

#[allow(dead_code)]
pub fn has_nvjpegenc() -> bool {
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

#[allow(dead_code)]
pub fn is_jetson() -> bool {
    cfg!(target_arch = "aarch64")
}

/// H264 encoder config with correct platform-specific properties.
#[allow(dead_code)]
pub fn h264_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if is_jetson() {
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::H264Jetson(H264JetsonProps {
                preset_level: Some(JetsonPresetLevel::UltraFast),
                ..Default::default()
            }))
    } else {
        EncoderConfig::new(Codec::H264, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::H264Dgpu(H264DgpuProps {
                preset: Some(DgpuPreset::P1),
                tuning_info: Some(TuningPreset::LowLatency),
                ..Default::default()
            }))
    }
}

/// HEVC encoder config with correct platform-specific properties.
#[allow(dead_code)]
pub fn hevc_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if is_jetson() {
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::HevcJetson(HevcJetsonProps {
                preset_level: Some(JetsonPresetLevel::UltraFast),
                ..Default::default()
            }))
    } else {
        EncoderConfig::new(Codec::Hevc, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::HevcDgpu(HevcDgpuProps {
                preset: Some(DgpuPreset::P1),
                tuning_info: Some(TuningPreset::LowLatency),
                ..Default::default()
            }))
    }
}

/// AV1 encoder config with correct platform-specific properties.
#[allow(dead_code)]
pub fn av1_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if is_jetson() {
        EncoderConfig::new(Codec::Av1, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::Av1Jetson(Av1JetsonProps {
                preset_level: Some(JetsonPresetLevel::UltraFast),
                ..Default::default()
            }))
    } else {
        EncoderConfig::new(Codec::Av1, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::Av1Dgpu(Av1DgpuProps {
                preset: Some(DgpuPreset::P1),
                tuning_info: Some(TuningPreset::LowLatency),
                ..Default::default()
            }))
    }
}

/// Returns best available encoder config: NVENC H264 > JPEG > PNG.
#[allow(dead_code)]
pub fn make_default_encoder_config(w: u32, h: u32) -> EncoderConfig {
    if has_nvenc() {
        h264_encoder_config(w, h)
    } else if has_nvjpegenc() {
        EncoderConfig::new(Codec::Jpeg, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(85) }))
    } else {
        EncoderConfig::new(Codec::Png, w, h)
            .format(VideoFormat::RGBA)
            .fps(30, 1)
            .properties(EncoderProperties::Png(PngProps {
                compression_level: Some(1),
            }))
    }
}
