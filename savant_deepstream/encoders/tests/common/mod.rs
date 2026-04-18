//! Shared utilities for `deepstream_encoders` integration tests.
//!
//! Mirrors `deepstream_decoders::tests::common`: initialisation,
//! capability probes, and a [`drain_to_frames`] helper that runs
//! [`NvEncoder::graceful_shutdown`] and returns the encoded frames as a
//! simple `Vec<EncodedFrame>`.

#![allow(dead_code)]

use std::time::Duration;

use deepstream_encoders::prelude::*;

// ── Initialisation / capability probes ──────────────────────────────

/// Initialize logging, GStreamer, and CUDA on GPU 0.
pub fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed — is a GPU available?");
}

pub fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

pub fn has_nvjpegenc() -> bool {
    let _ = gstreamer::init();
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

/// Returns `true` if the current kernel is a Jetson/Tegra kernel.
///
/// Checks `/proc/version` via [`nvidia_gpu_utils::is_jetson_kernel`] at
/// runtime. Do **not** use `cfg!(target_arch = "aarch64")` as a proxy:
/// aarch64 is also used by ARM servers (e.g. Grace Hopper) where Jetson
/// behaviors do not apply. Orin Nano — which is Jetson but lacks NVENC
/// — should be gated by [`has_nvenc`] instead.
pub fn is_jetson() -> bool {
    nvidia_gpu_utils::is_jetson_kernel()
}

// ── NvEncoderConfig constructors ────────────────────────────────────

/// Wrap an [`EncoderConfig`] into an [`NvEncoderConfig`] suitable for
/// tests (GPU 0, short operation timeout).
pub fn test_nv_encoder_config(encoder: EncoderConfig) -> NvEncoderConfig {
    NvEncoderConfig::new(0, encoder)
        .operation_timeout(Duration::from_secs(5))
        .name("test-encoder")
}

/// Best default encoder config: HEVC if NVENC is available, JPEG
/// otherwise, `None` when neither is present.
pub fn make_default_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(EncoderConfig::Hevc(HevcEncoderConfig::new(w, h)))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::Jpeg(
            JpegEncoderConfig::new(w, h).format(VideoFormat::I420),
        ))
    } else {
        None
    }
}

/// Same as [`make_default_config`] but with `RGBA` user format, so the
/// encoder exercises the RGBA → NV12/I420 out-of-band conversion path.
pub fn make_rgba_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(EncoderConfig::H264(
            H264EncoderConfig::new(w, h).format(VideoFormat::RGBA),
        ))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::Jpeg(
            JpegEncoderConfig::new(w, h).format(VideoFormat::RGBA),
        ))
    } else {
        None
    }
}

// ── Drain helpers ───────────────────────────────────────────────────

/// Run [`NvEncoder::graceful_shutdown`] with a 5-second idle timeout and
/// collect the encoded frames into a `Vec`.  Events (including per-source
/// EOS markers) are ignored.  Panics on pipeline errors.
pub fn drain_to_frames(encoder: &NvEncoder) -> Vec<EncodedFrame> {
    drain_to_frames_timeout(encoder, Duration::from_secs(5))
}

pub fn drain_to_frames_timeout(encoder: &NvEncoder, idle_timeout: Duration) -> Vec<EncodedFrame> {
    let mut frames = Vec::new();
    encoder
        .graceful_shutdown(Some(idle_timeout), |out| match out {
            NvEncoderOutput::Frame(f) => frames.push(f),
            NvEncoderOutput::Event(_) | NvEncoderOutput::SourceEos { .. } => {}
            NvEncoderOutput::Eos => {}
            NvEncoderOutput::Error(e) => panic!("encoder pipeline error: {e}"),
        })
        .expect("graceful_shutdown failed");
    frames
}

/// Drain up to `limit` outputs non-blockingly, returning only encoded
/// frames.
pub fn try_drain_frames(encoder: &NvEncoder, limit: usize) -> Vec<EncodedFrame> {
    let mut frames = Vec::new();
    for _ in 0..limit {
        match encoder.try_recv() {
            Ok(Some(NvEncoderOutput::Frame(f))) => frames.push(f),
            Ok(Some(NvEncoderOutput::Error(e))) => panic!("encoder error: {e}"),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => break,
        }
    }
    frames
}

/// Acquire one NVMM buffer from the encoder's pool as an owned
/// [`gstreamer::Buffer`].  Fails (panics) if the buffer has extra
/// references (should not happen in tests).
pub fn acquire_buffer(encoder: &NvEncoder, id: u128) -> gstreamer::Buffer {
    let shared = encoder
        .generator()
        .lock()
        .acquire(Some(id))
        .expect("acquire failed");
    shared.into_buffer().expect("sole owner of buffer")
}
