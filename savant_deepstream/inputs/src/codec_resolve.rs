//! Map `VideoFrame` codec strings to `DecoderConfig` (and H.264/HEVC detection helpers).

use deepstream_decoders::{
    Av1DecoderConfig, DecoderConfig, JpegDecoderConfig, PngDecoderConfig, RawRgbDecoderConfig,
    RawRgbaDecoderConfig, Vp8DecoderConfig, Vp9DecoderConfig,
};
use savant_core::primitives::video_codec::VideoCodec;

/// Encapsulates the two-phase detection logic for codecs whose
/// [`DecoderConfig`] depends on bitstream parameters (SPS/PPS).
///
/// Consumers call [`is_random_access_point`](Self::is_random_access_point) to
/// check whether a packet can seed the detector, then
/// [`detect_config`](Self::detect_config) to extract the actual config.
#[derive(Debug, Clone, Copy)]
pub struct DetectionStrategy {
    codec: VideoCodec,
}

impl DetectionStrategy {
    /// Strategy for H.264 (AVC) bitstreams.
    pub fn h264() -> Self {
        Self {
            codec: VideoCodec::H264,
        }
    }

    /// Strategy for H.265 (HEVC) bitstreams.
    pub fn hevc() -> Self {
        Self {
            codec: VideoCodec::Hevc,
        }
    }

    /// The underlying GStreamer codec tag.
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Returns `true` when the access unit is a valid random-access point from
    /// which a decoder can start producing pictures (e.g. contains
    /// SPS + PPS + IDR for H.264).
    pub fn is_random_access_point(&self, data: &[u8]) -> bool {
        deepstream_decoders::is_random_access_point(self.codec, data)
    }

    /// Inspect the keyframe's NALUs and return a complete [`DecoderConfig`], or
    /// `None` if detection fails (e.g. missing parameter sets).
    pub fn detect_config(&self, data: &[u8]) -> Option<DecoderConfig> {
        deepstream_decoders::detect_stream_config(self.codec, data)
    }
}

/// Result of resolving a [`VideoCodec`] to a decoder configuration.
///
/// Most codecs produce [`Ready`](Self::Ready) with a complete [`DecoderConfig`]
/// immediately. H.264 and HEVC require a two-phase flow because the
/// `DecoderConfig` depends on bitstream parameters (profile, level, chroma
/// format) extracted from the first keyframe's SPS/PPS — see
/// [`DetectionStrategy`] for that path.
///
/// [`NvDecoder`]: deepstream_decoders::NvDecoder
#[derive(Debug, Clone)]
pub enum CodecResolve {
    /// Bitstream inspection required. Call the enclosed
    /// [`DetectionStrategy`] methods to determine the keyframe and
    /// extract the [`DecoderConfig`].
    NeedDetection(DetectionStrategy),
    /// Config is fully known — decoder can be created immediately.
    Ready(DecoderConfig),
}

/// Map [`VideoCodec`] from `VideoFrameProxy::get_codec()` to a resolve plan.
///
/// `(width, height)` are needed so raw-pixel codecs (`raw_rgba`, `raw_rgb`)
/// can build a complete [`DecoderConfig`] in one step.
///
/// Returns `Err(None)` when `codec` is absent, or `Err(Some(name))` for unsupported codecs.
pub fn resolve_video_codec(
    codec: Option<VideoCodec>,
    width: u32,
    height: u32,
) -> Result<CodecResolve, Option<String>> {
    let c = match codec {
        None => return Err(None),
        Some(c) => c,
    };
    match c {
        VideoCodec::H264 => Ok(CodecResolve::NeedDetection(DetectionStrategy::h264())),
        VideoCodec::Hevc => Ok(CodecResolve::NeedDetection(DetectionStrategy::hevc())),
        VideoCodec::Jpeg => Ok(CodecResolve::Ready(DecoderConfig::Jpeg(
            JpegDecoderConfig::gpu(),
        ))),
        VideoCodec::SwJpeg => Ok(CodecResolve::Ready(DecoderConfig::Jpeg(
            JpegDecoderConfig::cpu(),
        ))),
        VideoCodec::Png => Ok(CodecResolve::Ready(DecoderConfig::Png(PngDecoderConfig))),
        VideoCodec::Vp8 => Ok(CodecResolve::Ready(DecoderConfig::Vp8(
            Vp8DecoderConfig::default(),
        ))),
        VideoCodec::Vp9 => Ok(CodecResolve::Ready(DecoderConfig::Vp9(
            Vp9DecoderConfig::default(),
        ))),
        VideoCodec::Av1 => Ok(CodecResolve::Ready(DecoderConfig::Av1(
            Av1DecoderConfig::default(),
        ))),
        VideoCodec::RawRgba => Ok(CodecResolve::Ready(DecoderConfig::RawRgba(
            RawRgbaDecoderConfig::new(width, height),
        ))),
        VideoCodec::RawRgb => Ok(CodecResolve::Ready(DecoderConfig::RawRgb(
            RawRgbDecoderConfig::new(width, height),
        ))),
        VideoCodec::RawNv12 => Err(Some(c.name().to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(name: &str, width: u32, height: u32) -> Result<CodecResolve, Option<String>> {
        match VideoCodec::from_name(name) {
            None => Err(Some(name.to_string())),
            Some(vc) => resolve_video_codec(Some(vc), width, height),
        }
    }

    #[test]
    fn swjpeg_is_cpu_jpeg() {
        let r = resolve("swjpeg", 0, 0).unwrap();
        match r {
            CodecResolve::Ready(DecoderConfig::Jpeg(j)) => {
                assert_eq!(j.backend, deepstream_decoders::JpegBackend::Cpu);
            }
            _ => panic!("expected swjpeg -> Jpeg cpu"),
        }
    }

    #[test]
    fn jpeg_is_gpu() {
        let r = resolve("jpeg", 0, 0).unwrap();
        match r {
            CodecResolve::Ready(DecoderConfig::Jpeg(j)) => {
                assert_eq!(j.backend, deepstream_decoders::JpegBackend::Gpu);
            }
            _ => panic!("expected jpeg gpu"),
        }
    }

    #[test]
    fn h264_needs_detection() {
        match resolve("h264", 0, 0).unwrap() {
            CodecResolve::NeedDetection(strategy) => assert_eq!(strategy.codec(), VideoCodec::H264),
            _ => panic!(),
        }
    }

    #[test]
    fn hevc_needs_detection() {
        match resolve("hevc", 0, 0).unwrap() {
            CodecResolve::NeedDetection(strategy) => assert_eq!(strategy.codec(), VideoCodec::Hevc),
            _ => panic!(),
        }
    }

    #[test]
    fn raw_rgba_ready() {
        match resolve("raw_rgba", 640, 480).unwrap() {
            CodecResolve::Ready(DecoderConfig::RawRgba(c)) => {
                assert_eq!(c.width, 640);
                assert_eq!(c.height, 480);
            }
            other => panic!("expected RawRgba, got: {other:?}"),
        }
    }

    #[test]
    fn raw_rgb_ready() {
        match resolve("raw_rgb", 320, 240).unwrap() {
            CodecResolve::Ready(DecoderConfig::RawRgb(c)) => {
                assert_eq!(c.width, 320);
                assert_eq!(c.height, 240);
            }
            other => panic!("expected RawRgb, got: {other:?}"),
        }
    }

    #[test]
    fn raw_nv12_unsupported() {
        assert!(resolve("raw_nv12", 640, 480).is_err());
    }

    #[test]
    fn none_codec() {
        assert!(resolve_video_codec(None, 0, 0).is_err());
    }

    #[test]
    fn empty_codec() {
        // empty string is not a valid codec name
        assert!(VideoCodec::from_name("").is_none());
    }

    #[test]
    fn unknown_codec() {
        let r = resolve("webm_opus", 0, 0);
        match r {
            Err(Some(s)) => assert_eq!(s, "webm_opus"),
            _ => panic!("expected Err(Some)"),
        }
    }
}
