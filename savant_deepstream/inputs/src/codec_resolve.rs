//! Map `VideoFrame` codec strings to `DecoderConfig` (and H.264/HEVC detection helpers).

#[allow(unused_imports)]
pub use deepstream_decoders::{detect_stream_config, is_random_access_point};
use deepstream_decoders::{
    Av1DecoderConfig, DecoderConfig, JpegDecoderConfig, PngDecoderConfig, RawRgbDecoderConfig,
    RawRgbaDecoderConfig, Vp8DecoderConfig, Vp9DecoderConfig,
};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::Codec;

/// Result of resolving a codec name before bitstream inspection.
#[derive(Debug, Clone)]
pub enum CodecResolve {
    /// Need RAP + `detect_stream_config` on first decodable AU.
    NeedDetection { codec: Codec },
    /// Ready to build `NvDecoder` immediately.
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
        VideoCodec::H264 => Ok(CodecResolve::NeedDetection { codec: Codec::H264 }),
        VideoCodec::Hevc => Ok(CodecResolve::NeedDetection { codec: Codec::Hevc }),
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
            CodecResolve::NeedDetection { codec } => assert_eq!(codec, Codec::H264),
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
