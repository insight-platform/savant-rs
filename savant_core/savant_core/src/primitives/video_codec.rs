//! Video codec identifier for [`super::frame::VideoFrame`] metadata.
//!
//! Canonical names match GStreamer / Savant conventions (`h264`, `hevc`, `jpeg`, …).
//! [`VideoCodec::SwJpeg`] is the software JPEG decode path (`"swjpeg"`).

use std::fmt;
use std::str::FromStr;

/// Encoded or raw pixel format carried on a [`super::frame::VideoFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCodec {
    H264,
    Hevc,
    /// Hardware JPEG (default `jpeg` in manifests).
    Jpeg,
    /// Software JPEG decode (`swjpeg` — not `sw_jpeg`).
    #[serde(rename = "swjpeg")]
    SwJpeg,
    Av1,
    Png,
    Vp8,
    Vp9,
    RawRgba,
    RawRgb,
    RawNv12,
}

impl VideoCodec {
    /// Parse a codec name (case-insensitive).
    ///
    /// Accepts: `h264`, `avc`, `hevc`, `h265`, `jpeg`, `swjpeg`, `av1`, `png`,
    /// `vp8`, `vp9`, `raw_rgba`, `raw_rgb`, `raw_nv12`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "h264" | "avc" => Some(Self::H264),
            "hevc" | "h265" => Some(Self::Hevc),
            "jpeg" => Some(Self::Jpeg),
            "swjpeg" => Some(Self::SwJpeg),
            "av1" => Some(Self::Av1),
            "png" => Some(Self::Png),
            "vp8" => Some(Self::Vp8),
            "vp9" => Some(Self::Vp9),
            "raw_rgba" => Some(Self::RawRgba),
            "raw_rgb" => Some(Self::RawRgb),
            "raw_nv12" => Some(Self::RawNv12),
            _ => None,
        }
    }

    /// Canonical wire / JSON name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Jpeg => "jpeg",
            Self::SwJpeg => "swjpeg",
            Self::Av1 => "av1",
            Self::Png => "png",
            Self::Vp8 => "vp8",
            Self::Vp9 => "vp9",
            Self::RawRgba => "raw_rgba",
            Self::RawRgb => "raw_rgb",
            Self::RawNv12 => "raw_nv12",
        }
    }

    /// Return the GStreamer encoder element name for this codec.
    ///
    /// Hardware encoders are returned for H.264 / HEVC / AV1 / JPEG.
    /// `Png` uses the CPU-based `pngenc`; `SwJpeg` uses the CPU-based `jpegenc`.
    /// Variants that have no GStreamer encoder (raw pixel formats, VP8, VP9)
    /// return `"identity"` so they can be plumbed through a pipeline as
    /// pass-through.
    pub const fn encoder_element(self) -> &'static str {
        match self {
            Self::H264 => "nvv4l2h264enc",
            Self::Hevc => "nvv4l2h265enc",
            Self::Jpeg => "nvjpegenc",
            Self::SwJpeg => "jpegenc",
            Self::Av1 => "nvv4l2av1enc",
            Self::Png => "pngenc",
            Self::Vp8 | Self::Vp9 | Self::RawRgba | Self::RawRgb | Self::RawNv12 => "identity",
        }
    }

    /// Return the GStreamer decoder element name for this codec.
    ///
    /// Hardware decoders are used for H.264 / HEVC / AV1 / VP8 / VP9 / JPEG.
    /// `Png` uses the CPU-based `pngdec`; `SwJpeg` uses the CPU-based `jpegdec`.
    /// Raw pixel formats return `"identity"`.
    pub const fn decoder_element(self) -> &'static str {
        match self {
            Self::H264 | Self::Hevc | Self::Vp8 | Self::Vp9 | Self::Av1 => "nvv4l2decoder",
            Self::Jpeg => "nvjpegdec",
            Self::SwJpeg => "jpegdec",
            Self::Png => "pngdec",
            Self::RawRgba | Self::RawRgb | Self::RawNv12 => "identity",
        }
    }

    /// Return the GStreamer parser element name for this codec.
    ///
    /// VP9 uses `vp9parse` (from `gst-plugins-bad`, `videoparsersbad`) on
    /// both dGPU and Jetson: it enriches `video/x-vp9` caps with
    /// `width`, `height`, `profile`, `chroma-format`, and
    /// `bit-depth-luma`/`bit-depth-chroma`, which dGPU `nvv4l2decoder`
    /// requires to avoid caps negotiation failure and which Tegra NVDEC
    /// needs to emit frames for pre-parsed super-frame-aligned packets.
    /// See `kb/decoders/caveats.md §10`.
    ///
    /// Codecs without a dedicated GStreamer parser (raw pixel formats,
    /// PNG, VP8) return `"identity"` as a pass-through. There is no
    /// `vp8parse` in upstream `gst-plugins-bad`, so VP8 intentionally
    /// stays on `identity`.
    pub const fn parser_element(self) -> &'static str {
        match self {
            Self::H264 => "h264parse",
            Self::Hevc => "h265parse",
            Self::Jpeg | Self::SwJpeg => "jpegparse",
            Self::Av1 => "av1parse",
            Self::Vp9 => "vp9parse",
            Self::Png | Self::Vp8 | Self::RawRgba | Self::RawRgb | Self::RawNv12 => "identity",
        }
    }

    /// Return the GStreamer caps string for an encoded bitstream of this codec.
    pub const fn caps_str(self) -> &'static str {
        match self {
            Self::H264 => "video/x-h264, stream-format=byte-stream",
            Self::Hevc => "video/x-h265, stream-format=byte-stream",
            Self::Jpeg | Self::SwJpeg => "image/jpeg",
            Self::Av1 => "video/x-av1",
            Self::Png => "image/png",
            Self::Vp8 => "video/x-vp8",
            Self::Vp9 => "video/x-vp9",
            Self::RawRgba => "video/x-raw,format=RGBA",
            Self::RawRgb => "video/x-raw,format=RGB",
            Self::RawNv12 => "video/x-raw,format=NV12",
        }
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for VideoCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| format!("unknown video codec: {s}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_CODECS: &[VideoCodec] = &[
        VideoCodec::H264,
        VideoCodec::Hevc,
        VideoCodec::Jpeg,
        VideoCodec::SwJpeg,
        VideoCodec::Av1,
        VideoCodec::Png,
        VideoCodec::Vp8,
        VideoCodec::Vp9,
        VideoCodec::RawRgba,
        VideoCodec::RawRgb,
        VideoCodec::RawNv12,
    ];

    #[test]
    fn from_name_roundtrip_covers_all_variants() {
        for &codec in ALL_CODECS {
            assert_eq!(VideoCodec::from_name(codec.name()), Some(codec));
        }
        assert_eq!(VideoCodec::from_name("H264"), Some(VideoCodec::H264));
        assert_eq!(VideoCodec::from_name("h265"), Some(VideoCodec::Hevc));
        assert_eq!(VideoCodec::from_name("avc"), Some(VideoCodec::H264));
        assert_eq!(VideoCodec::from_name(""), None);
    }

    #[test]
    fn encoder_element_mapping() {
        assert_eq!(VideoCodec::H264.encoder_element(), "nvv4l2h264enc");
        assert_eq!(VideoCodec::Hevc.encoder_element(), "nvv4l2h265enc");
        assert_eq!(VideoCodec::Jpeg.encoder_element(), "nvjpegenc");
        assert_eq!(VideoCodec::SwJpeg.encoder_element(), "jpegenc");
        assert_eq!(VideoCodec::Av1.encoder_element(), "nvv4l2av1enc");
        assert_eq!(VideoCodec::Png.encoder_element(), "pngenc");
        assert_eq!(VideoCodec::Vp8.encoder_element(), "identity");
        assert_eq!(VideoCodec::Vp9.encoder_element(), "identity");
        assert_eq!(VideoCodec::RawRgba.encoder_element(), "identity");
        assert_eq!(VideoCodec::RawRgb.encoder_element(), "identity");
        assert_eq!(VideoCodec::RawNv12.encoder_element(), "identity");
    }

    #[test]
    fn decoder_element_mapping() {
        assert_eq!(VideoCodec::H264.decoder_element(), "nvv4l2decoder");
        assert_eq!(VideoCodec::Hevc.decoder_element(), "nvv4l2decoder");
        assert_eq!(VideoCodec::Vp8.decoder_element(), "nvv4l2decoder");
        assert_eq!(VideoCodec::Vp9.decoder_element(), "nvv4l2decoder");
        assert_eq!(VideoCodec::Av1.decoder_element(), "nvv4l2decoder");
        assert_eq!(VideoCodec::Jpeg.decoder_element(), "nvjpegdec");
        assert_eq!(VideoCodec::SwJpeg.decoder_element(), "jpegdec");
        assert_eq!(VideoCodec::Png.decoder_element(), "pngdec");
        assert_eq!(VideoCodec::RawRgba.decoder_element(), "identity");
        assert_eq!(VideoCodec::RawRgb.decoder_element(), "identity");
        assert_eq!(VideoCodec::RawNv12.decoder_element(), "identity");
    }

    #[test]
    fn parser_element_mapping() {
        assert_eq!(VideoCodec::H264.parser_element(), "h264parse");
        assert_eq!(VideoCodec::Hevc.parser_element(), "h265parse");
        assert_eq!(VideoCodec::Jpeg.parser_element(), "jpegparse");
        assert_eq!(VideoCodec::SwJpeg.parser_element(), "jpegparse");
        assert_eq!(VideoCodec::Av1.parser_element(), "av1parse");
        assert_eq!(VideoCodec::Vp9.parser_element(), "vp9parse");
        for &codec in &[
            VideoCodec::Png,
            VideoCodec::Vp8,
            VideoCodec::RawRgba,
            VideoCodec::RawRgb,
            VideoCodec::RawNv12,
        ] {
            assert_eq!(codec.parser_element(), "identity");
        }
    }

    #[test]
    fn caps_str_mapping() {
        assert!(VideoCodec::H264.caps_str().contains("h264"));
        assert!(VideoCodec::Hevc.caps_str().contains("h265"));
        assert_eq!(VideoCodec::Jpeg.caps_str(), "image/jpeg");
        assert_eq!(VideoCodec::SwJpeg.caps_str(), "image/jpeg");
        assert_eq!(VideoCodec::Av1.caps_str(), "video/x-av1");
        assert_eq!(VideoCodec::Png.caps_str(), "image/png");
        assert_eq!(VideoCodec::Vp8.caps_str(), "video/x-vp8");
        assert_eq!(VideoCodec::Vp9.caps_str(), "video/x-vp9");
        assert_eq!(VideoCodec::RawRgba.caps_str(), "video/x-raw,format=RGBA");
        assert_eq!(VideoCodec::RawRgb.caps_str(), "video/x-raw,format=RGB");
        assert_eq!(VideoCodec::RawNv12.caps_str(), "video/x-raw,format=NV12");
    }

    #[test]
    fn display_matches_name() {
        for &codec in ALL_CODECS {
            assert_eq!(format!("{codec}"), codec.name());
        }
    }
}
