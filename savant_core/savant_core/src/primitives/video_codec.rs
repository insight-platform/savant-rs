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

    /// Canonical wire / JSON name (matches [`savant_gstreamer::Codec::name`] where both exist).
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
