//! Supported video pixel formats and their GStreamer string mappings.

/// Common video pixel formats used with NvBufSurface and GStreamer.
///
/// Each variant maps to the corresponding GStreamer `video/x-raw` format
/// string.  The enum covers the formats most commonly used in DeepStream /
/// NVMM pipelines; for uncommon formats you can construct one from a
/// string via [`VideoFormat::from_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoFormat {
    /// 8-bit RGBA (4 bytes/pixel). Common for rendering (Skia, OpenGL).
    RGBA,
    /// 8-bit BGRx (4 bytes/pixel, alpha ignored).
    BGRx,
    /// 8-bit NV12 (YUV 4:2:0 semi-planar). Default encoder-native format.
    NV12,
    /// 8-bit NV21 (YUV 4:2:0 semi-planar, UV swapped).
    NV21,
    /// 8-bit I420 (YUV 4:2:0 planar). JPEG encoder-native format.
    I420,
    /// 8-bit UYVY (YUV 4:2:2 packed).
    UYVY,
    /// 8-bit GRAY8 (single-channel grayscale).
    GRAY8,
}

impl VideoFormat {
    /// Return the GStreamer format string for this format.
    ///
    /// This is the string used in `video/x-raw` caps, e.g. `"RGBA"`,
    /// `"NV12"`.
    pub fn gst_name(&self) -> &'static str {
        match self {
            VideoFormat::RGBA => "RGBA",
            VideoFormat::BGRx => "BGRx",
            VideoFormat::NV12 => "NV12",
            VideoFormat::NV21 => "NV21",
            VideoFormat::I420 => "I420",
            VideoFormat::UYVY => "UYVY",
            VideoFormat::GRAY8 => "GRAY8",
        }
    }

    /// Parse a video format from a string name.
    ///
    /// Accepted names (case-sensitive, matching GStreamer conventions):
    /// `RGBA`, `BGRx`, `NV12`, `NV21`, `I420`, `UYVY`, `GRAY8`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "RGBA" => Some(VideoFormat::RGBA),
            "BGRx" => Some(VideoFormat::BGRx),
            "NV12" => Some(VideoFormat::NV12),
            "NV21" => Some(VideoFormat::NV21),
            "I420" => Some(VideoFormat::I420),
            "UYVY" => Some(VideoFormat::UYVY),
            "GRAY8" => Some(VideoFormat::GRAY8),
            _ => None,
        }
    }

    /// Return the canonical name of this format (same as [`gst_name`](Self::gst_name)).
    pub fn name(&self) -> &'static str {
        self.gst_name()
    }
}

impl std::fmt::Display for VideoFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.gst_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name() {
        assert_eq!(VideoFormat::from_name("RGBA"), Some(VideoFormat::RGBA));
        assert_eq!(VideoFormat::from_name("BGRx"), Some(VideoFormat::BGRx));
        assert_eq!(VideoFormat::from_name("NV12"), Some(VideoFormat::NV12));
        assert_eq!(VideoFormat::from_name("NV21"), Some(VideoFormat::NV21));
        assert_eq!(VideoFormat::from_name("I420"), Some(VideoFormat::I420));
        assert_eq!(VideoFormat::from_name("UYVY"), Some(VideoFormat::UYVY));
        assert_eq!(VideoFormat::from_name("GRAY8"), Some(VideoFormat::GRAY8));
        assert_eq!(VideoFormat::from_name("nv12"), None); // case-sensitive
        assert_eq!(VideoFormat::from_name(""), None);
    }

    #[test]
    fn test_gst_name() {
        assert_eq!(VideoFormat::RGBA.gst_name(), "RGBA");
        assert_eq!(VideoFormat::NV12.gst_name(), "NV12");
        assert_eq!(VideoFormat::I420.gst_name(), "I420");
        assert_eq!(VideoFormat::BGRx.gst_name(), "BGRx");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", VideoFormat::RGBA), "RGBA");
        assert_eq!(format!("{}", VideoFormat::NV12), "NV12");
    }

    #[test]
    fn test_roundtrip() {
        for fmt in &[
            VideoFormat::RGBA,
            VideoFormat::BGRx,
            VideoFormat::NV12,
            VideoFormat::NV21,
            VideoFormat::I420,
            VideoFormat::UYVY,
            VideoFormat::GRAY8,
        ] {
            assert_eq!(VideoFormat::from_name(fmt.gst_name()), Some(*fmt));
        }
    }
}
