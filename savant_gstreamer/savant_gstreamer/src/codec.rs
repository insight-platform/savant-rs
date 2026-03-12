//! Supported video codecs and their GStreamer element mappings.

/// Supported video codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// JPEG (Motion JPEG).
    Jpeg,
    /// AV1.
    Av1,
    /// PNG (CPU-based, lossless).
    Png,
    /// Raw RGBA pixel data (GPU-to-CPU download, no encoding).
    RawRgba,
    /// Raw RGB pixel data (GPU-to-CPU download, no encoding).
    RawRgb,
}

impl Codec {
    /// Return the GStreamer encoder element name for this codec.
    ///
    /// For PNG, returns `pngenc` (CPU-based). Other codecs use NVIDIA
    /// hardware encoders.
    pub fn encoder_element(&self) -> &'static str {
        match self {
            Codec::H264 => "nvv4l2h264enc",
            Codec::Hevc => "nvv4l2h265enc",
            Codec::Jpeg => "nvjpegenc",
            Codec::Av1 => "nvv4l2av1enc",
            Codec::Png => "pngenc",
            Codec::RawRgba | Codec::RawRgb => "identity",
        }
    }

    /// Return the GStreamer parser element name for this codec.
    ///
    /// PNG has no parser; returns `identity` as a pass-through.
    pub fn parser_element(&self) -> &'static str {
        match self {
            Codec::H264 => "h264parse",
            Codec::Hevc => "h265parse",
            Codec::Jpeg => "jpegparse",
            Codec::Av1 => "av1parse",
            Codec::Png | Codec::RawRgba | Codec::RawRgb => "identity",
        }
    }

    /// Return the GStreamer caps string for encoded bitstream of this codec.
    pub fn caps_str(&self) -> &'static str {
        match self {
            Codec::H264 => "video/x-h264, stream-format=byte-stream",
            Codec::Hevc => "video/x-h265, stream-format=byte-stream",
            Codec::Jpeg => "image/jpeg",
            Codec::Av1 => "video/x-av1",
            Codec::Png => "image/png",
            Codec::RawRgba => "video/x-raw,format=RGBA",
            Codec::RawRgb => "video/x-raw,format=RGB",
        }
    }

    /// Parse a codec from a string name.
    ///
    /// Accepted names (case-insensitive): `h264`, `hevc`, `h265`, `jpeg`, `av1`, `png`,
    /// `raw_rgba`, `raw_rgb`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "h264" => Some(Codec::H264),
            "hevc" | "h265" => Some(Codec::Hevc),
            "jpeg" => Some(Codec::Jpeg),
            "av1" => Some(Codec::Av1),
            "png" => Some(Codec::Png),
            "raw_rgba" => Some(Codec::RawRgba),
            "raw_rgb" => Some(Codec::RawRgb),
            _ => None,
        }
    }

    /// Return the canonical name of this codec.
    pub fn name(&self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Jpeg => "jpeg",
            Codec::Av1 => "av1",
            Codec::Png => "png",
            Codec::RawRgba => "raw_rgba",
            Codec::RawRgb => "raw_rgb",
        }
    }
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name() {
        assert_eq!(Codec::from_name("h264"), Some(Codec::H264));
        assert_eq!(Codec::from_name("H264"), Some(Codec::H264));
        assert_eq!(Codec::from_name("hevc"), Some(Codec::Hevc));
        assert_eq!(Codec::from_name("h265"), Some(Codec::Hevc));
        assert_eq!(Codec::from_name("jpeg"), Some(Codec::Jpeg));
        assert_eq!(Codec::from_name("av1"), Some(Codec::Av1));
        assert_eq!(Codec::from_name("png"), Some(Codec::Png));
        assert_eq!(Codec::from_name("PNG"), Some(Codec::Png));
        assert_eq!(Codec::from_name("raw_rgba"), Some(Codec::RawRgba));
        assert_eq!(Codec::from_name("RAW_RGBA"), Some(Codec::RawRgba));
        assert_eq!(Codec::from_name("raw_rgb"), Some(Codec::RawRgb));
        assert_eq!(Codec::from_name("RAW_RGB"), Some(Codec::RawRgb));
        assert_eq!(Codec::from_name("vp9"), None);
        assert_eq!(Codec::from_name(""), None);
    }

    #[test]
    fn test_names() {
        assert_eq!(Codec::H264.name(), "h264");
        assert_eq!(Codec::Hevc.name(), "hevc");
        assert_eq!(Codec::Jpeg.name(), "jpeg");
        assert_eq!(Codec::Av1.name(), "av1");
        assert_eq!(Codec::Png.name(), "png");
        assert_eq!(Codec::RawRgba.name(), "raw_rgba");
        assert_eq!(Codec::RawRgb.name(), "raw_rgb");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Codec::Hevc), "hevc");
    }

    #[test]
    fn test_caps_str() {
        assert!(Codec::H264.caps_str().contains("h264"));
        assert!(Codec::Hevc.caps_str().contains("h265"));
        assert!(Codec::Jpeg.caps_str().contains("jpeg"));
        assert!(Codec::Av1.caps_str().contains("av1"));
        assert!(Codec::Png.caps_str().contains("png"));
        assert_eq!(Codec::RawRgba.caps_str(), "video/x-raw,format=RGBA");
        assert_eq!(Codec::RawRgb.caps_str(), "video/x-raw,format=RGB");
    }

    #[test]
    fn test_parser_element() {
        assert_eq!(Codec::H264.parser_element(), "h264parse");
        assert_eq!(Codec::Hevc.parser_element(), "h265parse");
        assert_eq!(Codec::Jpeg.parser_element(), "jpegparse");
        assert_eq!(Codec::Av1.parser_element(), "av1parse");
        assert_eq!(Codec::Png.parser_element(), "identity");
        assert_eq!(Codec::RawRgba.parser_element(), "identity");
        assert_eq!(Codec::RawRgb.parser_element(), "identity");
    }

    #[test]
    fn test_encoder_element() {
        assert_eq!(Codec::H264.encoder_element(), "nvv4l2h264enc");
        assert_eq!(Codec::Hevc.encoder_element(), "nvv4l2h265enc");
        assert_eq!(Codec::Jpeg.encoder_element(), "nvjpegenc");
        assert_eq!(Codec::Av1.encoder_element(), "nvv4l2av1enc");
        assert_eq!(Codec::Png.encoder_element(), "pngenc");
        assert_eq!(Codec::RawRgba.encoder_element(), "identity");
        assert_eq!(Codec::RawRgb.encoder_element(), "identity");
    }
}
