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
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// Raw RGBA pixel data (GPU-to-CPU download, no encoding).
    RawRgba,
    /// Raw RGB pixel data (GPU-to-CPU download, no encoding).
    RawRgb,
    /// Raw NV12 pixel data.
    RawNv12,
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
            Codec::Vp8 | Codec::Vp9 | Codec::RawRgba | Codec::RawRgb | Codec::RawNv12 => "identity",
        }
    }

    /// Return the GStreamer decoder element name for this codec.
    pub fn decoder_element(&self) -> &'static str {
        match self {
            Codec::H264 | Codec::Hevc | Codec::Vp8 | Codec::Vp9 | Codec::Av1 => "nvv4l2decoder",
            Codec::Jpeg => "nvjpegdec",
            Codec::Png => "pngdec",
            Codec::RawRgba | Codec::RawRgb | Codec::RawNv12 => "identity",
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
            Codec::Png
            | Codec::Vp8
            | Codec::Vp9
            | Codec::RawRgba
            | Codec::RawRgb
            | Codec::RawNv12 => "identity",
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
            Codec::Vp8 => "video/x-vp8",
            Codec::Vp9 => "video/x-vp9",
            Codec::RawRgba => "video/x-raw,format=RGBA",
            Codec::RawRgb => "video/x-raw,format=RGB",
            Codec::RawNv12 => "video/x-raw,format=NV12",
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
            "vp8" => Some(Codec::Vp8),
            "vp9" => Some(Codec::Vp9),
            "raw_rgba" => Some(Codec::RawRgba),
            "raw_rgb" => Some(Codec::RawRgb),
            "raw_nv12" => Some(Codec::RawNv12),
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
            Codec::Vp8 => "vp8",
            Codec::Vp9 => "vp9",
            Codec::RawRgba => "raw_rgba",
            Codec::RawRgb => "raw_rgb",
            Codec::RawNv12 => "raw_nv12",
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
        assert_eq!(Codec::from_name("vp8"), Some(Codec::Vp8));
        assert_eq!(Codec::from_name("VP8"), Some(Codec::Vp8));
        assert_eq!(Codec::from_name("vp9"), Some(Codec::Vp9));
        assert_eq!(Codec::from_name("VP9"), Some(Codec::Vp9));
        assert_eq!(Codec::from_name("raw_nv12"), Some(Codec::RawNv12));
        assert_eq!(Codec::from_name("RAW_NV12"), Some(Codec::RawNv12));
        assert_eq!(Codec::from_name(""), None);
    }

    #[test]
    fn test_names() {
        assert_eq!(Codec::H264.name(), "h264");
        assert_eq!(Codec::Hevc.name(), "hevc");
        assert_eq!(Codec::Jpeg.name(), "jpeg");
        assert_eq!(Codec::Av1.name(), "av1");
        assert_eq!(Codec::Png.name(), "png");
        assert_eq!(Codec::Vp8.name(), "vp8");
        assert_eq!(Codec::Vp9.name(), "vp9");
        assert_eq!(Codec::RawRgba.name(), "raw_rgba");
        assert_eq!(Codec::RawRgb.name(), "raw_rgb");
        assert_eq!(Codec::RawNv12.name(), "raw_nv12");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Codec::Hevc), "hevc");
        assert_eq!(format!("{}", Codec::Vp8), "vp8");
        assert_eq!(format!("{}", Codec::Vp9), "vp9");
        assert_eq!(format!("{}", Codec::RawNv12), "raw_nv12");
    }

    #[test]
    fn test_caps_str() {
        assert!(Codec::H264.caps_str().contains("h264"));
        assert!(Codec::Hevc.caps_str().contains("h265"));
        assert!(Codec::Jpeg.caps_str().contains("jpeg"));
        assert!(Codec::Av1.caps_str().contains("av1"));
        assert!(Codec::Png.caps_str().contains("png"));
        assert_eq!(Codec::Vp8.caps_str(), "video/x-vp8");
        assert_eq!(Codec::Vp9.caps_str(), "video/x-vp9");
        assert_eq!(Codec::RawRgba.caps_str(), "video/x-raw,format=RGBA");
        assert_eq!(Codec::RawRgb.caps_str(), "video/x-raw,format=RGB");
        assert_eq!(Codec::RawNv12.caps_str(), "video/x-raw,format=NV12");
    }

    #[test]
    fn test_parser_element() {
        assert_eq!(Codec::H264.parser_element(), "h264parse");
        assert_eq!(Codec::Hevc.parser_element(), "h265parse");
        assert_eq!(Codec::Jpeg.parser_element(), "jpegparse");
        assert_eq!(Codec::Av1.parser_element(), "av1parse");
        assert_eq!(Codec::Png.parser_element(), "identity");
        assert_eq!(Codec::Vp8.parser_element(), "identity");
        assert_eq!(Codec::Vp9.parser_element(), "identity");
        assert_eq!(Codec::RawRgba.parser_element(), "identity");
        assert_eq!(Codec::RawRgb.parser_element(), "identity");
        assert_eq!(Codec::RawNv12.parser_element(), "identity");
    }

    #[test]
    fn test_encoder_element() {
        assert_eq!(Codec::H264.encoder_element(), "nvv4l2h264enc");
        assert_eq!(Codec::Hevc.encoder_element(), "nvv4l2h265enc");
        assert_eq!(Codec::Jpeg.encoder_element(), "nvjpegenc");
        assert_eq!(Codec::Av1.encoder_element(), "nvv4l2av1enc");
        assert_eq!(Codec::Png.encoder_element(), "pngenc");
        assert_eq!(Codec::Vp8.encoder_element(), "identity");
        assert_eq!(Codec::Vp9.encoder_element(), "identity");
        assert_eq!(Codec::RawRgba.encoder_element(), "identity");
        assert_eq!(Codec::RawRgb.encoder_element(), "identity");
        assert_eq!(Codec::RawNv12.encoder_element(), "identity");
    }

    #[test]
    fn test_decoder_element() {
        assert_eq!(Codec::H264.decoder_element(), "nvv4l2decoder");
        assert_eq!(Codec::Hevc.decoder_element(), "nvv4l2decoder");
        assert_eq!(Codec::Vp8.decoder_element(), "nvv4l2decoder");
        assert_eq!(Codec::Vp9.decoder_element(), "nvv4l2decoder");
        assert_eq!(Codec::Av1.decoder_element(), "nvv4l2decoder");
        assert_eq!(Codec::Jpeg.decoder_element(), "nvjpegdec");
        assert_eq!(Codec::Png.decoder_element(), "pngdec");
        assert_eq!(Codec::RawRgba.decoder_element(), "identity");
        assert_eq!(Codec::RawRgb.decoder_element(), "identity");
        assert_eq!(Codec::RawNv12.decoder_element(), "identity");
    }
}
