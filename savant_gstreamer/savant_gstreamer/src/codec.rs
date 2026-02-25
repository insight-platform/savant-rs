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
}

impl Codec {
    /// Return the GStreamer encoder element name for this codec.
    pub fn encoder_element(&self) -> &'static str {
        match self {
            Codec::H264 => "nvv4l2h264enc",
            Codec::Hevc => "nvv4l2h265enc",
            Codec::Jpeg => "nvjpegenc",
            Codec::Av1 => "nvv4l2av1enc",
        }
    }

    /// Return the GStreamer parser element name for this codec.
    pub fn parser_element(&self) -> &'static str {
        match self {
            Codec::H264 => "h264parse",
            Codec::Hevc => "h265parse",
            Codec::Jpeg => "jpegparse",
            Codec::Av1 => "av1parse",
        }
    }

    /// Return the GStreamer caps string for encoded bitstream of this codec.
    pub fn caps_str(&self) -> &'static str {
        match self {
            Codec::H264 => "video/x-h264, stream-format=byte-stream",
            Codec::Hevc => "video/x-h265, stream-format=byte-stream",
            Codec::Jpeg => "image/jpeg",
            Codec::Av1 => "video/x-av1",
        }
    }

    /// Parse a codec from a string name.
    ///
    /// Accepted names (case-insensitive): `h264`, `hevc`, `h265`, `jpeg`, `av1`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "h264" => Some(Codec::H264),
            "hevc" | "h265" => Some(Codec::Hevc),
            "jpeg" => Some(Codec::Jpeg),
            "av1" => Some(Codec::Av1),
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
        assert_eq!(Codec::from_name("vp9"), None);
        assert_eq!(Codec::from_name(""), None);
    }

    #[test]
    fn test_names() {
        assert_eq!(Codec::H264.name(), "h264");
        assert_eq!(Codec::Hevc.name(), "hevc");
        assert_eq!(Codec::Jpeg.name(), "jpeg");
        assert_eq!(Codec::Av1.name(), "av1");
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
    }

    #[test]
    fn test_parser_element() {
        assert_eq!(Codec::H264.parser_element(), "h264parse");
        assert_eq!(Codec::Hevc.parser_element(), "h265parse");
        assert_eq!(Codec::Jpeg.parser_element(), "jpegparse");
        assert_eq!(Codec::Av1.parser_element(), "av1parse");
    }

    #[test]
    fn test_encoder_element() {
        assert_eq!(Codec::H264.encoder_element(), "nvv4l2h264enc");
        assert_eq!(Codec::Hevc.encoder_element(), "nvv4l2h265enc");
        assert_eq!(Codec::Jpeg.encoder_element(), "nvjpegenc");
        assert_eq!(Codec::Av1.encoder_element(), "nvv4l2av1enc");
    }
}
