use crate::primitives::frame::VideoFrameTranscodingMethod;
use savant_protobuf::generated;

impl From<&VideoFrameTranscodingMethod> for generated::VideoFrameTranscodingMethod {
    fn from(value: &VideoFrameTranscodingMethod) -> Self {
        match value {
            VideoFrameTranscodingMethod::Copy => generated::VideoFrameTranscodingMethod::Copy,
            VideoFrameTranscodingMethod::Encoded => generated::VideoFrameTranscodingMethod::Encoded,
        }
    }
}

impl From<&generated::VideoFrameTranscodingMethod> for VideoFrameTranscodingMethod {
    fn from(value: &generated::VideoFrameTranscodingMethod) -> Self {
        match value {
            generated::VideoFrameTranscodingMethod::Copy => VideoFrameTranscodingMethod::Copy,
            generated::VideoFrameTranscodingMethod::Encoded => VideoFrameTranscodingMethod::Encoded,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::frame::VideoFrameTranscodingMethod;
    use savant_protobuf::generated;

    #[test]
    fn test_video_frame_transcoding_method() {
        assert_eq!(
            VideoFrameTranscodingMethod::Copy,
            VideoFrameTranscodingMethod::from(&generated::VideoFrameTranscodingMethod::Copy)
        );
        assert_eq!(
            VideoFrameTranscodingMethod::Encoded,
            VideoFrameTranscodingMethod::from(&generated::VideoFrameTranscodingMethod::Encoded)
        );
        assert_eq!(
            generated::VideoFrameTranscodingMethod::Copy,
            generated::VideoFrameTranscodingMethod::from(&VideoFrameTranscodingMethod::Copy)
        );
        assert_eq!(
            generated::VideoFrameTranscodingMethod::Encoded,
            generated::VideoFrameTranscodingMethod::from(&VideoFrameTranscodingMethod::Encoded)
        );
    }
}
