use crate::primitives::frame::VideoFrameContent;
use crate::protobuf::generated;

impl From<&VideoFrameContent> for generated::video_frame::Content {
    fn from(value: &VideoFrameContent) -> Self {
        match value {
            VideoFrameContent::External(e) => {
                generated::video_frame::Content::External(generated::ExternalFrame {
                    method: e.method.clone(),
                    location: e.location.clone(),
                })
            }
            VideoFrameContent::Internal(data) => {
                generated::video_frame::Content::Internal(data.clone())
            }
            VideoFrameContent::None => {
                generated::video_frame::Content::None(generated::NoneFrame {})
            }
        }
    }
}

impl From<&generated::video_frame::Content> for VideoFrameContent {
    fn from(value: &generated::video_frame::Content) -> Self {
        match value {
            generated::video_frame::Content::External(e) => {
                VideoFrameContent::External(crate::primitives::frame::ExternalFrame {
                    method: e.method.clone(),
                    location: e.location.clone(),
                })
            }
            generated::video_frame::Content::Internal(data) => {
                VideoFrameContent::Internal(data.clone())
            }
            generated::video_frame::Content::None(_) => VideoFrameContent::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::frame::VideoFrameContent;
    use crate::protobuf::generated;

    #[test]
    fn test_video_frame_content() {
        assert_eq!(
            VideoFrameContent::External(crate::primitives::frame::ExternalFrame {
                method: "method".to_string(),
                location: Some("location".to_string()),
            }),
            VideoFrameContent::from(&generated::video_frame::Content::External(
                generated::ExternalFrame {
                    method: "method".to_string(),
                    location: Some("location".to_string()),
                }
            ))
        );
        assert_eq!(
            VideoFrameContent::Internal(vec![1, 2, 3]),
            VideoFrameContent::from(&generated::video_frame::Content::Internal(vec![1, 2, 3]))
        );
        assert_eq!(
            VideoFrameContent::None,
            VideoFrameContent::from(&generated::video_frame::Content::None(
                generated::NoneFrame {}
            ))
        );

        assert_eq!(
            generated::video_frame::Content::External(generated::ExternalFrame {
                method: "method".to_string(),
                location: Some("location".to_string()),
            }),
            generated::video_frame::Content::from(&VideoFrameContent::External(
                crate::primitives::frame::ExternalFrame {
                    method: "method".to_string(),
                    location: Some("location".to_string()),
                }
            ))
        );

        assert_eq!(
            generated::video_frame::Content::Internal(vec![1, 2, 3]),
            generated::video_frame::Content::from(&VideoFrameContent::Internal(vec![1, 2, 3]))
        );

        assert_eq!(
            generated::video_frame::Content::None(generated::NoneFrame {}),
            generated::video_frame::Content::from(&VideoFrameContent::None)
        );
    }
}
